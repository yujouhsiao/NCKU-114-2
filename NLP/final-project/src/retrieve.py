"""Stage 5: Hybrid retrieval with optional reranking.

retrieve(query) → list[Chunk]

Flow:
  dense_search   → top TOP_N by cosine (Chroma)
  sparse_search  → top TOP_N by BM25
  rrf_fuse       → merge with Reciprocal Rank Fusion
  rerank         → BGE reranker cross-encoder
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.schemas import Chunk


# ── Lazy singletons ──────────────────────────────────────────────────────────

_embedder    = None
_chroma_col  = None
_bm25_data   = None   # {"bm25": BM25Okapi, "chunk_ids": [...], "texts": [...]}
_reranker    = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from src.embed import get_embedder
        _embedder = get_embedder()
    return _embedder


def _get_collection():
    global _chroma_col
    if _chroma_col is None:
        import chromadb
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        _chroma_col = client.get_collection(config.CHROMA_COLLECTION)
    return _chroma_col


def _get_bm25():
    global _bm25_data
    if _bm25_data is None:
        bm25_path = Path(config.BM25_PATH)
        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {config.BM25_PATH}. Run build_index.py.")
        with open(bm25_path, "rb") as f:
            _bm25_data = pickle.load(f)
    return _bm25_data


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        from src.embed import _pick_device
        device = _pick_device()
        print(f"Loading reranker: {config.RERANKER_MODEL} on {device}…")
        _reranker = CrossEncoder(config.RERANKER_MODEL, device=device)
    return _reranker


def _meta_to_chunk(meta: dict, text: str) -> Chunk:
    return Chunk(
        chunk_id  = meta.get("chunk_id", ""),
        paper_id  = meta.get("paper_id", ""),
        title     = meta.get("title", ""),
        authors   = meta.get("authors", ""),
        year      = meta.get("year", 0),
        section   = meta.get("section", ""),
        topic_tag = meta.get("topic_tag", ""),
        text      = text,
    )


# ── Search functions ─────────────────────────────────────────────────────────

def dense_search(query: str, n: int | None = None) -> List[Tuple[str, float]]:
    """Return list of (chunk_id, score) sorted by cosine similarity."""
    n = n or config.TOP_N
    embedder = _get_embedder()
    col      = _get_collection()

    q_vec = embedder.encode([query])[0].tolist()
    results = col.query(
        query_embeddings=[q_vec],
        n_results=min(n, col.count()),
        include=["metadatas", "documents", "distances"],
    )

    hits = []
    for chunk_id, dist in zip(results["ids"][0], results["distances"][0]):
        # Chroma cosine distance: score = 1 - distance
        hits.append((chunk_id, 1.0 - dist))
    return hits


def sparse_search(query: str, n: int | None = None) -> List[Tuple[str, float]]:
    """Return list of (chunk_id, bm25_score) sorted descending."""
    n = n or config.TOP_N
    data = _get_bm25()
    bm25, chunk_ids = data["bm25"], data["chunk_ids"]

    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    # Pair chunk_ids with scores, sort descending
    pairs = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)
    return [(cid, sc) for cid, sc in pairs[:n]]


def rrf_fuse(
    dense: List[Tuple[str, float]],
    sparse: List[Tuple[str, float]],
    k: int | None = None,
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion of two ranked lists. Returns merged list sorted by RRF score."""
    k = k if k is not None else config.RRF_K
    rrf_scores: Dict[str, float] = {}

    for rank, (cid, _) in enumerate(dense):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    for rank, (cid, _) in enumerate(sparse):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def rerank(query: str, candidates: List[Chunk], top_k: int | None = None) -> List[Chunk]:
    """Cross-encoder rerank of candidate chunks; returns top_k."""
    top_k = top_k or config.TOP_K
    reranker = _get_reranker()

    pairs  = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs)
    if not isinstance(scores, list):
        scores = scores.tolist()

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]


# ── Main retrieve function ────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int | None = None,
    top_n: int | None = None,
    use_hybrid: bool | None = None,
    use_reranker: bool | None = None,
) -> List[Chunk]:
    """
    Full retrieval pipeline: dense (+ optional BM25) → RRF → optional reranker.

    Returns a list of Chunk objects of length top_k.
    """
    top_k       = top_k       if top_k       is not None else config.TOP_K
    top_n       = top_n       if top_n       is not None else config.TOP_N
    use_hybrid  = use_hybrid  if use_hybrid  is not None else config.USE_HYBRID
    use_reranker= use_reranker if use_reranker is not None else config.USE_RERANKER

    col = _get_collection()

    # 1. Dense retrieval
    dense_hits = dense_search(query, n=top_n)

    # 2. Fuse with BM25 if hybrid
    if use_hybrid:
        sparse_hits = sparse_search(query, n=top_n)
        fused = rrf_fuse(dense_hits, sparse_hits)
    else:
        fused = dense_hits

    # 3. Fetch chunk objects for the candidate set
    #    We need more candidates than top_k for the reranker
    candidate_ids = [cid for cid, _ in fused[: top_n * 2]]
    if not candidate_ids:
        return []

    results = col.get(
        ids      = candidate_ids,
        include  = ["metadatas", "documents"],
    )
    id_to_chunk: Dict[str, Chunk] = {}
    for cid, meta, doc in zip(results["ids"], results["metadatas"], results["documents"]):
        id_to_chunk[cid] = _meta_to_chunk(meta, doc)

    candidates = [id_to_chunk[cid] for cid in candidate_ids if cid in id_to_chunk]

    # 4. Rerank or just truncate
    if use_reranker and len(candidates) > top_k:
        return rerank(query, candidates, top_k=top_k)
    else:
        return candidates[:top_k]
