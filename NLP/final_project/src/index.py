"""Stage 4: Build ChromaDB + BM25 indexes.

Writes:
  data/chroma/   — persistent Chroma collection
  data/bm25.pkl  — pickled (BM25Okapi, chunk_ids) tuple
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.schemas import Chunk


def _chroma_client():
    import chromadb
    return chromadb.PersistentClient(path=config.CHROMA_DIR)


def _get_collection(client, reset: bool = False):
    if reset:
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(chunks: List[Chunk], embedder=None, reset: bool = False) -> None:
    """
    Build Chroma (dense) and BM25 (sparse) indexes from chunks.

    Args:
        chunks:   List of Chunk objects from chunk_all().
        embedder: Pre-loaded Embedder; loaded from config if None.
        reset:    If True, delete and recreate the Chroma collection.
    """
    from src.embed import get_embedder

    if embedder is None:
        embedder = get_embedder()

    # ── Dense (Chroma) ───────────────────────────────────────────────────────
    Path(config.CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    client     = _chroma_client()
    collection = _get_collection(client, reset=reset)

    existing_ids = set(collection.get(include=[])["ids"])

    new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
    print(f"\nChroma: {len(existing_ids)} existing, {len(new_chunks)} new chunks to add.")

    if new_chunks:
        texts     = [c.text for c in new_chunks]
        ids       = [c.chunk_id for c in new_chunks]
        metadatas = [c.to_chroma_metadata() for c in new_chunks]

        print("Computing embeddings…")
        embeddings = embedder.encode(texts)

        BATCH = 500
        for i in range(0, len(new_chunks), BATCH):
            collection.add(
                ids        = ids[i : i + BATCH],
                embeddings = embeddings[i : i + BATCH].tolist(),
                documents  = texts[i : i + BATCH],
                metadatas  = metadatas[i : i + BATCH],
            )
        print(f"Added {len(new_chunks)} chunks to Chroma.")

    # ── Sparse (BM25) ────────────────────────────────────────────────────────
    # BM25 must cover ALL chunks (including pre-existing ones) for RRF alignment
    all_ids = [c.chunk_id for c in chunks]
    all_texts = [c.text for c in chunks]
    corpus_tokenized = [text.lower().split() for text in all_texts]

    from rank_bm25 import BM25Okapi
    print("Building BM25 index…")
    bm25 = BM25Okapi(corpus_tokenized)

    Path(config.BM25_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": all_ids, "texts": all_texts}, f)
    print(f"BM25 index saved to {config.BM25_PATH}  ({len(all_ids)} chunks).")
