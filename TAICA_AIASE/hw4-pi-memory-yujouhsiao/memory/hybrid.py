"""Hybrid Retrieval：BM25 + Embedding 語意搜尋，合併分數後回傳前 K 筆。"""
from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from .bm25 import bm25_search

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """懶載入：第一次呼叫才下載/載入模型。"""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def hybrid_search(
    query: str,
    docs: list[dict],
    k: int = 8,
    bm25_weight: float = 0.5,
    embed_weight: float = 0.5,
) -> list[dict]:
    """
    合併 BM25 分數與 Embedding 相似度，回傳前 K 筆。
    docs = [{"id": str, "text": str}, ...]
    回傳 [{"id": str, "score": float}, ...]
    """
    if not docs:
        return []

    # BM25 分數（正規化到 0~1）
    bm25_results = bm25_search(query, docs, k=len(docs))
    bm25_map = {r["id"]: r["score"] for r in bm25_results}
    max_bm25 = max(bm25_map.values()) if bm25_map else 1.0
    if max_bm25 == 0:
        max_bm25 = 1.0

    # Embedding 相似度
    model = _get_model()
    query_vec = model.encode(query, normalize_embeddings=True)
    doc_texts = [doc["text"] for doc in docs]
    doc_vecs = model.encode(doc_texts, normalize_embeddings=True)

    # 合併分數
    scores = []
    for i, doc in enumerate(docs):
        bm25_score = bm25_map.get(doc["id"], 0.0) / max_bm25
        embed_score = cosine_similarity(query_vec, doc_vecs[i])
        final_score = bm25_weight * bm25_score + embed_weight * embed_score
        scores.append({"id": doc["id"], "score": final_score})

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:k]