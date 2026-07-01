"""Stage 8: End-to-end RAG orchestration.

answer(query)        → {"answer": str, "sources": [...], "contexts": [...]}
answer_no_rag(query) → {"answer": str, "sources": [], "contexts": []}
"""
from __future__ import annotations

from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import re

from src.retrieve import retrieve
from src.prompt   import build_messages, _citation_key
from src.generate import generate


def _filter_hallucinated_citations(text: str, allowed: set[str]) -> str:
    """Remove [Author, Year] citations whose key is not in the allowed set."""
    def _replace(m: re.Match) -> str:
        key = m.group(1).strip()
        return f"[{key}]" if key in allowed else ""
    result = re.sub(r"\[([A-Za-z][^\[\]\n]+,\s*\d{4})\]", _replace, text)
    # Clean up double spaces and " ." left after removal
    result = re.sub(r" {2,}", " ", result)
    result = re.sub(r" \.", ".", result)
    return result.strip()


def answer(
    query: str,
    top_k: int | None = None,
    use_hybrid: bool | None = None,
    use_reranker: bool | None = None,
) -> dict:
    """
    Full RAG pipeline.

    Returns:
        {
          "answer":  str,                  # LLM response
          "sources": list[dict],           # chunk metadata for each retrieved chunk
          "contexts": list[str],           # raw chunk texts (for RAGAS evaluation)
        }
    """
    chunks = retrieve(
        query,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
    )

    messages = build_messages(query, chunks)
    response = generate(messages)

    # Build the set of allowed citation keys and strip any hallucinated ones
    seen_ids: set[str] = set()
    allowed_keys: set[str] = set()
    for c in chunks:
        if c.paper_id not in seen_ids:
            seen_ids.add(c.paper_id)
            allowed_keys.add(_citation_key(c.authors, c.year))
    response = _filter_hallucinated_citations(response, allowed_keys)

    sources = [
        {
            "chunk_id":  c.chunk_id,
            "paper_id":  c.paper_id,
            "title":     c.title,
            "authors":   c.authors,
            "year":      c.year,
            "section":   c.section,
            "topic_tag": c.topic_tag,
        }
        for c in chunks
    ]

    return {
        "answer":   response,
        "sources":  sources,
        "contexts": [c.text for c in chunks],
    }


def answer_no_rag(query: str) -> dict:
    """Direct LLM call without retrieval (No-RAG baseline)."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert research assistant in neural network pruning and "
                "statistical machine learning. Answer the question based on your knowledge."
            ),
        },
        {"role": "user", "content": query},
    ]
    response = generate(messages)
    return {
        "answer":   response,
        "sources":  [],
        "contexts": [],
    }
