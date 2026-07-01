#!/usr/bin/env python3
"""Ablation experiment matrix → results.csv.

Each row varies a single dimension while holding others at default.
Dimensions:
  1. no_rag vs. rag (baseline)
  2. embedding model
  3. chunking: section_aware vs. fixed; chunk_size in {256, 512, 1024}
  4. top_k: 3 / 5 / 10
  5. reranker on/off; hybrid on/off
  6. LLM model

Usage:
    python eval/run_experiments.py [--qa eval/qa_set.json] [--output results.csv]
    python eval/run_experiments.py --only embedding   # run one dimension
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as _cfg


# ── Metrics helpers (copied from eval_retrieval / eval_generation) ───────────

def _recall_at_k(retrieved, gold, k):
    return len(set(retrieved[:k]) & set(gold)) / len(gold) if gold else 0.0

def _mrr(retrieved, gold):
    gold_set = set(gold)
    for i, p in enumerate(retrieved, 1):
        if p in gold_set:
            return 1.0 / i
    return 0.0

def _ndcg_at_k(retrieved, gold, k):
    seen, unique = set(), []
    for p in retrieved:
        if p not in seen:
            seen.add(p); unique.append(p)
    gold_set = set(gold)
    dcg  = sum(1.0/math.log2(r+1) for r, p in enumerate(unique[:k], 1) if p in gold_set)
    idcg = sum(1.0/math.log2(i+2) for i in range(min(len(gold), k)))
    return dcg/idcg if idcg else 0.0

def _cit_accuracy(answer, sources):
    pattern = r"\[([A-Za-z\s]+(?:et al\.)?),\s*(\d{4})\]"
    citations = re.findall(pattern, answer)
    if not citations:
        return 1.0
    keys = set()
    for s in sources:
        yr = str(s.get("year", ""))
        sur = (s.get("authors", "").split(",")[0].split()[-1] or "").lower()
        keys.add((sur, yr))
    hits = sum(1 for a, y in citations if (a.strip().split()[0].lower(), y) in keys)
    return hits / len(citations)


# ── Per-condition runner ─────────────────────────────────────────────────────

def _patch_config(overrides: dict):
    """Apply a dict of {attr: value} to the config module in-place."""
    for k, v in overrides.items():
        setattr(_cfg, k, v)


def _reset_singletons():
    """Force retrieve.py singletons to reload (new embedder / collection)."""
    import src.retrieve as _ret
    _ret._embedder   = None
    _ret._chroma_col = None
    _ret._bm25_data  = None
    _ret._reranker   = None


def run_condition(
    qa_items: list[dict],
    overrides: dict,
    label: str,
    no_rag: bool = False,
    rebuild_index: bool = False,
) -> dict:
    """Run a single experiment condition and return a metrics dict."""
    _patch_config(overrides)
    _reset_singletons()

    if rebuild_index:
        print(f"  Rebuilding index for condition: {label}…")
        from src.chunk import chunk_all
        from src.embed import get_embedder
        from src.index import build_index
        chunks   = chunk_all(
            chunk_size    = overrides.get("CHUNK_SIZE", _cfg.CHUNK_SIZE),
            chunk_overlap = overrides.get("CHUNK_OVERLAP", _cfg.CHUNK_OVERLAP),
            section_aware = overrides.get("SECTION_AWARE", _cfg.SECTION_AWARE),
            embed_model   = overrides.get("EMBED_MODEL", _cfg.EMBED_MODEL),
        )
        embedder = get_embedder(overrides.get("EMBED_MODEL", _cfg.EMBED_MODEL))
        build_index(chunks, embedder=embedder, reset=True)

    from src.rag import answer, answer_no_rag

    recall3 = recall5 = recall10 = mrr_sum = ndcg5 = cit_acc = 0.0
    n = len(qa_items)

    for item in qa_items:
        question = item["question"]
        gold     = item["gold_paper_ids"]

        if no_rag:
            result = answer_no_rag(question)
        else:
            result = answer(
                question,
                top_k        = overrides.get("TOP_K",        _cfg.TOP_K),
                use_hybrid   = overrides.get("USE_HYBRID",   _cfg.USE_HYBRID),
                use_reranker = overrides.get("USE_RERANKER", _cfg.USE_RERANKER),
            )

        rpids = [s["paper_id"] for s in result.get("sources", [])]
        recall3  += _recall_at_k(rpids, gold, 3)
        recall5  += _recall_at_k(rpids, gold, 5)
        recall10 += _recall_at_k(rpids, gold, 10)
        mrr_sum  += _mrr(rpids, gold)
        ndcg5    += _ndcg_at_k(rpids, gold, 5)
        cit_acc  += _cit_accuracy(result["answer"], result.get("sources", []))

    return {
        "condition":  label,
        "no_rag":     no_rag,
        "recall@3":   recall3  / n,
        "recall@5":   recall5  / n,
        "recall@10":  recall10 / n,
        "mrr":        mrr_sum  / n,
        "ndcg@5":     ndcg5    / n,
        "citation_accuracy": cit_acc / n,
        **{k: str(v) for k, v in overrides.items()},
    }


# ── Experiment matrix ────────────────────────────────────────────────────────

def build_experiments(only: str | None) -> list[dict]:
    default = {}  # use config defaults

    experiments = []

    def add(label, overrides, no_rag=False, rebuild=False):
        experiments.append({
            "label": label,
            "overrides": overrides,
            "no_rag": no_rag,
            "rebuild": rebuild,
        })

    if only in (None, "baseline"):
        add("no_rag",            {},       no_rag=True)
        add("rag_default",       {})

    if only in (None, "embedding"):
        add("embed_minilm",      {"EMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2"},   rebuild=True)
        add("embed_bge_m3",      {"EMBED_MODEL": "BAAI/bge-m3"},                              rebuild=True)
        add("embed_qwen3",       {"EMBED_MODEL": "Qwen/Qwen3-Embedding-0.6B"},                rebuild=True)

    if only in (None, "chunking"):
        add("chunk_fixed",       {"SECTION_AWARE": False},             rebuild=True)
        add("chunk_size_256",    {"CHUNK_SIZE": 256},                   rebuild=True)
        add("chunk_size_512",    {"CHUNK_SIZE": 512},                   rebuild=True)
        add("chunk_size_1024",   {"CHUNK_SIZE": 1024},                  rebuild=True)

    if only in (None, "topk"):
        add("topk_3",            {"TOP_K": 3})
        add("topk_5",            {"TOP_K": 5})
        add("topk_10",           {"TOP_K": 10})

    if only in (None, "reranker"):
        add("dense_only",        {"USE_HYBRID": False, "USE_RERANKER": False})
        add("hybrid_no_rerank",  {"USE_HYBRID": True,  "USE_RERANKER": False})
        add("hybrid_rerank",     {"USE_HYBRID": True,  "USE_RERANKER": True})

    if only in (None, "llm"):
        add("llm_llama3",        {"LLM_MODEL": "llama3.1:8b-instruct"})
        add("llm_gemma3",        {"LLM_MODEL": "gemma3:9b"})
        add("llm_qwen2_5",       {"LLM_MODEL": "qwen2.5:7b"})

    return experiments


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run ablation experiment matrix")
    parser.add_argument("--qa",     default="eval/qa_set.json")
    parser.add_argument("--output", default="eval/results.csv")
    parser.add_argument("--only",   default=None,
                        choices=["baseline", "embedding", "chunking", "topk", "reranker", "llm"],
                        help="Run only one experimental dimension")
    args = parser.parse_args()

    qa_path = Path(args.qa)
    if not qa_path.exists():
        print(f"QA file not found: {qa_path}")
        sys.exit(1)

    with open(qa_path, encoding="utf-8") as f:
        qa_items = json.load(f)

    experiments = build_experiments(args.only)
    print(f"\nRunning {len(experiments)} conditions over {len(qa_items)} questions…\n{'─'*60}")

    rows = []
    for exp in experiments:
        label = exp["label"]
        print(f"\n[{label}]")
        try:
            row = run_condition(
                qa_items,
                overrides    = exp["overrides"],
                label        = label,
                no_rag       = exp["no_rag"],
                rebuild_index= exp.get("rebuild", False),
            )
            rows.append(row)
            print(f"  recall@5={row['recall@5']:.4f}  MRR={row['mrr']:.4f}  nDCG@5={row['ndcg@5']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({"condition": label, "error": str(e)})
        time.sleep(1)

    # Write CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = sorted(set(k for r in rows for k in r))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'─'*60}")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
