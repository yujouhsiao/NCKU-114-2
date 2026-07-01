#!/usr/bin/env python3
"""Retrieval evaluation: Recall@k, MRR, nDCG@k.

Usage:
    python eval/eval_retrieval.py [--qa eval/qa_set.json] [--k 3 5 10]
"""
from __future__ import annotations

import json
import math
import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    hits = set(retrieved_ids[:k]) & set(gold_ids)
    return len(hits) / len(gold_ids) if gold_ids else 0.0


def mrr(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    gold_set = set(gold_ids)
    for rank, pid in enumerate(retrieved_ids, 1):
        if pid in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    gold_set = set(gold_ids)
    dcg  = sum(
        1.0 / math.log2(rank + 1)
        for rank, pid in enumerate(retrieved_ids[:k], 1)
        if pid in gold_set
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics")
    parser.add_argument("--qa",  default="eval/qa_set.json")
    parser.add_argument("--k",   type=int, nargs="+", default=config.EVAL_K_VALUES)
    parser.add_argument("--no-hybrid",   action="store_true")
    parser.add_argument("--no-reranker", action="store_true")
    args = parser.parse_args()

    qa_path = Path(args.qa)
    if not qa_path.exists():
        print(f"QA file not found: {qa_path}\nRun gen_qa_draft.py and review it first.")
        sys.exit(1)

    with open(qa_path, encoding="utf-8") as f:
        qa_items = json.load(f)

    from src.retrieve import retrieve

    results_per_q = []
    for item in qa_items:
        qid         = item["id"]
        question    = item["question"]
        gold_pids   = item["gold_paper_ids"]

        chunks = retrieve(
            question,
            top_k=max(args.k),
            use_hybrid=not args.no_hybrid,
            use_reranker=not args.no_reranker,
        )
        retrieved_pids = list(dict.fromkeys(c.paper_id for c in chunks))

        row = {"id": qid, "question": question, "type": item.get("type", "")}
        for k in args.k:
            row[f"recall@{k}"]= recall_at_k(retrieved_pids, gold_pids, k)
            row[f"ndcg@{k}"]  = ndcg_at_k(retrieved_pids, gold_pids, k)
        row["mrr"] = mrr(retrieved_pids, gold_pids)
        results_per_q.append(row)

        print(f"  [{qid}] MRR={row['mrr']:.3f}  " +
              "  ".join(f"R@{k}={row[f'recall@{k}']:.3f}" for k in args.k))

    # Aggregate
    n = len(results_per_q)
    print(f"\n{'─'*60}")
    print(f"Results over {n} questions:")
    avg_mrr = sum(r["mrr"] for r in results_per_q) / n
    print(f"  MRR: {avg_mrr:.4f}")
    for k in args.k:
        avg_r = sum(r[f"recall@{k}"] for r in results_per_q) / n
        avg_n = sum(r[f"ndcg@{k}"]   for r in results_per_q) / n
        print(f"  Recall@{k}: {avg_r:.4f}  |  nDCG@{k}: {avg_n:.4f}")

    # Save per-question results
    out_path = Path("eval/retrieval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_per_q, f, indent=2)
    print(f"\nPer-question results saved to {out_path}")


if __name__ == "__main__":
    main()
