#!/usr/bin/env python3
"""Generation evaluation: citation accuracy + LLM-as-judge metrics (Qwen).

Usage:
    python eval/eval_generation.py [--qa eval/qa_set.json] [--no-rag]
    python eval/eval_generation.py --results eval/generation_results_rag.json
"""
from __future__ import annotations

import json
import re
import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Citation parsing ─────────────────────────────────────────────────────────

def _parse_citations(answer: str) -> list[tuple[str, str]]:
    pattern = r"\[([A-Za-z\s]+(?:et al\.)?),\s*(\d{4})\]"
    return re.findall(pattern, answer)


def citation_accuracy(answer: str, sources: List[dict]) -> float:
    citations = _parse_citations(answer)
    if not citations:
        return 1.0

    source_keys = set()
    for s in sources:
        year = str(s.get("year", ""))
        authors = s.get("authors", "")
        # First author name in "First Last, ..." format → surname = last token
        first_author = authors.split(",")[0].strip() if authors else ""
        surname = first_author.split()[-1].lower() if first_author else ""
        source_keys.add((surname, year))

    hits = 0
    for author, year in citations:
        # "Renda et al." → first token is the surname
        surname = author.strip().split()[0].lower()
        if (surname, year) in source_keys:
            hits += 1

    return hits / len(citations)


# ── LLM-as-judge metrics (Qwen-based, no RAGAS dependency) ───────────────────

_FAITHFULNESS_PROMPT = """\
Given a question, an answer, and supporting context passages, judge whether \
the answer is fully supported by the context.
Reply with only a single number: 1 if fully supported, 0 if not.

Question: {question}
Context:
{context}
Answer: {answer}

Score (1 or 0):"""

_RELEVANCY_PROMPT = """\
Given a question and an answer, judge how relevant the answer is to the question.
Reply with only a single integer from 1 (irrelevant) to 5 (highly relevant).

Question: {question}
Answer: {answer}

Score (1-5):"""


def _llm_judge(prompt: str) -> str:
    from src.generate import generate
    return generate([{"role": "user", "content": prompt}])


def _parse_score(text: str, max_val: int) -> float | None:
    m = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\b", text.strip())
    if m:
        val = float(m.group(1))
        if 0 <= val <= max_val:
            return val
    return None


def run_llm_judge(qa_items: list[dict], rag_results: list[dict]) -> dict:
    """Faithfulness and answer relevancy scored by local Qwen."""
    faith_scores, relev_scores = [], []
    total = len(qa_items)

    for i, (item, result) in enumerate(zip(qa_items, rag_results), 1):
        q      = item["question"]
        answer = result.get("answer", "")
        ctx    = "\n---\n".join(result.get("contexts", [])[:3])

        print(f"  Judge [{i}/{total}] {item['id']}", end="\r", flush=True)

        raw_f = _llm_judge(_FAITHFULNESS_PROMPT.format(
            question=q, context=ctx, answer=answer))
        s_f = _parse_score(raw_f, 1)
        if s_f is not None:
            faith_scores.append(s_f)

        raw_r = _llm_judge(_RELEVANCY_PROMPT.format(question=q, answer=answer))
        s_r = _parse_score(raw_r, 5)
        if s_r is not None:
            relev_scores.append(s_r / 5.0)

    print()
    return {
        "faithfulness":     sum(faith_scores) / len(faith_scores) if faith_scores else 0.0,
        "answer_relevancy": sum(relev_scores) / len(relev_scores) if relev_scores else 0.0,
        "n_faith":  len(faith_scores),
        "n_relev":  len(relev_scores),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate generation quality")
    parser.add_argument("--qa",      default="eval/qa_set.json")
    parser.add_argument("--no-rag",  action="store_true")
    parser.add_argument("--results", default=None, help="Pre-computed results JSON (skip RAG)")
    args = parser.parse_args()

    qa_path = Path(args.qa)
    if not qa_path.exists():
        print(f"QA file not found: {qa_path}")
        sys.exit(1)

    with open(qa_path, encoding="utf-8") as f:
        qa_items = json.load(f)

    if args.results and Path(args.results).exists():
        with open(args.results, encoding="utf-8") as f:
            rag_results = json.load(f)
    else:
        from src.rag import answer, answer_no_rag
        rag_results = []
        fn = answer_no_rag if args.no_rag else answer
        for item in qa_items:
            print(f"  [{item['id']}] Generating…")
            rag_results.append(fn(item["question"]))
        mode = "no_rag" if args.no_rag else "rag"
        out_path = Path(f"eval/generation_results_{mode}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rag_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {out_path}")

    # Citation accuracy
    cit_scores = [
        citation_accuracy(r["answer"], r.get("sources", []))
        for r in rag_results
    ]
    avg_cit = sum(cit_scores) / len(cit_scores) if cit_scores else 0.0
    print(f"\nCitation Accuracy: {avg_cit:.4f}")

    # LLM-as-judge
    print("\nRunning LLM-as-judge evaluation (Qwen)…")
    judge_scores = run_llm_judge(qa_items, rag_results)
    print("\nLLM-judge scores:")
    print(f"  Faithfulness:     {judge_scores['faithfulness']:.4f}  (n={judge_scores['n_faith']})")
    print(f"  Answer Relevancy: {judge_scores['answer_relevancy']:.4f}  (n={judge_scores['n_relev']})")

    summary = {
        "citation_accuracy": avg_cit,
        **judge_scores,
        "n_questions": len(qa_items),
        "mode": "no_rag" if args.no_rag else "rag",
    }
    out_path = Path("eval/generation_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
