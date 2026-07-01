#!/usr/bin/env python3
"""Auto-generate QA draft from indexed chunks.

⚠️  WARNING: This script produces a DRAFT ONLY.
    The generated gold_answers may contain errors.
    Using LLM-generated answers as gold answers introduces circular evaluation.
    YOU MUST manually review and correct eval/qa_set_draft.json before
    using it as the final eval/qa_set.json.

Usage:
    python eval/gen_qa_draft.py [--max-per-paper 2] [--output eval/qa_set_draft.json]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

WARNING = """\
╔══════════════════════════════════════════════════════════════════════════╗
║  ⚠  QA DRAFT WARNING                                                   ║
║                                                                          ║
║  This file is AUTO-GENERATED and must NOT be used directly as gold.     ║
║  Steps before using as eval/qa_set.json:                                ║
║    1. Verify each gold_answer is factually correct against the paper.   ║
║    2. Fix or remove incorrect questions.                                 ║
║    3. Adjust gold_paper_ids if the retrieval should come from multiple  ║
║       papers.                                                            ║
║    4. Add at least 5 "trend" type questions manually.                   ║
║    5. Save the corrected file as eval/qa_set.json.                      ║
╚══════════════════════════════════════════════════════════════════════════╝"""


TYPES = ["definition", "comparison", "method", "trend"]

QA_PROMPTS = {
    "definition": (
        "Based on the following passage, write ONE definition question and its answer.\n"
        "The question should ask 'What is X?' or 'Define X.' where X is a key term.\n"
        "Reply in JSON: {{\"question\": \"...\", \"gold_answer\": \"...\"}}\n\n"
        "Passage:\n{text}"
    ),
    "comparison": (
        "Based on the following passage, write ONE comparison question and its answer.\n"
        "The question should compare two methods, approaches, or results.\n"
        "Reply in JSON: {{\"question\": \"...\", \"gold_answer\": \"...\"}}\n\n"
        "Passage:\n{text}"
    ),
    "method": (
        "Based on the following passage, write ONE method/technique question and its answer.\n"
        "The question should ask how a specific technique works.\n"
        "Reply in JSON: {{\"question\": \"...\", \"gold_answer\": \"...\"}}\n\n"
        "Passage:\n{text}"
    ),
    "trend": (
        "Based on the following passage, write ONE trend or motivation question and its answer.\n"
        "The question should ask about research trends, motivations, or future directions.\n"
        "Reply in JSON: {{\"question\": \"...\", \"gold_answer\": \"...\"}}\n\n"
        "Passage:\n{text}"
    ),
}


def _generate_qa(text: str, qa_type: str) -> dict | None:
    """Ask the LLM to produce a question-answer pair for the given passage."""
    from src.generate import generate
    import torch

    prompt = QA_PROMPTS[qa_type].format(text=text[:1200])
    messages = [{"role": "user", "content": prompt}]

    try:
        raw = generate(messages)

        # Extract JSON block
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        qa = json.loads(raw[start:end])
        if "question" in qa and "gold_answer" in qa:
            return qa
    except Exception as e:
        print(f"    LLM error: {e}")
        torch.cuda.empty_cache()
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-paper", type=int, default=2)
    parser.add_argument("--output", default="eval/qa_set_draft.json")
    args = parser.parse_args()

    print(WARNING)
    print()

    # Load chunks from Chroma
    import chromadb
    client     = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_collection(config.CHROMA_COLLECTION)

    all_items = collection.get(include=["metadatas", "documents"])
    ids       = all_items["ids"]
    metas     = all_items["metadatas"]
    docs      = all_items["documents"]

    # Group chunks by paper_id
    by_paper: dict[str, list[tuple]] = {}
    for cid, meta, doc in zip(ids, metas, docs):
        pid = meta.get("paper_id", "unknown")
        by_paper.setdefault(pid, []).append((cid, meta, doc))

    qa_items  = []
    q_counter = 1
    type_cycle = TYPES.copy()
    random.seed(config.RANDOM_SEED)

    for paper_id, chunks in by_paper.items():
        n_generate = min(args.max_per_paper, len(chunks))
        sampled    = random.sample(chunks, n_generate)

        for cid, meta, doc in sampled:
            qa_type = type_cycle[(q_counter - 1) % len(type_cycle)]
            print(f"  [{q_counter:03d}] {paper_id} | type={qa_type}")

            qa = _generate_qa(doc, qa_type)
            if qa is None:
                print("    (skipped — LLM returned no valid JSON)")
                continue

            qa_items.append({
                "id":             f"q{q_counter:02d}",
                "type":           qa_type,
                "question":       qa["question"],
                "gold_answer":    qa["gold_answer"],
                "gold_paper_ids": [paper_id],
                "_source_chunk":  cid,
            })
            q_counter += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qa_items, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(qa_items)} draft questions to {out_path}")
    print()
    print(WARNING)


if __name__ == "__main__":
    main()
