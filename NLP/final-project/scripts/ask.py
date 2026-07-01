#!/usr/bin/env python3
"""CLI query tool.

Usage:
    python scripts/ask.py "What is structured pruning?"
    python scripts/ask.py --no-rag "What is structured pruning?"
    python scripts/ask.py --top-k 3 "Compare SNIP and GraSP."
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def main():
    parser = argparse.ArgumentParser(description="Ask the pruning RAG system a question")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument("--no-rag",      action="store_true", help="Skip retrieval (No-RAG baseline)")
    parser.add_argument("--top-k",       type=int, default=None, help="Number of chunks to retrieve")
    parser.add_argument("--no-hybrid",   action="store_true", help="Disable BM25 hybrid retrieval")
    parser.add_argument("--no-reranker", action="store_true", help="Disable cross-encoder reranker")
    args = parser.parse_args()

    if args.top_k:
        config.TOP_K = args.top_k

    from src.rag import answer, answer_no_rag

    print(f"\nQuestion: {args.question}\n{'─'*60}")

    if args.no_rag:
        result = answer_no_rag(args.question)
        print("[Mode: No-RAG]\n")
    else:
        result = answer(
            args.question,
            top_k=args.top_k,
            use_hybrid=not args.no_hybrid,
            use_reranker=not args.no_reranker,
        )
        print(f"[Mode: RAG  | hybrid={not args.no_hybrid} | reranker={not args.no_reranker}]\n")

    print("Answer:")
    print(result["answer"])

    if result["sources"]:
        print(f"\n{'─'*60}")
        print("Sources:")
        seen = set()
        for s in result["sources"]:
            pid = s["paper_id"]
            if pid not in seen:
                seen.add(pid)
                print(f"  • [{s['authors']}, {s['year']}] {s['title']}")


if __name__ == "__main__":
    main()
