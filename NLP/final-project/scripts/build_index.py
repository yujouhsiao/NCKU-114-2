#!/usr/bin/env python3
"""CLI: parse → chunk → embed → index.

Usage:
    python scripts/build_index.py [--rebuild] [--skip-parse] [--skip-chunk]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def main():
    parser = argparse.ArgumentParser(description="Build the pruning-RAG index")
    parser.add_argument("--rebuild",      action="store_true", help="Reset Chroma collection before indexing")
    parser.add_argument("--skip-parse",   action="store_true", help="Skip Stage 1 (reuse existing parsed/*.json)")
    parser.add_argument("--skip-chunk",   action="store_true", help="Not applicable here but kept for parity")
    parser.add_argument("--embed-model",  default=None,        help="Override EMBED_MODEL")
    args = parser.parse_args()

    if args.embed_model:
        config.EMBED_MODEL = args.embed_model

    # Stage 1: Parse
    if not args.skip_parse:
        print("\n── Stage 1: Parsing PDFs ──────────────────────────────────")
        from src.parse import parse_all
        parse_all()
    else:
        print("[skip] Stage 1 (parse)")

    # Stage 2: Chunk
    print("\n── Stage 2: Chunking ──────────────────────────────────────")
    from src.chunk import chunk_all
    chunks = chunk_all()

    # Stage 3+4: Embed + Index
    print("\n── Stage 3+4: Embedding + Indexing ───────────────────────")
    from src.embed import get_embedder
    from src.index import build_index

    embedder = get_embedder(config.EMBED_MODEL)
    build_index(chunks, embedder=embedder, reset=args.rebuild)

    print("\n✓ Index built successfully.")


if __name__ == "__main__":
    main()
