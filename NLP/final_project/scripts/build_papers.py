#!/usr/bin/env python3
"""Stage 0 CLI: Scan data/pdfs/ and auto-generate data/papers.json.

Usage:
    python scripts/build_papers.py [--pdf-dir PATH]
"""
import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.metadata import build_papers


def main():
    parser = argparse.ArgumentParser(description="Build papers.json from PDFs")
    parser.add_argument("--pdf-dir", default=None, help="Override PDF directory")
    args = parser.parse_args()

    if args.pdf_dir:
        config.PDF_DIR = args.pdf_dir

    build_papers()


if __name__ == "__main__":
    main()
