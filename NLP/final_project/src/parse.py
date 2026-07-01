"""Stage 1: PDF → structured JSON with section list.

Output: data/parsed/{paper_id}.json
  {"paper_id": ..., "sections": [{"title": str, "text": str}, ...]}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def _load_papers() -> list[dict]:
    papers_path = Path(config.PAPERS_JSON)
    if not papers_path.exists():
        raise FileNotFoundError(f"{config.PAPERS_JSON} not found. Run build_papers.py first.")
    with open(papers_path, encoding="utf-8") as f:
        return json.load(f)


def parse_pdf(pdf_path: str | Path) -> dict:
    """Parse a single PDF with Docling and return structured section data."""
    from docling.document_converter import DocumentConverter

    pdf_path = Path(pdf_path)
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    sections: list[dict] = []
    current_title = "Abstract"
    current_texts: list[str] = []

    # Docling exports a document model; iterate over its items in reading order
    for item in doc.iterate_items():
        # item is a (element, level) tuple in some Docling versions; handle both
        if isinstance(item, tuple):
            element = item[0]
        else:
            element = item

        label = getattr(element, "label", None) or ""
        text  = getattr(element, "text",  None) or ""

        if label in ("section_header", "title") and text.strip():
            # Flush previous section
            body = " ".join(current_texts).strip()
            if body:
                sections.append({"title": current_title, "text": body})
            current_title = text.strip()
            current_texts = []
        elif label in ("paragraph", "text", "caption", "table", "list_item") and text.strip():
            current_texts.append(text.strip())

    # Flush last section
    body = " ".join(current_texts).strip()
    if body:
        sections.append({"title": current_title, "text": body})

    # If Docling gave us no structured sections, fall back to the full markdown
    if not sections:
        md = doc.export_to_markdown()
        sections = _split_markdown_sections(md)

    return sections


def _split_markdown_sections(md: str) -> list[dict]:
    """Parse markdown headings into sections as a fallback."""
    sections = []
    current_title = "Introduction"
    current_lines: list[str] = []

    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            body = "\n".join(current_lines).strip()
            if body:
                sections.append({"title": current_title, "text": body})
            current_title = stripped.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body:
        sections.append({"title": current_title, "text": body})

    return sections if sections else [{"title": "Full Text", "text": md}]


def parse_all() -> None:
    """Parse every PDF listed in papers.json and write to PARSED_DIR."""
    papers = _load_papers()
    parsed_dir = Path(config.PARSED_DIR)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nParsing {len(papers)} paper(s)…\n{'─'*60}")
    for paper in papers:
        paper_id    = paper["paper_id"]
        pdf_name    = paper["pdf_filename"]
        pdf_path    = Path(config.PDF_DIR) / pdf_name
        out_path    = parsed_dir / f"{paper_id}.json"

        if not pdf_path.exists():
            print(f"  [SKIP] {pdf_name} not found.")
            continue

        print(f"\n[{paper_id}] {pdf_name}")
        sections = parse_pdf(pdf_path)

        record = {"paper_id": paper_id, "sections": sections}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        total_chars = sum(len(s["text"]) for s in sections)
        print(f"  Sections: {len(sections)}  |  Total chars: {total_chars:,}")

    print(f"\nDone. Parsed files in: {parsed_dir}")
