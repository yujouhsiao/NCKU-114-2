"""Stage 0: PDF → papers.json

Automatically extracts title from each PDF, looks up metadata via
Semantic Scholar (with arxiv fallback), classifies topic_tag via LLM,
and writes data/papers.json.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests
from rapidfuzz import fuzz

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Title extraction ─────────────────────────────────────────────────────────

def extract_title(pdf_path: str | Path) -> str:
    """Use Docling to parse the PDF and return the most likely title."""
    from docling.document_converter import DocumentConverter

    pdf_path = Path(pdf_path)
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    # Strategy 1: look for items explicitly labelled "title" in the document model
    try:
        for item in doc.texts:
            label = str(getattr(item, "label", "")).lower()
            text  = (getattr(item, "text", "") or "").strip()
            if "title" in label and 10 < len(text) < 300 and text != pdf_path.stem:
                return text
    except Exception:
        pass

    # Strategy 2: first heading in the exported markdown
    md_text = doc.export_to_markdown()
    for line in md_text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            candidate = line.lstrip("#").strip()
            if 10 < len(candidate) < 300 and candidate != pdf_path.stem:
                return candidate

    # Strategy 3: first substantial non-heading line
    for line in md_text.splitlines():
        line = line.strip()
        if len(line) > 20 and line != pdf_path.stem:
            return line

    return pdf_path.stem  # absolute last resort


# ── Semantic Scholar lookup ──────────────────────────────────────────────────

def _s2_search(title: str) -> Optional[dict]:
    """Query Semantic Scholar and return the best-matching paper dict or None."""
    fields = "title,year,venue,authors,externalIds,abstract,url"
    params = {"query": title, "fields": fields, "limit": 5}
    headers = {"User-Agent": "pruning-rag/1.0"}

    for attempt in range(config.API_RETRY):
        try:
            resp = requests.get(
                config.SEMANTIC_SCHOLAR_API,
                params=params,
                headers=headers,
                timeout=15,
            )
            if resp.status_code == 429:
                time.sleep(config.API_SLEEP * (attempt + 2))
                continue
            resp.raise_for_status()
            data = resp.json()
            papers = data.get("data", [])
            if not papers:
                return None, 0.0
            # Pick the result whose title best matches ours
            best, best_score = None, 0
            for p in papers:
                score = fuzz.token_sort_ratio(title.lower(), (p.get("title") or "").lower())
                if score > best_score:
                    best_score = score
                    best = p
            if best and best_score >= config.SIMILARITY_THRESHOLD * 100:
                return best, best_score / 100.0
            return best, best_score / 100.0
        except Exception as e:
            print(f"  [S2 attempt {attempt+1}] error: {e}")
            time.sleep(config.API_SLEEP)
    return None, 0.0


def _arxiv_fallback(title: str) -> dict:
    """Fallback: search arXiv by title and return partial metadata dict."""
    try:
        import arxiv
        client = arxiv.Client()
        search = arxiv.Search(query=f'ti:"{title}"', max_results=3)
        results = list(client.results(search))
        if not results:
            return {}, 0.0
        best = results[0]
        return {
            "title": best.title,
            "year": best.published.year if best.published else None,
            "venue": "arXiv",
            "authors": ", ".join(str(a) for a in best.authors[:3]) + (" et al." if len(best.authors) > 3 else ""),
            "externalIds": {"ArXiv": best.entry_id.split("/")[-1].split("v")[0]},
            "abstract": best.summary,
            "url": best.entry_id,
        }, 0.5
    except Exception as e:
        print(f"  [arXiv fallback] error: {e}")
        return {}, 0.0


def lookup_metadata(title: str) -> tuple[dict, float]:
    """Return (metadata_dict, similarity_score). Falls back to arXiv."""
    result, score = _s2_search(title)
    if result:
        return result, score
    print(f"  S2 returned no result for '{title}', trying arXiv…")
    return _arxiv_fallback(title)


# ── Topic classification ─────────────────────────────────────────────────────

_TOPIC_PROMPT = """You are a research paper classifier. Given a paper title and abstract, classify it into exactly one of these categories:

- pruning_classic: Traditional weight or neuron pruning methods (magnitude pruning, lottery ticket, etc.)
- structured: Structured pruning that removes entire filters, heads, or layers
- llm: Pruning or compression methods specifically targeting large language models (LLMs)
- statistics: Statistically-motivated pruning, Bayesian methods, or theoretical analysis

Reply with ONLY the category name (one of: pruning_classic, structured, llm, statistics).

Title: {title}
Abstract: {abstract}

Category:"""

VALID_TAGS = {"pruning_classic", "structured", "llm", "statistics"}


def classify_topic(title: str, abstract: str) -> str:
    """Use the configured LLM to classify the paper's topic_tag."""
    prompt = _TOPIC_PROMPT.format(title=title, abstract=abstract[:800])

    if config.LLM_BACKEND == "ollama":
        try:
            import ollama as _ollama
            resp = _ollama.generate(model=config.LLM_MODEL, prompt=prompt, options={"temperature": 0.0})
            tag = resp["response"].strip().lower().split()[0]
            return tag if tag in VALID_TAGS else "pruning_classic"
        except Exception as e:
            print(f"  [classify_topic] ollama error: {e}")
    else:
        try:
            from transformers import pipeline as hf_pipeline
            pipe = hf_pipeline("text-generation", model=config.HF_MODEL, max_new_tokens=10)
            out = pipe(prompt)[0]["generated_text"]
            # Extract the last word after "Category:"
            tag = out.split("Category:")[-1].strip().lower().split()[0]
            return tag if tag in VALID_TAGS else "pruning_classic"
        except Exception as e:
            print(f"  [classify_topic] hf error: {e}")

    # Heuristic fallback
    combined = (title + " " + abstract).lower()
    if any(w in combined for w in ["large language model", "llm", "gpt", "bert"]):
        return "llm"
    if any(w in combined for w in ["structured", "filter", "channel", "head"]):
        return "structured"
    if any(w in combined for w in ["bayes", "statistic", "theoretical", "convergence"]):
        return "statistics"
    return "pruning_classic"


# ── paper_id generation ──────────────────────────────────────────────────────

def _make_paper_id(authors_str: str, year: Optional[int], title: str) -> str:
    lastname = "unknown"
    if authors_str:
        first_author = authors_str.split(",")[0].split(" et")[0].strip()
        parts = first_author.split()
        if parts:
            lastname = parts[-1].lower()
            lastname = re.sub(r"[^a-z]", "", lastname)

    year_str = str(year) if year else "0000"
    slug_words = re.sub(r"[^a-z0-9 ]", "", title.lower()).split()[:4]
    slug = "_".join(slug_words)
    return f"{lastname}{year_str}_{slug}"


# ── Main builder ─────────────────────────────────────────────────────────────

def build_papers() -> None:
    """Scan PDF_DIR, extract metadata, and write PAPERS_JSON."""
    pdf_dir = Path(config.PDF_DIR)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}. Put your papers there and re-run.")
        return

    papers = []
    print(f"\nProcessing {len(pdfs)} PDF(s)…\n{'─'*60}")

    for pdf_path in pdfs:
        print(f"\n[{pdf_path.name}]")

        # 1. Extract title
        print("  Extracting title…")
        title = extract_title(pdf_path)
        print(f"  Title: {title}")

        # 2. Lookup metadata
        print("  Querying Semantic Scholar…")
        meta, score = lookup_metadata(title)
        needs_review = score < config.SIMILARITY_THRESHOLD
        flag = " ⚠ NEEDS_REVIEW" if needs_review else ""
        print(f"  Match score: {score:.2f}{flag}")

        api_title   = (meta.get("title") or title) if meta else title
        year        = (meta.get("year") or 0) if meta else 0
        venue       = (meta.get("venue") or "") if meta else ""
        abstract    = (meta.get("abstract") or "") if meta else ""
        url         = (meta.get("url") or "") if meta else ""
        arxiv_id    = ""
        authors_str = ""

        if meta:
            ext_ids = meta.get("externalIds") or {}
            arxiv_id = ext_ids.get("ArXiv", "")
            raw_authors = meta.get("authors") or []
            if isinstance(raw_authors, list):
                names = [a.get("name", "") for a in raw_authors[:3] if isinstance(a, dict)]
                authors_str = ", ".join(names) + (" et al." if len(raw_authors) > 3 else "")
            else:
                authors_str = str(raw_authors)

        # 3. Classify topic
        print("  Classifying topic…")
        topic_tag = classify_topic(api_title, abstract)
        print(f"  Topic: {topic_tag}")

        # 4. Generate paper_id
        paper_id = _make_paper_id(authors_str, year, api_title)
        print(f"  paper_id: {paper_id}")

        entry = {
            "paper_id":     paper_id,
            "title":        api_title,
            "authors":      authors_str,
            "year":         year,
            "venue":        venue,
            "arxiv_id":     arxiv_id,
            "url":          url,
            "topic_tag":    topic_tag,
            "pdf_filename": pdf_path.name,
        }
        if needs_review:
            entry["NEEDS_REVIEW"] = True

        papers.append(entry)
        time.sleep(config.API_SLEEP)

    out_path = Path(config.PAPERS_JSON)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"\n{'─'*60}")
    print(f"Wrote {len(papers)} entries to {out_path}")
    needs = [p for p in papers if p.get("NEEDS_REVIEW")]
    if needs:
        print(f"⚠  {len(needs)} paper(s) flagged NEEDS_REVIEW — please inspect manually.")
