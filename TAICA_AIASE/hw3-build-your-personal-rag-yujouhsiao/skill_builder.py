#!/usr/bin/env python3
"""
skill_builder.py — Automatically generate skill.md from the RAG knowledge base
by running a series of global knowledge-extraction queries.
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Reuse pipeline components from rag_query
sys.path.insert(0, str(Path(__file__).parent))
from rag_query import rag_query, get_collection

DEFAULT_OUTPUT = "skill.md"
DEFAULT_MODEL  = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.5-flash")

# ── Global questions for knowledge extraction ─────────────────────────────────
GLOBAL_QUESTIONS = [
    {
        "key": "overview",
        "question": (
            "What is the scope of this shadow removal knowledge base? "
            "Summarize in ≤200 words: what sub-problems are covered (detection, removal, "
            "video shadow removal, document shadow removal, etc.), what types of methods "
            "(traditional, CNN, GAN, Transformer, diffusion), and what the overall research "
            "landscape looks like."
        ),
    },
    {
        "key": "core_concepts",
        "question": (
            "List and briefly explain the 10–15 most important technical concepts "
            "in shadow removal research. Include concepts such as: shadow matte, "
            "illumination estimation, shadow detection, penumbra/umbra, physics-based models, "
            "attention mechanisms for shadows, paired/unpaired training, etc. "
            "Format as a numbered list: '<Concept>: <1–2 sentence explanation>'."
        ),
    },
    {
        "key": "key_trends",
        "question": (
            "What are the most significant research trends and recent breakthroughs "
            "in shadow removal? Consider: deep learning vs. traditional approaches, "
            "GAN-based methods, Transformer-based methods, diffusion model approaches, "
            "video shadow removal, document shadow removal, and unsupervised methods. "
            "List 5–8 trends with a brief explanation."
        ),
    },
    {
        "key": "key_entities",
        "question": (
            "In the shadow removal literature, who are the major authors and research groups? "
            "What are the key benchmark datasets (e.g. ISTD, SRD, LRSS, Video Shadow datasets)? "
            "What are the most referenced baseline methods and frameworks? "
            "Organize into: Authors/Groups, Datasets/Benchmarks, Key Methods/Baselines."
        ),
    },
    {
        "key": "methodology",
        "question": (
            "What are the dominant methodologies and pipelines in shadow removal? "
            "Describe: (1) detection-then-removal pipelines vs. end-to-end methods, "
            "(2) physics-based vs. data-driven approaches, "
            "(3) how attention, multi-scale features, and context are used, "
            "(4) common training strategies and loss functions used in shadow removal."
        ),
    },
    {
        "key": "diffusion_methods",
        "question": (
            "Summarize all diffusion-based shadow removal methods mentioned in this knowledge base. "
            "For each method, describe: (1) its key technical innovation, "
            "(2) how it uses diffusion models differently from GANs, "
            "(3) what datasets it was evaluated on, and (4) its main advantage over prior methods. "
            "Include methods like ShadowDiffusion, Diff-Shadow, Detail-Preserving Latent Diffusion, "
            "Latent Feature-Guided Diffusion, Boundary-Aware Divide and Conquer, etc."
        ),
    },
    {
        "key": "benchmarks_comparison",
        "question": (
            "Compare the main shadow removal benchmark datasets: ISTD, ISTD+, SRD, WSRD+, and others. "
            "For each dataset describe: number of training/test samples, whether shadow masks are provided, "
            "scene types (indoor/outdoor), resolution, and known limitations. "
            "Also mention the NTIRE 2023/2024/2025 challenge datasets and what they contribute."
        ),
    },
    {
        "key": "mask_free_methods",
        "question": (
            "What shadow removal methods in this knowledge base do NOT require a shadow mask as input? "
            "For each mask-free method, explain: (1) how it detects or handles shadows without a mask, "
            "(2) its architecture, and (3) how its performance compares to mask-guided methods. "
            "Include methods like ShadowRefiner, HomoFormer, PhaSR, Polarization-guided, etc."
        ),
    },
    {
        "key": "gaps",
        "question": (
            "What are the open challenges and limitations in shadow removal research? "
            "Consider: over-smoothing artifacts, color inconsistency, generalization to "
            "real-world shadows, video temporal consistency, evaluation metric limitations, "
            "lack of large-scale real paired datasets, and shadow-agnostic regions."
        ),
    },
    {
        "key": "example_qa",
        "question": (
            "Generate 5 representative question-answer pairs about shadow removal research. "
            "Cover different aspects: dataset choice, method comparison, loss function design, "
            "evaluation metrics, and handling specific shadow types. "
            "Format as:\nQ: <question>\nA: <concise answer>\n"
        ),
    },
]


# ── Source inventory ──────────────────────────────────────────────────────────

def list_sources() -> list[str]:
    """Fetch unique source filenames from ChromaDB."""
    col = get_collection()
    if col.count() == 0:
        return []
    results = col.get(include=["metadatas"])
    seen, sources = set(), []
    for meta in results["metadatas"]:
        s = meta.get("source", "unknown")
        if s not in seen:
            seen.add(s)
            sources.append(s)
    return sorted(sources)


# ── Skill.md writer ───────────────────────────────────────────────────────────

SKILL_TEMPLATE = """\
# Skill: Shadow Removal in Computer Vision

## Metadata
- **Knowledge Domain**: Shadow Detection & Removal, Image Restoration, Computational Photography
- **Number of Sources**: {n_sources} documents
- **Last Updated**: {today}
- **Target Agent Type**: Research assistant, shadow removal method advisor, CV paper review bot

---

## Overview

{overview}

---

## Core Concepts

{core_concepts}

---

## Key Trends

{key_trends}

---

## Key Entities

{key_entities}

---

## Methodology & Best Practices

{methodology}

---

## Diffusion-Based Methods

{diffusion_methods}

---

## Benchmark Datasets Comparison

{benchmarks_comparison}

---

## Mask-Free Shadow Removal Methods

{mask_free_methods}

---

## Knowledge Gaps & Limitations

{gaps}

This skill is based on a static snapshot of documents collected up to {today}.
It does not include papers published after that date. Coverage is focused on
the sub-topics present in the curated document set and may not represent the
entire CV/DL field.

---

## Example Q&A

{example_qa}

---

## Source References

| # | Filename | Type |
|---|----------|------|
{source_table}
"""


def build_source_table(sources: list[str]) -> str:
    rows = []
    for i, s in enumerate(sources, 1):
        ext = Path(s).suffix.upper().lstrip(".") or "TXT"
        rows.append(f"| {i} | `{s}` | {ext} |")
    return "\n".join(rows) if rows else "| — | (no sources indexed) | — |"


# ── Main ──────────────────────────────────────────────────────────────────────

def run(output: str, model: str):
    print("\n" + "="*60)
    print("  Skill Builder — Extracting knowledge from RAG...")
    print(f"  LLM Model : {model}")
    print(f"  Output    : {output}")
    print("="*60 + "\n")

    col = get_collection()
    if col.count() == 0:
        print("[ERROR] Knowledge base is empty. Run data_update.py --rebuild first.",
              file=sys.stderr)
        sys.exit(1)

    answers = {}
    for item in GLOBAL_QUESTIONS:
        key = item["key"]
        question = item["question"]
        print(f"  Querying: [{key}] ...")
        try:
            answer, _, _ = rag_query(question, [], top_k=6, model=model)
            answers[key] = answer.strip()
        except Exception as e:
            print(f"  [WARN] Failed to get answer for '{key}': {e}")
            answers[key] = "(Could not retrieve — check LLM connection)"

    sources = list_sources()
    skill_md = SKILL_TEMPLATE.format(
        n_sources          = len(sources),
        today              = date.today().isoformat(),
        overview           = answers.get("overview", ""),
        core_concepts      = answers.get("core_concepts", ""),
        key_trends         = answers.get("key_trends", ""),
        key_entities       = answers.get("key_entities", ""),
        methodology        = answers.get("methodology", ""),
        diffusion_methods  = answers.get("diffusion_methods", ""),
        benchmarks_comparison = answers.get("benchmarks_comparison", ""),
        mask_free_methods  = answers.get("mask_free_methods", ""),
        gaps               = answers.get("gaps", ""),
        example_qa         = answers.get("example_qa", ""),
        source_table       = build_source_table(sources),
    )

    Path(output).write_text(skill_md, encoding="utf-8")
    print(f"\n  ✅ skill.md written → {output}")
    print(f"     Sources indexed: {len(sources)}")
    print(f"     Characters     : {len(skill_md)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate skill.md from the RAG knowledge base"
    )
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model name (default: {DEFAULT_MODEL})")
    args = parser.parse_args()
    run(args.output, args.model)


if __name__ == "__main__":
    main()