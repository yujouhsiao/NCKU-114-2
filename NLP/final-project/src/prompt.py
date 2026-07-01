"""Stage 6: Prompt assembly.

Builds the messages list (OpenAI/Ollama chat format) from retrieved chunks.
"""
from __future__ import annotations

from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.schemas import Chunk


SYSTEM_PROMPT = """\
You are an expert research assistant specialising in neural network pruning and \
statistical machine learning. You help researchers understand papers, methods, \
and results in this domain.

Rules you MUST follow:
1. Answer ONLY based on the provided context passages below.
2. CITATION RULE — this is strict:
   - You MUST cite using ONLY the keys listed in the CITATION GUIDE at the top of \
the user message.
   - Copy each citation key EXACTLY as written in the guide (e.g. [Han et al., 2015]).
   - NEVER invent, guess, or recall a citation key that is not in the CITATION GUIDE. \
If no guide key applies to a claim, write the claim without a citation rather than \
fabricating one.
3. Place citations immediately after the sentence or claim they support.
4. If the context does not contain enough information to answer the question, \
say exactly: "The corpus does not contain relevant information to answer this question." \
Do NOT fabricate information or rely on prior knowledge outside the context.
5. Be precise and concise. Prefer bullet points for lists of methods or comparisons.
"""


def _citation_key(authors: str, year: int) -> str:
    """Derive 'Surname et al., Year' from a full author string.

    Handles two common formats:
      • "First Last, First Last, ..."   (comma-separated, first-name-first)
      • "Last, First and Last, First"   (and-separated, last-name-first)
    """
    and_parts = [a.strip() for a in authors.split(" and ") if a.strip()]

    if len(and_parts) >= 2:
        # "and"-separated list — first token is "Last, First" or "First Last"
        first_entry = and_parts[0]
        if "," in first_entry:
            # last-name-first: "Han, Song" → "Han"
            surname = first_entry.split(",")[0].strip()
        else:
            # first-name-last: "Song Han" → "Han"
            surname = first_entry.split()[-1]
        suffix = " et al."
    else:
        # Single string or comma-separated first-name-first list
        # e.g. "Alex Renda, Jonathan Frankle, Michael Carbin"
        comma_parts = [a.strip() for a in authors.split(",") if a.strip()]
        if not comma_parts:
            return f"Unknown, {year}"
        first_entry = comma_parts[0]          # "Alex Renda"
        tokens = first_entry.split()
        surname = tokens[-1] if tokens else first_entry   # last word = surname
        suffix = " et al." if len(comma_parts) > 1 else ""

    return f"{surname}{suffix}, {year}"


def build_messages(query: str, chunks: List[Chunk]) -> List[dict]:
    """Build chat messages with an explicit Citation Guide so the model
    copies exact keys rather than hallucinating author names."""

    # Deduplicate sources by paper_id while preserving order
    seen: set[str] = set()
    unique_chunks: List[Chunk] = []
    for chunk in chunks:
        if chunk.paper_id not in seen:
            seen.add(chunk.paper_id)
            unique_chunks.append(chunk)

    # Build citation guide
    cite_lines = []
    cite_map: dict[str, str] = {}   # paper_id → citation key
    for chunk in unique_chunks:
        key = _citation_key(chunk.authors, chunk.year)
        cite_map[chunk.paper_id] = key
        cite_lines.append(f"  [{key}] — {chunk.title}")
    citation_guide = "CITATION GUIDE — use ONLY these keys:\n" + "\n".join(cite_lines)

    # Build context blocks (tag each block with its citation key)
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        key = cite_map[chunk.paper_id]
        header = f"[Source {i} | cite as: [{key}]]\n{chunk.title}"
        block  = f"{header}\n{chunk.text}"
        context_blocks.append(block)

    context_str = "\n\n" + ("\n\n" + "─" * 60 + "\n\n").join(context_blocks) + "\n\n"

    allowed_keys = ", ".join(f"[{cite_map[c.paper_id]}]" for c in unique_chunks)
    user_content = (
        f"{citation_guide}\n\n"
        f"Context passages:\n{context_str}"
        f"Question: {query}\n\n"
        f"REMINDER: The ONLY allowed citation keys are: {allowed_keys}\n"
        f"Do NOT write any other citation — not even in parentheses.\n\n"
        f"Answer:"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
