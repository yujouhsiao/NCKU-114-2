"""Stage 2: Section-aware chunking."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.schemas import Chunk


# ── Tokenizer factory ────────────────────────────────────────────────────────

_tokenizer_cache: dict[str, tuple] = {}


def _get_tokenizer(model_name: str) -> tuple:
    """Return (kind, tok_object) — kind is 'hf' or 'tiktoken'."""
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        obj = ("hf", tok)
    except Exception:
        import tiktoken
        obj = ("tiktoken", tiktoken.get_encoding("cl100k_base"))

    _tokenizer_cache[model_name] = obj
    return obj


def _encode(tok_obj: tuple, text: str) -> list:
    kind, tok = tok_obj
    if kind == "hf":
        return tok(text, add_special_tokens=False)["input_ids"]
    return tok.encode(text)


def _decode(tok_obj: tuple, tokens: list) -> str:
    kind, tok = tok_obj
    if kind == "hf":
        return tok.decode(tokens, skip_special_tokens=True)
    return tok.decode(tokens)


# ── Core chunking ────────────────────────────────────────────────────────────

def _split_tokens(text: str, tok_obj: tuple, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping token-bounded chunks and decode back to strings."""
    tokens = _encode(tok_obj, text)
    if not tokens:
        return []

    chunks_tokens = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks_tokens.append(tokens[start:end])
        if end == len(tokens):
            break
        start = end - overlap

    return [_decode(tok_obj, toks) for toks in chunks_tokens]


def chunk_paper(
    parsed: dict,
    meta: dict,
    *,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    section_aware: Optional[bool] = None,
    embed_model: Optional[str] = None,
) -> list[Chunk]:
    cs  = chunk_size    if chunk_size    is not None else config.CHUNK_SIZE
    co  = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    sa  = section_aware if section_aware is not None else config.SECTION_AWARE
    em  = embed_model   or config.EMBED_MODEL

    tok_obj   = _get_tokenizer(em)
    paper_id  = parsed["paper_id"]
    title     = meta.get("title", "")
    authors   = meta.get("authors", "")
    year      = meta.get("year", 0)
    topic_tag = meta.get("topic_tag", "")
    sections  = parsed.get("sections", [])

    chunks: list[Chunk] = []

    if sa:
        for sec_idx, section in enumerate(sections):
            sec_title = section.get("title", f"Section {sec_idx}")
            sec_text  = section.get("text", "").strip()
            if not sec_text:
                continue

            prefix    = f"{title} — {sec_title}\n\n"
            sub_texts = _split_tokens(sec_text, tok_obj, cs, co)

            for n, sub in enumerate(sub_texts):
                chunks.append(Chunk(
                    chunk_id  = f"{paper_id}__{sec_idx}__{n}",
                    paper_id  = paper_id,
                    title     = title,
                    authors   = authors,
                    year      = year,
                    section   = sec_title,
                    topic_tag = topic_tag,
                    text      = prefix + sub,
                ))
    else:
        full_text = "\n\n".join(s.get("text", "") for s in sections).strip()
        prefix    = f"{title}\n\n"
        sub_texts = _split_tokens(full_text, tok_obj, cs, co)
        for n, sub in enumerate(sub_texts):
            chunks.append(Chunk(
                chunk_id  = f"{paper_id}__0__{n}",
                paper_id  = paper_id,
                title     = title,
                authors   = authors,
                year      = year,
                section   = "Full Text",
                topic_tag = topic_tag,
                text      = prefix + sub,
            ))

    return chunks


def chunk_all(**kwargs) -> list[Chunk]:
    papers_path = Path(config.PAPERS_JSON)
    if not papers_path.exists():
        raise FileNotFoundError(f"{config.PAPERS_JSON} not found.")
    with open(papers_path, encoding="utf-8") as f:
        papers_meta = json.load(f)

    meta_by_id = {p["paper_id"]: p for p in papers_meta}
    parsed_dir = Path(config.PARSED_DIR)
    all_chunks: list[Chunk] = []

    print(f"\nChunking {len(papers_meta)} paper(s)…\n{'─'*60}")
    for paper_id, meta in meta_by_id.items():
        parsed_path = parsed_dir / f"{paper_id}.json"
        if not parsed_path.exists():
            print(f"  [SKIP] {paper_id} — parsed file not found.")
            continue
        with open(parsed_path, encoding="utf-8") as f:
            parsed = json.load(f)

        paper_chunks = chunk_paper(parsed, meta, **kwargs)
        all_chunks.extend(paper_chunks)
        print(f"  {paper_id}: {len(paper_chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    if all_chunks:
        print("\n── Sample chunk ──")
        sample = all_chunks[0]
        preview = sample.text[:300].replace("\n", " ")
        print(f"  chunk_id : {sample.chunk_id}")
        print(f"  section  : {sample.section}")
        print(f"  text[:300]: {preview}…")

    return all_chunks
