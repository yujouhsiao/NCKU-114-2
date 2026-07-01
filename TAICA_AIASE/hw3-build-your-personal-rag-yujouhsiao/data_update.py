#!/usr/bin/env python3
"""
data_update.py — Data Collection, Cleaning, Chunking, and Indexing Pipeline
Supports: .pdf, .md, .txt
Vector DB: ChromaDB (local, no server required)
Embedding: sentence-transformers (all-MiniLM-L6-v2, fully local & free)
"""
 
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional
 
# ── Constants ────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
HASH_CACHE    = Path("data/.file_hashes.json")
CHROMA_DIR    = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
COLLECTION    = "cv_papers"
EMBED_MODEL   = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
 
CHUNK_SIZE    = 512   # characters
CHUNK_OVERLAP = 64    # characters
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def file_hash(path: Path) -> str:
    """SHA-256 hash of a file (for change detection)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()
 
 
def load_hash_cache() -> dict:
    if HASH_CACHE.exists():
        return json.loads(HASH_CACHE.read_text())
    return {}
 
 
def save_hash_cache(cache: dict):
    HASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    HASH_CACHE.write_text(json.dumps(cache, indent=2))
 
 
# ── Text Extraction ───────────────────────────────────────────────────────────
 
def extract_pdf(path: Path) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("  [ERROR] PyMuPDF not installed. Run: pip install pymupdf", file=sys.stderr)
        return ""
 
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)
 
 
def extract_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")
 
 
def extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")
 
 
EXTRACTORS = {
    ".pdf": extract_pdf,
    ".md":  extract_markdown,
    ".txt": extract_txt,
}
 
 
# ── Text Cleaning ─────────────────────────────────────────────────────────────
 
def clean_text(text: str) -> str:
    """Remove noise: excess whitespace, page numbers, headers/footers patterns."""
    # Remove lines that are pure numbers (page numbers)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove excessive spaces
    text = re.sub(r" {3,}", " ", text)
    # Remove common PDF artefacts
    text = re.sub(r"\x0c", "\n", text)   # form feeds
    text = re.sub(r"\ufeff", "", text)   # BOM
    return text.strip()
 
 
# ── Chunking ──────────────────────────────────────────────────────────────────
 
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Fixed-length character chunking with overlap.
    Tries to break at paragraph boundaries when possible.
    """
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current = ""
 
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
 
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # Para itself may be longer than chunk_size → split it
            while len(para) > chunk_size:
                chunks.append(para[:chunk_size])
                para = para[chunk_size - overlap:]
            current = para
 
    if current:
        chunks.append(current)
 
    return [c for c in chunks if len(c.strip()) > 30]
 
 
# ── Embedding ─────────────────────────────────────────────────────────────────
 
def get_embedder():
    """Load sentence-transformers model (cached after first call)."""
    from sentence_transformers import SentenceTransformer
    print(f"  Loading embedding model: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)
 
 
def embed_chunks(model, chunks: list[str]) -> list[list[float]]:
    vecs = model.encode(chunks, show_progress_bar=True, batch_size=32)
    return vecs.tolist()
 
 
# ── ChromaDB ──────────────────────────────────────────────────────────────────
 
def get_collection(reset: bool = False):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
            print(f"  Deleted existing collection: {COLLECTION}")
        except Exception:
            pass
    col = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return col
 
 
def delete_doc_from_collection(col, source_name: str):
    """Remove all chunks belonging to a source document."""
    try:
        existing = col.get(where={"source": source_name})
        if existing["ids"]:
            col.delete(ids=existing["ids"])
    except Exception:
        pass
 
 
def upsert_chunks(col, chunks: list[str], embeddings: list[list[float]],
                  source: str, doc_id: str):
    ids, docs, embs, metas = [], [], [], []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        ids.append(f"{doc_id}_chunk_{i:04d}")
        docs.append(chunk)
        embs.append(emb)
        metas.append({"source": source, "chunk_index": i,
                      "chunk_total": len(chunks)})
 
    # ChromaDB max batch size is 166 — split into batches
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        col.upsert(
            ids=ids[start:end],
            documents=docs[start:end],
            embeddings=embs[start:end],
            metadatas=metas[start:end],
        )
 
 
# ── Main Pipeline ─────────────────────────────────────────────────────────────
 
def process_file(raw_path: Path, embedder, col,
                 force: bool = False, hash_cache: dict = None) -> Optional[str]:
    """
    Process a single raw file:
      1. Extract text
      2. Clean & save to data/processed/
      3. Chunk
      4. Embed
      5. Upsert to ChromaDB
 
    Returns the file hash if processed, None if skipped.
    """
    suffix = raw_path.suffix.lower()
    if suffix not in EXTRACTORS:
        print(f"  [SKIP] Unsupported format: {raw_path.name}")
        return None
 
    current_hash = file_hash(raw_path)
    doc_id = raw_path.stem.replace(" ", "_")
    source = raw_path.name
 
    # Incremental update: skip if unchanged
    if not force and hash_cache and hash_cache.get(str(raw_path)) == current_hash:
        print(f"  [SKIP] Unchanged: {raw_path.name}")
        return None
 
    print(f"  [PROC] {raw_path.name}")
 
    # 1. Extract
    raw_text = EXTRACTORS[suffix](raw_path)
    if not raw_text.strip():
        print(f"  [WARN] Empty text extracted from {raw_path.name}")
        return current_hash
 
    # 2. Clean & save processed
    cleaned = clean_text(raw_text)
    processed_path = PROCESSED_DIR / (raw_path.stem + ".txt")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_path.write_text(cleaned, encoding="utf-8")
 
    # 3. Chunk
    chunks = chunk_text(cleaned)
    print(f"       → {len(chunks)} chunks")
 
    # 4. Embed
    embeddings = embed_chunks(embedder, chunks)
 
    # 5. Upsert (remove old version first)
    delete_doc_from_collection(col, source)
    upsert_chunks(col, chunks, embeddings, source, doc_id)
 
    return current_hash
 
 
def run_pipeline(rebuild: bool = False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
 
    raw_files = sorted(
        p for p in RAW_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTRACTORS
    )
 
    if not raw_files:
        print(f"[WARN] No supported files found in {RAW_DIR}/")
        print("       Please add .pdf, .md, or .txt files.")
        return
 
    print(f"\n{'='*60}")
    print(f"  RAG Data Update Pipeline")
    print(f"  Mode   : {'FULL REBUILD' if rebuild else 'INCREMENTAL'}")
    print(f"  Files  : {len(raw_files)} found in {RAW_DIR}/")
    print(f"  Vector : ChromaDB @ {CHROMA_DIR}/")
    print(f"{'='*60}\n")
 
    # Prepare ChromaDB
    col = get_collection(reset=rebuild)
 
    # Load hash cache
    hash_cache = {} if rebuild else load_hash_cache()
    if rebuild:
        # Also clear processed dir
        if PROCESSED_DIR.exists():
            shutil.rmtree(PROCESSED_DIR)
        PROCESSED_DIR.mkdir(parents=True)
        print("  Cleared data/processed/ for full rebuild.\n")
 
    # Load embedder
    embedder = get_embedder()
    print()
 
    updated = 0
    for raw_path in raw_files:
        result = process_file(raw_path, embedder, col,
                              force=rebuild, hash_cache=hash_cache)
        if result:
            hash_cache[str(raw_path)] = result
            updated += 1
 
    save_hash_cache(hash_cache)
 
    total_docs = col.count()
    print(f"\n{'='*60}")
    print(f"  Done! Updated: {updated} file(s)")
    print(f"  Total chunks in DB: {total_docs}")
    print(f"{'='*60}\n")
 
 
# ── CLI ───────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="RAG Data Update Pipeline — Indexes data/raw/ into ChromaDB"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Full rebuild: clear processed/ and recreate entire index from scratch"
    )
    args = parser.parse_args()
    run_pipeline(rebuild=args.rebuild)
 
 
if __name__ == "__main__":
    main()