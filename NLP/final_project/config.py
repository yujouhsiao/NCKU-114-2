import os

# ── Paths ────────────────────────────────────────────────────────────────────
PDF_DIR      = "data/pdfs"
PARSED_DIR   = "data/parsed"
PAPERS_JSON  = "data/papers.json"
CHROMA_DIR   = "data/chroma"
BM25_PATH    = "data/bm25.pkl"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512   # tokens
CHUNK_OVERLAP = 64    # tokens
SECTION_AWARE = True  # False → fixed-size (ablation)

# ── Embedding ────────────────────────────────────────────────────────────────
# Options: "BAAI/bge-m3"
#          "sentence-transformers/all-MiniLM-L6-v2"
#          "Qwen/Qwen3-Embedding-0.6B"
EMBED_MODEL = "BAAI/bge-m3"

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_N         = 20    # dense/sparse each take N before fuse
TOP_K         = 5     # final chunks passed to LLM
USE_HYBRID    = True  # dense + BM25 with RRF
USE_RERANKER  = True
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RRF_K         = 60    # RRF constant

# ── Generation ───────────────────────────────────────────────────────────────
LLM_BACKEND = "hf"              # "ollama" or "hf"
LLM_MODEL   = "llama3.1:8b-instruct"   # ollama only (ignored when backend=hf)
LLM_TEMPERATURE = 0.2

# ── Metadata / Stage-0 ───────────────────────────────────────────────────────
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
SIMILARITY_THRESHOLD = 0.6   # below → flag NEEDS_REVIEW
API_SLEEP        = 1.5       # seconds between S2 requests
API_RETRY        = 3

# ── Evaluation ───────────────────────────────────────────────────────────────
EVAL_K_VALUES = [3, 5, 10]
RANDOM_SEED   = 42

# ── Chroma ───────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "pruning"

# ── HF generation fallback ───────────────────────────────────────────────────
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
