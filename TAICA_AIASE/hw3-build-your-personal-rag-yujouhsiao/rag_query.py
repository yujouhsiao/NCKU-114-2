#!/usr/bin/env python3
"""
rag_query.py — RAG Query Interface
Supports: single-query mode and interactive multi-turn conversation
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR  = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
COLLECTION  = "cv_papers"
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_K   = 5
DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = """\
You are an expert research assistant specializing in shadow removal, shadow detection,
and related image restoration topics in computer vision.
Answer the user's question based ONLY on the provided context passages.
If the context does not contain enough information, say so clearly.
Always cite your sources using the format [Source: <filename>, chunk <N>] inline.
Be concise but thorough. Use technical language appropriate for a researcher.
"""


# ── Embedding ─────────────────────────────────────────────────────────────────

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def embed_query(query: str) -> list[float]:
    model = get_embedder()
    return model.encode([query])[0].tolist()


# ── Retrieval ─────────────────────────────────────────────────────────────────

_col = None

def get_collection():
    global _col
    if _col is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _col = client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _col


def retrieve(query: str, top_k: int = DEFAULT_K) -> list[dict]:
    """Embed query and fetch top-k chunks from ChromaDB."""
    col = get_collection()
    if col.count() == 0:
        print("[ERROR] Vector DB is empty. Run data_update.py --rebuild first.",
              file=sys.stderr)
        return []

    qvec = embed_query(query)
    results = col.query(
        query_embeddings=[qvec],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":        doc,
            "source":      meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "score":       round(1 - dist, 4),   # cosine similarity
        })
    return chunks


# ── Prompt Assembly ───────────────────────────────────────────────────────────

def build_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, c in enumerate(chunks):
        context_parts.append(
            f"[Context {i+1}] Source: {c['source']}, chunk {c['chunk_index']} "
            f"(similarity: {c['score']})\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    return (
        f"Context passages:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}\n\n"
        f"Please answer based on the context above. "
        f"Cite sources inline as [Source: <filename>, chunk <N>]."
    )


# ── LLM Generation ────────────────────────────────────────────────────────────

def generate(messages: list[dict], model: str = DEFAULT_MODEL) -> str:
    """Call LLM via LiteLLM unified interface (OpenAI-compatible endpoint)."""
    from litellm import completion

    # Prefix with "openai/" to force LiteLLM to use OpenAI-compatible mode
    # instead of routing to Google Vertex AI directly
    litellm_model = f"openai/{model}" if not model.startswith("openai/") else model

    response = completion(
        model=litellm_model,
        messages=messages,
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url=os.getenv("LITELLM_BASE_URL",
                           "https://litellm.netdb.csie.ncku.edu.tw"),
    )
    return response.choices[0].message.content.strip()


# ── RAG Query (single turn) ───────────────────────────────────────────────────

def rag_query(query: str, history: list[dict], top_k: int,
              model: str) -> tuple[str, list[dict]]:
    """
    Full RAG pipeline for one turn.
    Returns (answer, updated_history).
    """
    # Retrieve
    chunks = retrieve(query, top_k)
    if not chunks:
        return "No relevant documents found in the knowledge base.", history

    # Assemble prompt
    user_content = build_prompt(query, chunks)

    # Build message list (system + history + current)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    # Generate
    answer = generate(messages, model)

    # Update history (keep last 6 messages = 3 turns)
    history.append({"role": "user",      "content": query})
    history.append({"role": "assistant", "content": answer})
    history = history[-6:]

    return answer, history, chunks


# ── Display ───────────────────────────────────────────────────────────────────

def print_answer(answer: str, chunks: list[dict]):
    print("\n" + "="*60)
    print("ANSWER")
    print("="*60)
    print(answer)
    print("\n" + "-"*60)
    print("SOURCES RETRIEVED")
    print("-"*60)
    for i, c in enumerate(chunks):
        print(f"  [{i+1}] {c['source']}  (chunk {c['chunk_index']}, "
              f"similarity={c['score']})")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "RAG Query Interface — Ask questions about your CV/DL knowledge base"
        )
    )
    parser.add_argument("--query",   "-q", type=str,
                        help="Single query (non-interactive mode)")
    parser.add_argument("--top-k",   "-k", type=int, default=DEFAULT_K,
                        help=f"Number of chunks to retrieve (default: {DEFAULT_K})")
    parser.add_argument("--model",   "-m", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model name (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    # ── Single query mode ──────────────────────────────────────────────────
    if args.query:
        answer, _, chunks = rag_query(args.query, [], args.top_k, args.model)
        print_answer(answer, chunks)
        return

    # ── Interactive multi-turn mode ────────────────────────────────────────
    print("\n" + "="*60)
    print("  CV/DL RAG Knowledge Base — Interactive Mode")
    print("  Type 'exit' or 'quit' to stop")
    print("  Type 'clear' to reset conversation history")
    print("="*60 + "\n")

    history = []
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Bye!")
            break
        if query.lower() == "clear":
            history = []
            print("  [Conversation history cleared]\n")
            continue

        answer, history, chunks = rag_query(query, history, args.top_k, args.model)
        print_answer(answer, chunks)


if __name__ == "__main__":
    main()