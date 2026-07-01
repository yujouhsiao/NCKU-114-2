"""Stage 3: Pluggable embedding models.

All embedders implement Embedder.encode() and return L2-normalised numpy arrays.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


BATCH_SIZE = 32


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _pick_device() -> str:
    """Return the single CUDA device with the most free memory, or 'cpu'."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        best_dev, best_free = 0, 0
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free = free
                best_dev = i
        print(f"  Selected GPU: cuda:{best_dev} ({best_free/1024**3:.1f} GiB free)")
        return f"cuda:{best_dev}"
    except Exception:
        return "cpu"


# ── Abstract base ────────────────────────────────────────────────────────────

class Embedder(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return (N, D) float32 array, L2-normalised."""
        ...


# ── BGE-M3 via FlagEmbedding ─────────────────────────────────────────────────

class BGEM3Embedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from FlagEmbedding import BGEM3FlagModel
        device = _pick_device()
        print(f"Loading {model_name} on {device}…")
        # Pass a single device to prevent FlagEmbedding from spawning multi-GPU workers
        self.model = BGEM3FlagModel(model_name, use_fp16=True, devices=[device])

    def encode(self, texts: List[str]) -> np.ndarray:
        all_vecs = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding (BGE-M3)", leave=False):
            batch = texts[i : i + BATCH_SIZE]
            out = self.model.encode(
                batch,
                batch_size=len(batch),
                max_length=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vecs = out["dense_vecs"]
            all_vecs.append(np.array(vecs, dtype=np.float32))
        result = np.vstack(all_vecs)
        return _l2_normalize(result)


# ── MiniLM via sentence-transformers ────────────────────────────────────────

class MiniLMEmbedder(Embedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"Loading {model_name}…")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        all_vecs = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding (MiniLM)", leave=False):
            batch = texts[i : i + BATCH_SIZE]
            vecs = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_vecs.append(np.array(vecs, dtype=np.float32))
        result = np.vstack(all_vecs)
        return _l2_normalize(result)


# ── Qwen3-Embedding-0.6B ────────────────────────────────────────────────────

class Qwen3Embedder(Embedder):
    """
    Qwen3-Embedding uses a special instruction prefix and last-token pooling.
    Prefer the sentence-transformers interface if available; fallback to raw HF.
    """
    INSTRUCTION = "Instruct: Retrieve semantically similar scientific text.\nQuery: "

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading {model_name} via sentence-transformers…")
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
            self._backend = "st"
        except Exception:
            from transformers import AutoTokenizer, AutoModel
            import torch
            print(f"Loading {model_name} via transformers…")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()
            self._backend = "hf"

    def _hf_encode(self, texts: list[str]) -> np.ndarray:
        import torch
        all_vecs = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                out = self.model(**enc)
            # Last-token pooling
            seq_len = enc["attention_mask"].sum(dim=1) - 1
            vecs = out.last_hidden_state[torch.arange(len(batch)), seq_len]
            all_vecs.append(vecs.float().numpy())
        return np.vstack(all_vecs)

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._backend == "st":
            all_vecs = []
            for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding (Qwen3)", leave=False):
                batch = texts[i : i + BATCH_SIZE]
                vecs = self.model.encode(batch, prompt=self.INSTRUCTION, convert_to_numpy=True)
                all_vecs.append(np.array(vecs, dtype=np.float32))
            result = np.vstack(all_vecs)
        else:
            result = self._hf_encode(texts)
        return _l2_normalize(result)


# ── Factory ──────────────────────────────────────────────────────────────────

_MODEL_MAP = {
    "BAAI/bge-m3":                              BGEM3Embedder,
    "sentence-transformers/all-MiniLM-L6-v2":   MiniLMEmbedder,
    "Qwen/Qwen3-Embedding-0.6B":                Qwen3Embedder,
}


def get_embedder(name: str | None = None) -> Embedder:
    """Return an Embedder for the given model name (default: config.EMBED_MODEL)."""
    name = name or config.EMBED_MODEL
    cls = _MODEL_MAP.get(name)
    if cls is None:
        # Try heuristics
        name_lower = name.lower()
        if "bge-m3" in name_lower:
            cls = BGEM3Embedder
        elif "qwen" in name_lower:
            cls = Qwen3Embedder
        else:
            cls = MiniLMEmbedder
    return cls(name)
