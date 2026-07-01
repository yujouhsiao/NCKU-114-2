"""Stage 7: LLM generation.

generate(messages) → str

Supports:
  LLM_BACKEND = "ollama"  → local Ollama server
  LLM_BACKEND = "hf"      → HuggingFace transformers pipeline
"""
from __future__ import annotations

from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def _generate_ollama(messages: List[dict]) -> str:
    import ollama
    resp = ollama.chat(
        model=config.LLM_MODEL,
        messages=messages,
        options={
            "temperature": config.LLM_TEMPERATURE,
            "seed": 42,
        },
    )
    return resp["message"]["content"]


_hf_pipeline = None


def _hf_max_memory() -> dict:
    """Build max_memory dict that excludes GPUs with <3 GiB free."""
    try:
        import torch
        mem = {}
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            free_gib = free / 1024 ** 3
            if free_gib >= 3.0:
                mem[i] = f"{int(free_gib * 0.90)}GiB"
            else:
                mem[i] = "0GiB"
        usable = {k: v for k, v in mem.items() if v != "0GiB"}
        print(f"  LLM max_memory: {usable}")
        return mem
    except Exception:
        return {}


def _generate_hf(messages: List[dict]) -> str:
    global _hf_pipeline
    if _hf_pipeline is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        print(f"Loading HF model {config.HF_MODEL}…")
        max_mem = _hf_max_memory()
        tok = AutoTokenizer.from_pretrained(config.HF_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            config.HF_MODEL,
            device_map="auto",
            max_memory=max_mem if max_mem else None,
            dtype=torch.float16,
        )
        _hf_pipeline = pipeline("text-generation", model=model, tokenizer=tok)

    try:
        tok = _hf_pipeline.tokenizer
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"

    from transformers import GenerationConfig
    gen_cfg = GenerationConfig(
        max_new_tokens=1024,
        do_sample=config.LLM_TEMPERATURE > 0,
        temperature=config.LLM_TEMPERATURE if config.LLM_TEMPERATURE > 0 else 1.0,
    )
    out = _hf_pipeline(prompt, return_full_text=False, generation_config=gen_cfg)
    return out[0]["generated_text"].strip()


def generate(messages: List[dict]) -> str:
    """Generate a response from the LLM given a messages list."""
    if config.LLM_BACKEND == "ollama":
        return _generate_ollama(messages)
    elif config.LLM_BACKEND == "hf":
        return _generate_hf(messages)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {config.LLM_BACKEND!r}. Use 'ollama' or 'hf'.")
