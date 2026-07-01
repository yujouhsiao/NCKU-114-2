"""
下載 20 篇新論文 PDF 到 data/pdfs/

分兩類：
  A. 影像辨識模型剪枝（2021-2026）：10 篇 Vision Transformer / CNN 剪枝
  B. 統計 / 無母數方法（2021-2026）：10 篇 統計驅動的剪枝分析

執行：
    python download_papers_v2.py
"""

import os
import time
import requests

try:
    import arxiv
except ImportError:
    raise SystemExit("請先安裝：pip install arxiv")

OUT_DIR = "data/pdfs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── A. 影像辨識模型剪枝（2021-2026）────────────────────────────────────────────
VISION_PRUNING = [
    # (存檔名, 搜尋標題)
    ("fang2023_depgraph",
     "DepGraph: Towards Any Structural Pruning"),
    ("rao2021_dynamicvit",
     "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification"),
    ("sui2021_chip",
     "CHIP: CHannel Independence-based Pruning for Compact Neural Networks"),
    ("tang2022_patch_slimming",
     "Patch Slimming for Efficient Vision Transformers"),
    ("yu2022_unified_vit_compression",
     "Unified Visual Transformer Compression"),
    ("chen2023_sparsevit",
     "SparseViT: Revisiting Activation Sparsity for Efficient High-Resolution Vision Transformer"),
    ("kong2022_spvit",
     "SPViT: Enabling Faster Vision ViTs via Latency-Aware Soft Token Pruning"),
    ("xu2022_evovit",
     "Evo-ViT: Slow-Fast Token Evolution for Dynamic Vision Transformer"),
    ("hou2022_chex",
     "Chex: Channel Exploration for CNN Model Compression"),
    ("liu2021_group_fisher",
     "Group Fisher Pruning for Practical Network Compression"),
]

# ── B. 統計 / 無母數方法（2021-2026）────────────────────────────────────────────
STATISTICAL_PRUNING = [
    ("zhang2022_platon",
     "PLATON: Pruning Large Transformer Models with Upper Confidence Bound of Weight Importance"),
    ("lubana2021_mechanistic",
     "A Mechanistic Analysis of Lottery Ticket Masking"),
    ("paul2022_unmasking_lth",
     "Unmasking the Lottery Ticket Hypothesis: What's Encoded in a Winning Ticket's Mask"),
    ("rachwan2022_winning_ahead",
     "Winning the Lottery Ahead of Time: Efficient Early Network Maturation by Activation Analysis"),
    ("kwon2022_fast_post_training",
     "A Fast Post-Training Pruning Framework for Transformers"),
    ("ding2022_optimal_brain_compression",
     "Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning"),
    ("frantar2022_spdy",
     "SPDY: Accurate Pruning with Speedup Guarantees"),
    ("zhang2022_pruning_generalization",
     "Pruning's Effect on Generalization Through the Lens of Training and Regularization"),
    ("chen2021_elastic_kernel",
     "Only Train Once: A One-Shot Neural Network Training And Pruning Framework"),
    ("kurtic2022_optimal_bert_surgeon",
     "The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models"),
]

ALL_PAPERS = [
    ("A_vision_pruning", p) for p in VISION_PRUNING
] + [
    ("B_statistical", p) for p in STATISTICAL_PRUNING
]


def fetch(title: str):
    """arXiv 標題搜尋，回傳第一筆結果。"""
    client = arxiv.Client(page_size=3, delay_seconds=3.0, num_retries=3)
    for q in (f'ti:"{title}"', title):
        try:
            results = list(client.results(arxiv.Search(query=q, max_results=3)))
        except Exception as e:
            print(f"      查詢出錯：{e}")
            results = []
        if results:
            return results[0]
    return None


def main():
    ok, missing = 0, []

    print("\n" + "=" * 60)
    print("A. 影像辨識模型剪枝（2021-2026）")
    print("=" * 60)
    for fname, title in VISION_PRUNING:
        ok, missing = _download(fname, title, ok, missing)

    print("\n" + "=" * 60)
    print("B. 統計 / 無母數方法（2021-2026）")
    print("=" * 60)
    for fname, title in STATISTICAL_PRUNING:
        ok, missing = _download(fname, title, ok, missing)

    print("\n" + "=" * 60)
    print(f"完成：{ok}/20 篇下載成功")
    if missing:
        print("\n以下需手動下載：")
        for fname, title in missing:
            print(f"  [{fname}] {title}")


def _download(fname: str, title: str, ok: int, missing: list):
    target = os.path.join(OUT_DIR, f"{fname}.pdf")
    if os.path.exists(target):
        print(f"[SKIP] 已存在 {fname}.pdf")
        return ok + 1, missing

    print(f"[..] 搜尋：{title}")
    paper = fetch(title)
    if paper is None:
        print(f"[MISS] 找不到：{title}")
        missing.append((fname, title))
        return ok, missing

    print(f"      配對到：{paper.title}")
    try:
        resp = requests.get(
            paper.pdf_url, timeout=60,
            headers={"User-Agent": "pruning-rag/1.0"}
        )
        resp.raise_for_status()
        with open(target, "wb") as f:
            f.write(resp.content)
        print(f"[OK]  已存 {fname}.pdf")
        ok += 1
    except Exception as e:
        print(f"[ERR] 下載失敗：{e}")
        missing.append((fname, title))
    time.sleep(3)
    return ok, missing


if __name__ == "__main__":
    main()
