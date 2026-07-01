"""
下載 30 篇剪枝論文 PDF 到 data/pdfs/ 並打包成 zip。

在你自己的電腦執行（需要正常網路）：
    pip install arxiv
    python download_papers.py

流程：
- 27 篇在 arXiv 上的，用「標題搜尋」自動下載（會印出實際配對到的標題，請順手核對）。
- 3 篇不在 arXiv（OBD、OBS、Coreset），會列在最後請你手動下載。
- 全部完成後把 data/pdfs/ 壓成 pruning_pdfs.zip。
"""

import os
import time
import shutil
import requests

try:
    import arxiv
except ImportError:
    raise SystemExit("請先安裝：pip install arxiv")

OUT_DIR = "data/pdfs"
os.makedirs(OUT_DIR, exist_ok=True)

# (檔名, 標題) — 確認在 arXiv 上的 27 篇
ARXIV_PAPERS = [
    # A. classic
    ("han2015_weights_connections", "Learning both Weights and Connections for Efficient Neural Networks"),
    ("han2016_deep_compression",    "Deep Compression Compressing Deep Neural Networks with Pruning Trained Quantization and Huffman Coding"),
    # B. unstructured
    ("frankle2019_lottery_ticket",  "The Lottery Ticket Hypothesis Finding Sparse Trainable Neural Networks"),
    ("liu2019_rethinking_pruning",  "Rethinking the Value of Network Pruning"),
    ("zhu2017_to_prune_or_not",     "To prune or not to prune exploring the efficacy of pruning for model compression"),
    ("gale2019_state_of_sparsity",  "The State of Sparsity in Deep Neural Networks"),
    ("renda2020_rewinding",         "Comparing Rewinding and Fine-tuning in Neural Network Pruning"),
    # C. structured
    ("li2017_pruning_filters",      "Pruning Filters for Efficient ConvNets"),
    ("liu2017_network_slimming",    "Learning Efficient Convolutional Networks through Network Slimming"),
    ("luo2017_thinet",              "ThiNet A Filter Level Pruning Method for Deep Neural Network Compression"),
    ("he2017_channel_pruning",      "Channel Pruning for Accelerating Very Deep Neural Networks"),
    ("molchanov2017_taylor",        "Pruning Convolutional Neural Networks for Resource Efficient Inference"),
    ("molchanov2019_importance",    "Importance Estimation for Neural Network Pruning"),
    ("he2018_amc",                  "AMC AutoML for Model Compression and Acceleration on Mobile Devices"),
    ("he2019_fpgm",                 "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration"),
    ("lin2020_hrank",               "HRank Filter Pruning using High-Rank Feature Map"),
    ("liu2019_metapruning",         "MetaPruning Meta Learning for Automatic Neural Network Channel Pruning"),
    ("wen2016_structured_sparsity", "Learning Structured Sparsity in Deep Neural Networks"),
    # D. at_init_dynamic
    ("lee2019_snip",                "SNIP Single-shot Network Pruning based on Connection Sensitivity"),
    ("wang2020_grasp",              "Picking Winning Tickets Before Training by Preserving Gradient Flow"),
    ("tanaka2020_synflow",          "Pruning neural networks without any data by iteratively conserving synaptic flow"),
    ("evci2020_rigl",               "Rigging the Lottery Making All Tickets Winners"),
    ("mocanu2018_set",              "Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science"),
    # E. nonparametric
    ("hu2016_network_trimming",     "Network Trimming A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures"),
    ("luo2017_entropy_pruning",     "An Entropy-based Pruning Method for CNN Compression"),
    ("li2019_kse",                  "Exploiting Kernel Sparsity and Entropy for Interpretable CNN Compression"),
    # F. survey
    ("blalock2020_state_of_pruning","What is the State of Neural Network Pruning"),
]

# 不在 arXiv，請手動下載
MANUAL_PAPERS = [
    ("lecun1989_optimal_brain_damage", "Optimal Brain Damage (LeCun et al., 1989, NeurIPS)",
     "https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html"),
    ("hassibi1992_optimal_brain_surgeon", "Optimal Brain Surgeon (Hassibi & Stork, 1992, NeurIPS)",
     "https://proceedings.neurips.cc/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html"),
    ("dubey2018_coreset", "Coreset-Based Neural Network Compression (Dubey et al., 2018, ECCV)",
     "https://scholar.google.com/scholar?q=Coreset-Based+Neural+Network+Compression+Dubey"),
]


def fetch(title):
    """用標題搜尋 arXiv，回傳第一筆結果（找不到回 None）。"""
    client = arxiv.Client(page_size=3, delay_seconds=3.0, num_retries=3)
    # 先用標題欄位精準查，找不到再用一般查詢
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
    for fname, title in ARXIV_PAPERS:
        target = os.path.join(OUT_DIR, f"{fname}.pdf")
        if os.path.exists(target):
            print(f"[SKIP] 已存在 {fname}.pdf")
            ok += 1
            continue
        print(f"[..] 搜尋：{title}")
        paper = fetch(title)
        if paper is None:
            print(f"[MISS] 找不到，請手動下載：{title}")
            missing.append((fname, title))
            continue
        print(f"      配對到：{paper.title}")
        try:
            pdf_url = paper.pdf_url
            resp = requests.get(pdf_url, timeout=60, headers={"User-Agent": "pruning-rag/1.0"})
            resp.raise_for_status()
            with open(target, "wb") as f:
                f.write(resp.content)
            print(f"[OK] 已存 {fname}.pdf")
            ok += 1
        except Exception as e:
            print(f"[ERR] 下載失敗：{e}")
            missing.append((fname, title))
        time.sleep(3)  # 對 arXiv 客氣一點，避免被限流

    print("\n" + "=" * 60)
    print(f"自動下載完成：{ok}/{len(ARXIV_PAPERS)} 篇")

    print("\n以下請手動下載後放進 data/pdfs/（建議用括號內檔名）：")
    for fname, desc, url in MANUAL_PAPERS:
        print(f"  - {desc}\n      建議檔名：{fname}.pdf\n      {url}")
    for fname, title in missing:
        print(f"  - {title}\n      建議檔名：{fname}.pdf")

    # 打包
    zip_path = shutil.make_archive("pruning_pdfs", "zip", OUT_DIR)
    print(f"\n已打包：{zip_path}")
    print("（手動補的 PDF 放進 data/pdfs/ 後，可重跑本腳本重新打包。）")


if __name__ == "__main__":
    main()
