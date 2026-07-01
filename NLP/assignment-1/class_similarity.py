"""
類別間相似度視覺化
執行方式：python class_similarity.py
輸出：eda_results/ 資料夾
需要：pip install scikit-learn
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

TRAIN_DIR = "train/"
SAVE_DIR  = "./eda_results"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE    = 64     # 縮小加速計算
N_SAMPLE    = 100    # 每類取樣數（太多會很慢）
TSNE_SAMPLE = 300    # t-SNE 用的樣本數

CLASS_INFO = {
    "0_spaghetti": ("Spaghetti", "#4C72B0"),
    "1_ramen":     ("Ramen",     "#DD8452"),
    "2_udon":      ("Udon",      "#55A868"),
}

# ─────────────────────────────────────────────
# 載入圖片 → 壓縮成向量
# ─────────────────────────────────────────────
print("載入圖片中...")
images, labels, label_names, paths = [], [], [], []

np.random.seed(42)
for folder, (cls_name, _) in CLASS_INFO.items():
    folder_path = os.path.join(TRAIN_DIR, folder)
    files = sorted([f for f in os.listdir(folder_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    chosen = np.random.choice(files, size=min(N_SAMPLE, len(files)), replace=False)
    for fname in chosen:
        fpath = os.path.join(folder_path, fname)
        img   = Image.open(fpath).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        images.append(np.array(img).flatten() / 255.0)
        labels.append(folder)
        label_names.append(cls_name)
        paths.append(fpath)

images = np.array(images)
unique_folders = list(CLASS_INFO.keys())
label_ids = np.array([unique_folders.index(l) for l in labels])

print(f"共載入 {len(images)} 張 ({N_SAMPLE} per class)")

# ─────────────────────────────────────────────
# 1. 平均臉（Average Image per Class）
# ─────────────────────────────────────────────
print("\n1. 平均圖像...")

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for col, (folder, (cls_name, color)) in enumerate(CLASS_INFO.items()):
    mask = np.array(labels) == folder
    avg  = images[mask].mean(axis=0).reshape(IMG_SIZE, IMG_SIZE, 3)
    avg  = np.clip(avg, 0, 1)
    axes[col].imshow(avg)
    axes[col].set_title(cls_name, fontsize=13, fontweight="bold", color=color)
    axes[col].axis("off")

plt.suptitle("Average Image per Class\n(blurrier = more variation within class)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "average_images.png"), dpi=150)
plt.close()
print("   → average_images.png")

# ─────────────────────────────────────────────
# 2. PCA 2D 散布圖
# ─────────────────────────────────────────────
print("2. PCA 散布圖...")

pca  = PCA(n_components=2, random_state=42)
emb  = pca.fit_transform(images)

fig, ax = plt.subplots(figsize=(8, 6))
for idx, (folder, (cls_name, color)) in enumerate(CLASS_INFO.items()):
    mask = label_ids == idx
    ax.scatter(emb[mask, 0], emb[mask, 1],
               c=color, label=cls_name, alpha=0.5, s=20, edgecolors="none")

ax.set_title(f"PCA (explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
ax.legend(fontsize=11); ax.grid(alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "pca_scatter.png"), dpi=150)
plt.close()
print("   → pca_scatter.png")

# ─────────────────────────────────────────────
# 3. t-SNE 散布圖（更能抓非線性結構）
# ─────────────────────────────────────────────
print("3. t-SNE 散布圖（可能需要 1~2 分鐘）...")

# t-SNE 前先用 PCA 降到 50 維加速
pca50 = PCA(n_components=min(50, images.shape[1]), random_state=42)
imgs_pca50 = pca50.fit_transform(images[:TSNE_SAMPLE])
ids_sub    = label_ids[:TSNE_SAMPLE]

tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
emb2 = tsne.fit_transform(imgs_pca50)

fig, ax = plt.subplots(figsize=(8, 6))
for idx, (folder, (cls_name, color)) in enumerate(CLASS_INFO.items()):
    mask = ids_sub == idx
    ax.scatter(emb2[mask, 0], emb2[mask, 1],
               c=color, label=cls_name, alpha=0.55, s=20, edgecolors="none")

ax.set_title("t-SNE (overlapping clusters = visually similar classes)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=11); ax.grid(alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "tsne_scatter.png"), dpi=150)
plt.close()
print("   → tsne_scatter.png")

# ─────────────────────────────────────────────
# 4. 類別間 Cosine 相似度矩陣
# ─────────────────────────────────────────────
print("4. 類別間相似度矩陣...")

# 計算每個類別的平均向量，然後算 cosine similarity
class_means = []
cls_labels  = []
for folder, (cls_name, _) in CLASS_INFO.items():
    mask = np.array(labels) == folder
    class_means.append(images[mask].mean(axis=0))
    cls_labels.append(cls_name)

sim_matrix = cosine_similarity(class_means)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0.5, vmax=1.0)
plt.colorbar(im, ax=ax, label="Cosine Similarity")

for i in range(3):
    for j in range(3):
        val = sim_matrix[i, j]
        color = "white" if val > 0.85 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)

ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(cls_labels); ax.set_yticklabels(cls_labels)
ax.set_title("Inter-class Cosine Similarity\n(higher = more similar pixel patterns)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "similarity_matrix.png"), dpi=150)
plt.close()

print("   → similarity_matrix.png")
print(f"\n   相似度結果：")
for i, n1 in enumerate(cls_labels):
    for j, n2 in enumerate(cls_labels):
        if j > i:
            print(f"   {n1} vs {n2}: {sim_matrix[i,j]:.4f}")

# ─────────────────────────────────────────────
# 完成
# ─────────────────────────────────────────────
print(f"\n完成！輸出至 {SAVE_DIR}/")
print("  average_images.png    → 各類平均圖像")
print("  pca_scatter.png       → PCA 2D 散布圖")
print("  tsne_scatter.png      → t-SNE 散布圖")
print("  similarity_matrix.png → 類別間 Cosine 相似度矩陣")