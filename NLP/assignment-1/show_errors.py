"""
顯示分類錯誤的圖片
執行方式：python show_errors.py
"""

import os
import csv
import math
import matplotlib.pyplot as plt
from PIL import Image

CSV_PATH = "test_predictions_efficientnet-b3.csv"
TEST_DIR = "test/unknown"

# ─────────────────────────────────────────────
# 設定要看哪種錯誤
# ─────────────────────────────────────────────
GT_CLASS   = "ramen"    # 真實類別
PRED_CLASS = "udon"     # 預測類別（改這兩行可以看其他組合）
MAX_SHOW   = 30         # 最多顯示幾張

# ─────────────────────────────────────────────
# 讀取 CSV，篩出目標錯誤
# ─────────────────────────────────────────────
errors = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["gt_class"] == GT_CLASS and row["pred_class"] == PRED_CLASS:
            errors.append(row)

print(f"GT={GT_CLASS}, Pred={PRED_CLASS}: 共 {len(errors)} 張錯誤")

if len(errors) == 0:
    print("沒有這種錯誤！")
    exit()

# ─────────────────────────────────────────────
# 顯示圖片
# ─────────────────────────────────────────────
show = errors[:MAX_SHOW]
ncols = 6
nrows = math.ceil(len(show) / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

for i, row in enumerate(show):
    img_path = os.path.join(TEST_DIR, row["file"])
    img = Image.open(img_path).convert("RGB")
    axes[i].imshow(img)
    axes[i].set_title(
        f"Conf: {row['confidence']}",
        fontsize=20, color="red"
    )
    axes[i].axis("off")

# 隱藏多餘的格子
for j in range(len(show), len(axes)):
    axes[j].axis("off")

plt.suptitle(
    f"Misclassified: GT={GT_CLASS} → Pred={PRED_CLASS}  ({len(errors)} total)",
    fontsize=28, fontweight="bold"
)
plt.tight_layout()

save_path = f"errors_{GT_CLASS}_as_{PRED_CLASS}.png"
plt.savefig(save_path, dpi=120, bbox_inches="tight")
print(f"Saved {save_path}")
plt.show()