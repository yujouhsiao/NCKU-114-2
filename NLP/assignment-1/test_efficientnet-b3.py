import os
import csv
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# 0. 設定
# ─────────────────────────────────────────────
DEVICE      = torch.device("cuda:1")
NUM_CLASSES = 3
IMG_SIZE    = 300            # 與 train.py 一致
CHECKPOINT  = "./checkpoints_efficientnet-b3/best_model.pth"
TEST_DIR    = "test/unknown"
GT_CSV      = "/ssd7/hsiao/NLP_assignment1/solution_test_dataset2025_for_kaggle.csv"
OUTPUT_CSV  = "test_predictions_efficientnet-b3.csv"

IDX_TO_CLS  = {0: "spaghetti", 1: "ramen", 2: "udon"}
CLASS_NAMES = ["spaghetti", "ramen", "udon"]

print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 1. Transform（與 train.py 的 val_transform 一致）
# ─────────────────────────────────────────────
test_transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# 2. 載入 EfficientNet-B3 模型
# ─────────────────────────────────────────────
model = models.efficientnet_b3(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, NUM_CLASSES)
)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

print(f"Loaded checkpoint from fold {checkpoint['fold']} epoch {checkpoint['epoch']} "
      f"(val_acc={checkpoint['val_acc']:.4f}, smooth={checkpoint['smooth_val_acc']:.4f})")

# ─────────────────────────────────────────────
# 3. 載入 Ground Truth
# ─────────────────────────────────────────────
gt_labels = {}
with open(GT_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        gt_labels[int(row["ID"])] = int(row["Target"])

print(f"Loaded {len(gt_labels)} ground truth labels")

# ─────────────────────────────────────────────
# 4. 推論
# ─────────────────────────────────────────────
image_files = sorted([
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])
print(f"Found {len(image_files)} images in {TEST_DIR}\n")

results   = []
preds_all = []
gts_all   = []

with torch.no_grad():
    for fname in tqdm(image_files, desc="Inference"):
        img_id = int(os.path.splitext(fname)[0].replace("test_", ""))
        img    = Image.open(os.path.join(TEST_DIR, fname)).convert("RGB")
        tensor = test_transform(img).unsqueeze(0).to(DEVICE)
        probs  = torch.softmax(model(tensor), dim=1)[0]
        pred   = probs.argmax().item()
        gt     = gt_labels.get(img_id, -1)

        results.append({
            "ID":         img_id,
            "file":       fname,
            "pred_class": IDX_TO_CLS[pred],
            "pred_idx":   pred,
            "gt_idx":     gt,
            "gt_class":   IDX_TO_CLS.get(gt, "unknown"),
            "confidence": f"{probs[pred].item():.4f}",
            "correct":    pred == gt,
        })

        if gt != -1:
            preds_all.append(pred)
            gts_all.append(gt)

# ─────────────────────────────────────────────
# 5. Accuracy
# ─────────────────────────────────────────────
total   = len(preds_all)
correct = sum(p == g for p, g in zip(preds_all, gts_all))
acc     = correct / total

print(f"\nOverall Accuracy: {correct}/{total} = {acc:.4f} ({acc*100:.2f}%)")

print(f"\n{'Class':<12} {'Correct':>8} {'Total':>8} {'Acc':>8}")
print("-" * 42)
for cls_idx in range(NUM_CLASSES):
    cls_gt   = [i for i, g in enumerate(gts_all) if g == cls_idx]
    cls_corr = sum(preds_all[i] == cls_idx for i in cls_gt)
    cls_acc  = cls_corr / len(cls_gt) if cls_gt else 0
    print(f"{CLASS_NAMES[cls_idx]:<12} {cls_corr:>8} {len(cls_gt):>8} {cls_acc:>8.4f}")

# ─────────────────────────────────────────────
# 6. Confusion Matrix
# ─────────────────────────────────────────────
cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
for p, g in zip(preds_all, gts_all):
    cm[g][p] += 1

print(f"\nConfusion Matrix (row=GT, col=Pred):")
print(f"{'':>12}", end="")
for name in CLASS_NAMES:
    print(f"{name:>12}", end="")
print()
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:>12}", end="")
    for j in range(NUM_CLASSES):
        print(f"{cm[i][j]:>12}", end="")
    print()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
ax.set_title(f"Confusion Matrix (Acc={acc*100:.2f}%)")
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        color = "white" if cm[i][j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                color=color, fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix_efficientnet-b3.png", dpi=150)
print("\nSaved confusion_matrix_efficientnet-b3.png")

# ─────────────────────────────────────────────
# 7. 存 CSV
# ─────────────────────────────────────────────
results.sort(key=lambda x: x["ID"])
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "ID", "file", "pred_class", "pred_idx",
        "gt_class", "gt_idx", "confidence", "correct"
    ])
    writer.writeheader()
    writer.writerows(results)

print(f"Saved {OUTPUT_CSV}")
