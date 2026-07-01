"""
EfficientNet-B3 + 5-Fold Stratified Cross-Validation
需要：pip install scikit-learn
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# ─────────────────────────────────────────────
# 0. 設定
# ─────────────────────────────────────────────
GPU_IDS       = [0, 1]
DEVICE        = torch.device(f"cuda:{GPU_IDS[0]}")
NUM_CLASSES   = 3
BATCH_SIZE    = 64
NUM_EPOCHS    = 300
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
IMG_SIZE      = 300          # EfficientNet-B3 建議輸入尺寸
N_FOLDS       = 5
PATIENCE      = 20
SMOOTH_WINDOW = 5
SEED          = 42
SAVE_DIR      = "./checkpoints_efficientnet-b3"
os.makedirs(SAVE_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Using GPUs: {GPU_IDS}")
print(f"Model: EfficientNet-B3 (from scratch)")
print(f"5-Fold Stratified Cross-Validation\n")

# ─────────────────────────────────────────────
# 1. Transform
#    EfficientNet-B3 建議 300×300
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(320),             # 比 IMG_SIZE 稍大再裁切
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, label = self.dataset.samples[self.indices[i]]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


full_dataset = datasets.ImageFolder(root="train/")
all_indices  = np.arange(len(full_dataset))
all_labels   = np.array([label for _, label in full_dataset.samples])

label_map = {v: k for k, v in full_dataset.class_to_idx.items()}
print(f"Total samples: {len(full_dataset)}")
for label_id, count in zip(*np.unique(all_labels, return_counts=True)):
    print(f"  {label_map[label_id]}: {count}")
print()

# ─────────────────────────────────────────────
# 3. 建立 EfficientNet-B3 模型
# ─────────────────────────────────────────────
def build_model():
    model = models.efficientnet_b3(weights=None)   # 從頭訓練

    # 替換分類頭
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES)
    )

    # 權重初始化
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)
    model = nn.DataParallel(model, device_ids=GPU_IDS)
    return model.to(DEVICE)


# 印參數量
_tmp = models.efficientnet_b3(weights=None)
_tmp.classifier[1] = nn.Linear(_tmp.classifier[1].in_features, NUM_CLASSES)
print(f"EfficientNet-B3 total params: {sum(p.numel() for p in _tmp.parameters()):,}")
del _tmp

# ─────────────────────────────────────────────
# 4. Train / Val 函式
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="    Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="    Val  ", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# 5. 5-Fold Cross-Validation 主迴圈
# ─────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_results     = []
all_fold_history = []
global_best_acc  = 0.0

for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels), start=1):

    print(f"{'='*60}")
    print(f"  Fold {fold}/{N_FOLDS}  |  "
          f"Train: {len(train_idx)}  |  Val: {len(val_idx)}")
    print(f"{'='*60}")

    train_set = TransformSubset(full_dataset, train_idx.tolist(), train_transform)
    val_set   = TransformSubset(full_dataset, val_idx.tolist(),   val_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   [],
               "smooth_val_acc": []}
    best_val_acc = 0.0
    no_improve   = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc   = validate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        window = history["val_acc"][-SMOOTH_WINDOW:]
        smooth_val_acc = sum(window) / len(window)
        history["smooth_val_acc"].append(smooth_val_acc)

        cur_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch [{epoch:3d}/{NUM_EPOCHS}] LR={cur_lr:.6f} | "
              f"Train Acc={train_acc:.4f} | "
              f"Val Acc={val_acc:.4f} | Smooth={smooth_val_acc:.4f} | "
              f"No improve={no_improve}/{PATIENCE}")

        if smooth_val_acc > best_val_acc:
            best_val_acc = smooth_val_acc
            no_improve   = 0
            if smooth_val_acc > global_best_acc:
                global_best_acc = smooth_val_acc
                torch.save({
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "smooth_val_acc": smooth_val_acc,
                }, os.path.join(SAVE_DIR, "best_model.pth"))
                print(f"    ★ New global best! Saved best_model.pth "
                      f"(fold={fold}, smooth={smooth_val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best smooth_val_acc={best_val_acc:.4f})")
                break

    fold_results.append(best_val_acc)
    all_fold_history.append(history)
    print(f"\n  Fold {fold} best smooth val acc: {best_val_acc:.4f}\n")

    del model, optimizer, scheduler
    torch.cuda.empty_cache()

# ─────────────────────────────────────────────
# 6. 摘要
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  5-Fold Cross-Validation Summary")
print(f"{'='*60}")
for fold, acc in enumerate(fold_results, start=1):
    print(f"  Fold {fold}: best smooth val acc = {acc:.4f}")
print(f"  Mean: {np.mean(fold_results):.4f}  |  Std: {np.std(fold_results):.4f}")
print(f"\n  Global best smooth val acc: {global_best_acc:.4f}")
print(f"  → checkpoints_efficientnet-b3/best_model.pth")

# ─────────────────────────────────────────────
# 7. 訓練曲線
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, N_FOLDS, figsize=(22, 8))

for fold_idx, history in enumerate(all_fold_history):
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0, fold_idx].plot(ep, history["train_loss"], label="Train", alpha=0.7)
    axes[0, fold_idx].plot(ep, history["val_loss"],   label="Val",   alpha=0.7)
    axes[0, fold_idx].set_title(f"Fold {fold_idx+1} Loss", fontweight="bold")
    axes[0, fold_idx].set_xlabel("Epoch")
    axes[0, fold_idx].legend(fontsize=8); axes[0, fold_idx].grid(True)

    axes[1, fold_idx].plot(ep, history["train_acc"],      label="Train",  alpha=0.5)
    axes[1, fold_idx].plot(ep, history["val_acc"],        label="Val",    alpha=0.3)
    axes[1, fold_idx].plot(ep, history["smooth_val_acc"], label=f"Smooth({SMOOTH_WINDOW}ep)",
                           linewidth=2, color="orange")
    axes[1, fold_idx].set_title(f"Fold {fold_idx+1} Accuracy", fontweight="bold")
    axes[1, fold_idx].set_xlabel("Epoch")
    axes[1, fold_idx].legend(fontsize=8); axes[1, fold_idx].grid(True)

plt.suptitle(
    f"EfficientNet-B3 | 5-Fold CV | "
    f"Mean={np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("training_curve_5fold_efficientnet-b3.png", dpi=120)
print("Saved training_curve_5fold_efficientnet-b3.png")
