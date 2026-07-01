"""
Assignment #2 - Training Script
訓練 Task 1（二元）與 Task 2 Stage 2（T2/T3/T4）模型，儲存 checkpoint
"""

# pip install transformers torch pandas metapub scikit-learn tqdm

import os, time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# ============================================================
# 0. 設定
# ============================================================
TRAIN_PATH         = "train.csv"
TEST_PATH          = "test.csv"
MODEL_NAME         = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
CACHE_FILE         = "pubmed_cache.csv"
MAX_LEN            = 512
BATCH_SIZE         = 16
ACCUMULATION_STEPS = 2      # 等效 batch size = 32
EPOCHS             = 12
PATIENCE           = 3      # early stopping
LR                 = 2e-5
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {DEVICE}")

# ============================================================
# 1. 讀取 CSV
# ============================================================
def load_dataset(train_path, test_path):
    col_rename = {"Curate (0: T0, 1: T2/4)": "binary_label", "PMID": "pmid"}
    train_df = pd.read_csv(train_path).rename(columns=col_rename)
    test_df  = pd.read_csv(test_path).rename(columns=col_rename)
    print(f"訓練集: {len(train_df)} 筆 | 測試集: {len(test_df)} 筆")
    return train_df, test_df

# ============================================================
# 2. 從 PubMed 抓取文字（有快取避免重複請求）
# ============================================================
def fetch_pubmed_texts(pmids):
    cache_dict = {}
    if os.path.exists(CACHE_FILE):
        cache = pd.read_csv(CACHE_FILE, dtype={"pmid": str})
        cache_dict = dict(zip(cache["pmid"].astype(str), cache["text"].fillna("")))

    from metapub import PubMedFetcher
    fetcher  = PubMedFetcher()
    new_rows = []
    to_fetch = [str(p) for p in pmids if str(p) not in cache_dict]
    print(f"需要抓取 {len(to_fetch)} 筆（快取已有 {len(cache_dict)} 筆）")

    for pmid in tqdm(to_fetch, desc="Fetching PubMed"):
        try:
            article  = fetcher.article_by_pmid(pmid)
            title    = article.title    or ""
            abstract = article.abstract or ""
            text     = f"{title} {abstract}".strip()
        except Exception:
            text = ""
        cache_dict[pmid] = text
        new_rows.append({"pmid": pmid, "text": text})
        time.sleep(0.4)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        new_df.to_csv(
            CACHE_FILE,
            mode="a" if os.path.exists(CACHE_FILE) else "w",
            header=not os.path.exists(CACHE_FILE),
            index=False
        )
    return {str(p): _clean(cache_dict.get(str(p), "")) for p in pmids}

def _clean(text):
    import re
    text = re.sub(r'\s+', ' ', text)   # 合併多餘空白/換行
    return text.strip()

# ============================================================
# 3. 關鍵字規則：自動標注 T0 / T2 / T3 / T4
# ============================================================
T4_KEYWORDS = [
    "population health", "public health", "population impact",
    "real world", "real-world", "epidemiology",
    "disease prevention", "health disparities", "surveillance",
    "population-based", "community-based", "health promotion",
    "burden of disease", "incidence rate", "prevalence",
    "national", "global health", "population-attributable",
    "mortality rate", "morbidity", "pandemic", "endemic"
]
T3_KEYWORDS = [
    "implementation", "clinical practice", "guideline",
    "dissemination", "adoption", "health system",
    "health policy", "clinical translation", "practice change",
    "scale-up", "fidelity", "quality improvement",
    "healthcare delivery", "barriers to", "facilitators",
    "workflow", "clinician", "provider"
]
T2_KEYWORDS = [
    "clinical utility", "clinical validity", "evidence-based",
    "systematic review", "meta-analysis", "clinical trial",
    "randomized", "cohort study", "clinical outcome",
    "diagnostic accuracy", "therapeutic", "sensitivity",
    "specificity", "positive predictive", "negative predictive",
    "receiver operating", "area under"
]

def keyword_label(text):
    t = text.lower()
    if any(k in t for k in T4_KEYWORDS): return 3
    if any(k in t for k in T3_KEYWORDS): return 2
    if any(k in t for k in T2_KEYWORDS): return 1
    return 0

def make_multiclass_labels(df, text_map):
    labels = []
    for _, row in df.iterrows():
        if row["binary_label"] == 0:
            labels.append(0)
        else:
            text = text_map.get(str(row["pmid"]), "")
            labels.append(max(keyword_label(text), 1))
    return labels

# ============================================================
# 4. PyTorch Dataset
# ============================================================
class BiomedDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ============================================================
# 5. Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt  = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

# ============================================================
# 6. 訓練 / 預測
# ============================================================
def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="  Training", leave=False)):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labs = batch["label"].to(DEVICE)
        out  = model(input_ids=ids, attention_mask=mask)
        loss = criterion(out.logits, labs) / ACCUMULATION_STEPS
        loss.backward()
        total_loss += loss.item() * ACCUMULATION_STEPS
        if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    return total_loss / len(loader)

def get_predictions(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out  = model(input_ids=ids, attention_mask=mask)
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            trues.extend(batch["label"].numpy())
    return np.array(preds), np.array(trues)

def predict_only(model, tokenizer, texts):
    loader = DataLoader(
        BiomedDataset(texts, [0] * len(texts), tokenizer),
        batch_size=BATCH_SIZE)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out  = model(input_ids=ids, attention_mask=mask)
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
    return np.array(preds)

# ============================================================
# 7. 核心訓練流程（Focal Loss + Early Stopping）
# ============================================================
def build_and_train(train_texts, train_labels, val_texts, val_labels, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME, num_labels=num_labels).to(DEVICE)

    train_loader = DataLoader(
        BiomedDataset(train_texts, train_labels, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(
        BiomedDataset(val_texts, val_labels, tokenizer),
        batch_size=BATCH_SIZE)

    cw = compute_class_weight("balanced",
                              classes=np.unique(train_labels),
                              y=train_labels)
    criterion = FocalLoss(
        weight=torch.tensor(cw, dtype=torch.float).to(DEVICE),
        gamma=2.0)

    effective_steps = (len(train_loader) + ACCUMULATION_STEPS - 1) // ACCUMULATION_STEPS
    total_steps     = effective_steps * EPOCHS
    optimizer       = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler       = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=int(0.1 * total_steps),
                        num_training_steps=total_steps)

    best_f1    = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        preds, trues = get_predictions(model, val_loader)
        f1  = f1_score(trues, preds, average="macro")
        tag = " ★" if f1 > best_f1 else ""
        print(f"  Epoch {epoch}/{EPOCHS} | Loss: {loss:.4f} | Macro-F1: {f1:.4f}{tag}")
        if f1 > best_f1:
            best_f1    = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping（patience={PATIENCE}，best F1={best_f1:.4f}）")
            break

    model.load_state_dict(best_state)
    return model, tokenizer

# ============================================================
# 8. Task 1：二元分類
# ============================================================
def run_task1(train_texts, train_labels, test_texts, test_labels):
    print(f"\n{'='*57}")
    print(f"  Task 1: Binary Classification (T0 vs T2/T4)")
    print(f"{'='*57}")
    print("  標籤分布 (train):", pd.Series(train_labels).value_counts().sort_index().to_dict())
    print("  標籤分布 (test) :", pd.Series(test_labels).value_counts().sort_index().to_dict())

    model, tokenizer = build_and_train(
        train_texts, train_labels, test_texts, test_labels, num_labels=2)

    val_loader = DataLoader(
        BiomedDataset(test_texts, test_labels, tokenizer), batch_size=BATCH_SIZE)
    preds, trues = get_predictions(model, val_loader)

    print(f"\n[Task 1] 分類報告:")
    print(classification_report(trues, preds,
          target_names=["T0 (bench/basic)", "T2-T4 (translational)"], digits=4))
    print("混淆矩陣 (列=真實, 欄=預測):")
    print(confusion_matrix(trues, preds))

    torch.save(model.state_dict(), "model_task1.pt")
    print("\n模型已儲存: model_task1.pt")
    return model, tokenizer

# ============================================================
# 9. Task 2：兩階段多類別分類
# ============================================================
def run_task2_two_stage(train_df, test_df, train_texts, test_texts,
                        text_map, stage1_model, stage1_tokenizer):
    print(f"\n{'='*57}")
    print(f"  Task 2: Two-Stage Multi-Class (T0 / T2 / T3 / T4)")
    print(f"{'='*57}")

    train_multi = make_multiclass_labels(train_df, text_map)
    test_multi  = make_multiclass_labels(test_df,  text_map)
    print("  標籤分布 (train):", pd.Series(train_multi).value_counts().sort_index().to_dict())
    print("  標籤分布 (test) :", pd.Series(test_multi).value_counts().sort_index().to_dict())

    print("\n  [Stage 1] 套用 Task 1 模型...")
    s1_preds = predict_only(stage1_model, stage1_tokenizer, test_texts)

    s2_train_mask   = [l > 0 for l in train_multi]
    s2_train_texts  = [t for t, m in zip(train_texts, s2_train_mask) if m]
    s2_train_labels = [l - 1 for l, m in zip(train_multi, s2_train_mask) if m]

    s2_val_mask   = [l > 0 for l in test_multi]
    s2_val_texts  = [t for t, m in zip(test_texts, s2_val_mask) if m]
    s2_val_labels = [l - 1 for l, m in zip(test_multi, s2_val_mask) if m]

    print(f"\n  [Stage 2] 訓練 T2/T3/T4 分類器（{len(s2_train_texts)} 筆）...")
    print("  Stage 2 標籤分布 (train):",
          pd.Series(s2_train_labels).value_counts().sort_index().to_dict())

    s2_model, s2_tokenizer = build_and_train(
        s2_train_texts, s2_train_labels,
        s2_val_texts,   s2_val_labels,
        num_labels=3)

    s2_preds    = predict_only(s2_model, s2_tokenizer, test_texts)
    final_preds = np.where(s1_preds == 0, 0, s2_preds + 1)

    class_names = ["T0 (basic)", "T2 (clinical utility)",
                   "T3 (implementation)", "T4 (population health)"]
    print(f"\n[Task 2 Two-Stage] 分類報告:")
    print(classification_report(test_multi, final_preds,
                                target_names=class_names, digits=4))
    print("混淆矩陣 (列=真實, 欄=預測):")
    print(confusion_matrix(test_multi, final_preds))

    torch.save(s2_model.state_dict(), "model_task2.pt")
    print("\nStage 2 模型已儲存: model_task2.pt")

# ============================================================
# 10. 主程式
# ============================================================
if __name__ == "__main__":
    train_df, test_df = load_dataset(TRAIN_PATH, TEST_PATH)

    all_pmids = pd.concat([train_df["pmid"], test_df["pmid"]]).unique().tolist()
    print("\n開始從 PubMed 抓取文章...")
    text_map = fetch_pubmed_texts(all_pmids)

    train_texts = [text_map.get(str(p), "") for p in train_df["pmid"]]
    test_texts  = [text_map.get(str(p), "") for p in test_df["pmid"]]

    stage1_model, stage1_tokenizer = run_task1(
        train_texts,
        train_df["binary_label"].tolist(),
        test_texts,
        test_df["binary_label"].tolist()
    )

    run_task2_two_stage(
        train_df, test_df,
        train_texts, test_texts,
        text_map,
        stage1_model, stage1_tokenizer
    )

    print("\n\n訓練完成！")
