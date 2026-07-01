"""
Assignment #2 - Test Script
載入已訓練的模型（model_task1.pt、model_task2.pt），在測試集上評估
使用前請先執行 train.py
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ============================================================
# 設定
# ============================================================
TEST_PATH  = "test.csv"
CACHE_FILE = "pubmed_cache.csv"
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
MAX_LEN    = 512
BATCH_SIZE = 16
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {DEVICE}")

# ============================================================
# 資料載入
# ============================================================
def load_test(path):
    col_rename = {"Curate (0: T0, 1: T2/4)": "binary_label", "PMID": "pmid"}
    df = pd.read_csv(path).rename(columns=col_rename)
    print(f"測試集: {len(df)} 筆")
    return df

def load_texts_from_cache(pmids):
    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(f"找不到快取檔 {CACHE_FILE}，請先執行 train.py")
    cache = pd.read_csv(CACHE_FILE, dtype={"pmid": str})
    cache_dict = dict(zip(cache["pmid"].astype(str), cache["text"].fillna("")))
    return {str(p): _clean(cache_dict.get(str(p), "")) for p in pmids}

def _clean(text):
    import re
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ============================================================
# 關鍵字規則（Task 2 pseudo label）
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
# Dataset
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
# 模型載入 & 推論
# ============================================================
def load_model(path, num_labels):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到模型檔 {path}，請先執行 train.py")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME, num_labels=num_labels)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"  已載入模型: {path}")
    return model, tokenizer

def predict(model, tokenizer, texts):
    loader = DataLoader(
        BiomedDataset(texts, [0] * len(texts), tokenizer),
        batch_size=BATCH_SIZE)
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Predicting", leave=False):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out  = model(input_ids=ids, attention_mask=mask)
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
    return np.array(preds)

# ============================================================
# Task 1 評估
# ============================================================
def eval_task1(test_texts, test_labels):
    print(f"\n{'='*57}")
    print(f"  Task 1: Binary Classification (T0 vs T2/T4)")
    print(f"{'='*57}")

    model, tokenizer = load_model("model_task1.pt", num_labels=2)
    preds = predict(model, tokenizer, test_texts)

    print(f"\n[Task 1] 分類報告:")
    print(classification_report(test_labels, preds,
          target_names=["T0 (bench/basic)", "T2-T4 (translational)"], digits=4))
    print("混淆矩陣 (列=真實, 欄=預測):")
    print(confusion_matrix(test_labels, preds))
    return preds

# ============================================================
# Task 2 評估（Two-Stage）
# ============================================================
def eval_task2(test_df, test_texts, text_map, s1_preds):
    print(f"\n{'='*57}")
    print(f"  Task 2: Two-Stage Multi-Class (T0 / T2 / T3 / T4)")
    print(f"{'='*57}")

    test_multi = make_multiclass_labels(test_df, text_map)
    print("  標籤分布 (test):", pd.Series(test_multi).value_counts().sort_index().to_dict())

    s2_model, s2_tokenizer = load_model("model_task2.pt", num_labels=3)
    s2_preds    = predict(s2_model, s2_tokenizer, test_texts)
    final_preds = np.where(s1_preds == 0, 0, s2_preds + 1)

    class_names = ["T0 (basic)", "T2 (clinical utility)",
                   "T3 (implementation)", "T4 (population health)"]
    print(f"\n[Task 2 Two-Stage] 分類報告:")
    print(classification_report(test_multi, final_preds,
                                target_names=class_names, digits=4))
    print("混淆矩陣 (列=真實, 欄=預測):")
    print(confusion_matrix(test_multi, final_preds))

# ============================================================
# 主程式
# ============================================================
if __name__ == "__main__":
    test_df    = load_test(TEST_PATH)
    text_map   = load_texts_from_cache(test_df["pmid"].tolist())
    test_texts = [text_map.get(str(p), "") for p in test_df["pmid"]]

    s1_preds = eval_task1(test_texts, test_df["binary_label"].tolist())
    eval_task2(test_df, test_texts, text_map, s1_preds)

    print("\n\n評估完成！")
