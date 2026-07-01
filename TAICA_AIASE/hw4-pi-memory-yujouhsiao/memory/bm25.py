"""BM25-lite：給每筆文件對查詢打相關度分數，回傳排序後的前 K 筆。整個作業的核心。
tokenize() 已給你；bm25_search() 的計分要你填。"""
from __future__ import annotations
import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """小寫化後，取出英數字詞與單個 CJK 字元。不做 stemming。（已提供）"""
    return _TOKEN_RE.findall(text.lower())


def bm25_search(
    query: str,
    docs: list[dict],
    k: int = 8,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict]:
    if not docs:
        return []

    # 步驟1：斷詞，記每篇長度，算 avgdl
    tokenized_docs = [tokenize(doc["text"]) for doc in docs]
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avgdl = sum(doc_lengths) / len(doc_lengths)
    N = len(docs)

    # 步驟2：算每個詞的 document frequency（df）
    df = Counter()
    for tokens in tokenized_docs:
        for term in set(tokens):  # set() 讓每篇只算一次
            df[term] += 1

    # 步驟3：對每篇文件，加總查詢每個詞的 BM25 貢獻
    query_terms = tokenize(query)
    scores = []
    for i, (doc, tokens) in enumerate(zip(docs, tokenized_docs)):
        tf_map = Counter(tokens)
        dl = doc_lengths[i]
        score = 0.0
        for term in query_terms:
            if term not in tf_map:
                continue
            tf = tf_map[term]
            n = df.get(term, 0)
            idf = math.log((N - n + 0.5) / (n + 0.5) + 1)
            tf_score = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
            score += idf * tf_score
        scores.append({"id": doc["id"], "score": score})

    # 步驟4：依分數高到低排序，回傳前 k 筆
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:k]