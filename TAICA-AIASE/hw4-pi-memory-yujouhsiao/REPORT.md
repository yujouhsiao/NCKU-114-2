# REPORT.md — HW4 Pi Memory

## (1) 如何判斷「記憶有效」、為何指標不可作弊

記憶是否有效不能靠主觀感受判斷，必須用客觀指標量化。
本作業使用 Recall@k、MRR、nDCG@k 三個指標：

- **Recall@k**：前 k 筆結果中，有多少比例的正確答案被找到
- **MRR**：第一筆正確答案排在第幾名的倒數（越前面越好）
- **nDCG@k**：考慮排序位置的綜合分數

這些指標不可作弊的原因：benchmark 的標準答案（relevant_ids）
是事先固定的，retrieve() 必須對任意 query 都正確運作，
無法針對特定 query 硬編碼答案。隱藏測試集會換不同資料驗證，
因此只有真正實作正確的 BM25 才能通過。

## (2) Benchmark 分數與錯誤分析

### 純 BM25
| 指標 | 分數 |
|---|---|
| Recall@5 | 0.810 |
| MRR | 0.810 |
| nDCG@5 | 0.802 |

### Hybrid（BM25 + sentence-transformers all-MiniLM-L6-v2）
| 指標 | 分數 |
|---|---|
| Recall@5 | 0.857 |
| MRR | 0.857 |
| nDCG@5 | 0.857 |

### 純 BM25 接不到的題目分析

| 題目 | 正確答案 | 原因 |
|---|---|---|
| how big can an uploaded picture be? | o26 | 查詢用 "picture"，記憶寫 "image"，字面不同，BM25 無法匹配 → Hybrid 修好 |
| when is the team sync meeting each week? | o24 | "sync meeting" 與記憶實際用詞不匹配，語意差距大 |
| 我要怎麼把新功能慢慢開放給部分使用者？ | o27 | 記憶可能寫 "feature flag"，查詢未使用該詞彙 |
| 怎麼確認我的程式碼風格符合規範？ | o07 | 記憶可能寫 "linter" 或 "eslint"，查詢未使用該詞彙 |

Hybrid 透過語意向量解決了 picture/image 的同義詞問題，
但對於概念差距較大的題目（feature flag、linter）仍有限制。

## (3) 確定性 vs 機率性的分界

- **確定性**：BM25 計分、store 去重、token 預算截斷。
  相同輸入永遠產生相同輸出，可被單元測試客觀驗證。
- **機率性**：本地模型（Gemma）生成 observation 的 summary、
  以及 hybrid 的 embedding 向量（模型權重固定後為確定性，
  但不同模型版本可能有差異）。

設計原則：把「記憶該記什麼、何時想起」設計成確定性邏輯，
只讓模型負責理解語意，不讓它決定排序結果。

## (4) Token 預算取捨

`build_injection()` 預設 token_budget=2000，
照分數高到低依序加入記憶，直到接近上限為止。

取捨邏輯：
- 分數高的記憶優先注入，確保最相關的先進 context
- 超過預算的記憶直接捨棄，避免 context 爆炸
- 本地小模型 context 有限，寧可少注入也不要塞爆

## (5) 與 `/compact` 的關係

| | `/compact` | Pi Memory（本作業） |
|---|---|---|
| 作用範圍 | 單一 session 內壓縮 | 跨 session 持久保存 |
| 儲存位置 | 記憶體（session 結束即消失） | 硬碟 JSON |
| 目的 | 節省當前 context | 下次 session 還記得 |

兩者互補：`/compact` 讓當前對話不爆 context，
Pi Memory 讓重要知識跨 session 保留。

## (6) 本系統 vs 手寫 PROGRESS.md

| | Pi Memory | 手寫 PROGRESS.md |
|---|---|---|
| 更新方式 | 自動 capture | 人工維護 |
| 檢索方式 | BM25/Hybrid 自動找相關 | 人工翻閱 |
| 遺忘問題 | SHA-256 去重，不重複 | 容易忘記更新 |
| 缺點 | capture 品質依賴模型 | 費時但內容可控 |

## (7) 環境記錄

- Python 版本：3.11.5
- 本地模型：qwen2.5:0.5b
- 後端：Ollama（http://localhost:11434）
- VRAM：無獨立 GPU，使用 Intel Iris Xe 共享記憶體
- 作業系統：Windows 11，Dell Inspiron 14 5430
- Embedding 模型：sentence-transformers/all-MiniLM-L6-v2（本地離線）