---
name: text2sql-yujouhsiao
description: Convert a natural-language question + SQLite schema into verified SQL, then write the result via scripts/run.py (file-based output contract).
version: 2.0.0
metadata:
  hermes:
    tags: [sql, text2sql, data, aiase]
    category: data
    requires_toolsets: [terminal]
---

# text2sql (file-based output contract)



## When to Use
當使用者輸入 `/text2sql-yujouhsiao {json}` 時觸發。輸入形如
`{"task_id":"...","question":"...","db_schema":"CREATE TABLE ...","dialect":"sqlite"}`。

## Procedure
1. 讀懂 `question` 與 `db_schema`，想出能在該 schema 上執行、答對問題的 **SQLite SQL**（單行）。
2. **使用 `terminal` 工具（不要用 process/background 工具）**，以**絕對路徑**執行
   （`<skill_dir>` 由 Hermes 以 `[Skill directory: ...]` 提供）：
   ```
   python3 <skill_dir>/scripts/run.py \
     --task_id "<task_id>" --sql "<your SQL>" --rationale "<short reason>" --confidence 0.8
   ```
3. 執行 `scripts/run.py` 把最終結果寫入結果檔（路徑取自環境變數 `AIASE_RESULT_PATH`，
   未設定則寫工作目錄 `./aiase_result.json`）。**你不需要在對話訊息中再輸出或複述 JSON。**

## SQL Rules
- **永遠用 alias 限定所有欄位**（尤其 JOIN 時）：`t.name` 而非裸 `name`，避免 "ambiguous column name"。
- **"所有/每個 X 都符合 P"** 用 NOT EXISTS：`NOT EXISTS (SELECT 1 FROM X WHERE NOT P(X))`。
- UNION 各 SELECT 的欄位數必須相同；UNION 自動去重（UNION ALL 不去重）。
- SELECT 選欄位時確認該欄位真實存在於你 JOIN 進來的 table，不要憑空猜測。

## Pitfalls
- SQL 必須單行、語法正確、欄位真的存在於 schema。
- 不要用 `process`/background 工具跑 script；要用同步的 `terminal` 工具。

## Verification
結果檔存在、為合法 JSON object、`task_id` 一致、`sql` 非空、`confidence` 在 0..1。
