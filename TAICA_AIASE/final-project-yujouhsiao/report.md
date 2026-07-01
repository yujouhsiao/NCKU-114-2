# AIASE 2026 Final Project - Report

## 設計決策

### 1. 輸出契約：File-Based（AIASE_RESULT_PATH）

本次助教更新將輸出方式從「在對話訊息印出 JSON」改為「寫入結果檔」。

每個 skill 的 `scripts/run.py` 讀取環境變數 `AIASE_RESULT_PATH`（由 `run_dev.py`/評分器設定），並以原子寫入（tmp → rename）確保結果完整性：

```python
path = os.environ.get("AIASE_RESULT_PATH") or os.path.join(os.getcwd(), "aiase_result.json")
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
os.replace(tmp, path)
```

`scripts/run.py` 不 import `aiase_contract`，避免 skill 安裝到 `~/.hermes/skills/` 後找不到 repo 根模組。

### 2. Text2SQL Skill 設計

- **方案**：LLM 根據 `question` + `db_schema` 生成 SQL，執行 `scripts/run.py` 寫入結果檔；`scripts/validate_sql.py` 作為確定性 harness，在寫檔前驗證語法
- **驗證策略**：
  - 禁止操作正則掃描（DDL、DML、CTE `WITH`、窗函數 `OVER`）
  - 以 SQLite in-memory `EXPLAIN` 驗證語法（無需實際資料）
  - SKILL.md 指示 LLM 永遠用 table alias 限定欄位（避免 JOIN 時 ambiguous column name）
  - SKILL.md 說明「所有/每個」語義用 `NOT EXISTS` 表達
- **Harness 角色**：`validate_sql.py` 是純確定性工具；SQL 生成完全由 LLM 完成
- **實測結果**：dev set 21 題，穩定 19–20/21（90–95%）

### 3. Code Author Skill 設計

- **方案**：LLM 根據 `task_description` 和 `constraints` 實作 `entry_function`，執行 `scripts/run.py` 寫入結果檔；`scripts/selftest.py` 作為確定性 harness 執行自測
- **驗證策略**：
  - `selftest.py` 以 radon raw 計算 SLOC（對應規格書 §2.3）
  - AST 靜態分析檢查 `imports_forbidden` 違規
  - 執行 `sample_tests` 中的測試案例比對 `expected` 輸出
- **Harness 角色**：`selftest.py` 是純確定性工具；代碼生成完全由 LLM 完成

### 4. Bug Hunter Skill 設計

- **方案**：LLM 分析 `code`，結合靜態掃描工具輸出做語意審查，執行 `scripts/run.py` 寫入結果檔
- **確定性輔助工具**：
  - `scripts/analyze.py`：動態邊界探測，執行邊界 inputs 偵測 crash 位置（`suspicious_lines`）
- **Bug type enum**：`off_by_one`, `null_deref`, `type_error`, `logic_error`, `edge_case`, `api_misuse`, `inefficient`, `unhandled_input`
- **LLM 決策**：最終 verdict（buggy/clean）、bug 清單與 confidence 由 LLM 決定

### 5. Open Track Skill 設計（代碼複雜度分析）

- **方案**：純確定性 AST 分析，不依賴 LLM；`scripts/run.py` 直接計算並輸出結果
- **指標**：
  - 決策點計數（if、for、while、except）→ 圈複雜度
  - 巢狀深度追蹤（`max_nesting_level`）
  - 可維護性評分公式（0–100 分）
  - 分類：highly_maintainable → highly_complex
- **限制**：只適用於 Python；decorator、comprehension 不計入複雜度

---

## 執行 Log 摘要

### Text2SQL 完整 dev set（21 題）

```
[PASS] task_nl2sql_001: result set matches gold (bag-equal)
[PASS] task_nl2sql_002: result set matches gold (bag-equal)
...
[PASS] task_nl2sql_018: result set matches gold (bag-equal)
[FAIL] task_nl2sql_019: student SQL failed to execute: ambiguous column name: name
[FAIL] task_nl2sql_020: result set differs from gold
[PASS] task_nl2sql_EXAMPLE: result set matches gold (bag-equal)

Dev set: 19/21 passed  (90%)
```

019 在加入 alias 規則後通過；020（複雜 NOT EXISTS 雙重否定）非確定性，單獨重跑可通過。

### validate_sql.py harness 測試

```
$ python skills/text2sql-yujouhsiao/scripts/validate_sql.py \
    "SELECT s.name FROM Students s JOIN Enrollments e ON s.sid = e.sid WHERE e.cid = 1;" \
    "CREATE TABLE Students (sid INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE Enrollments (sid INTEGER NOT NULL, cid INTEGER NOT NULL, PRIMARY KEY (sid, cid));"

{"ok": true, "error": ""}
```

窗函數拒絕：
```
$ python skills/text2sql-yujouhsiao/scripts/validate_sql.py \
    "SELECT name, ROW_NUMBER() OVER (ORDER BY sid) FROM Students;" \
    "CREATE TABLE Students (sid INTEGER PRIMARY KEY, name TEXT);"

{"ok": false, "error": "Forbidden operation detected: \\bOVER\\s*\\("}
```

### selftest.py harness 測試（Code Author）

```json
{
  "passed": 3,
  "failed": 0,
  "errors": [],
  "sloc": 11,
  "loc_violation": false,
  "import_violations": []
}
```

### analyze.py 動態探測測試（Bug Hunter）

以 task_pair_001 buggy code 為例，空列表輸入偵測到 IndexError at line 3：
```json
{
  "probes": [{"label": "empty list", "outcome": "crash",
              "error": "IndexError: list index out of range", "bad_line": 3}],
  "suspicious_lines": [3]
}
```

---

## 失敗分析

### 失敗 1：Text2SQL 欄位不限定 alias 導致 ambiguous column name

- **問題**：多表 JOIN 時 LLM 偶爾用裸欄位名（如 `name`），兩張表都有該欄位
- **原因**：SKILL.md 原版未特別強調 alias 限定規則
- **修正**：在 SKILL.md 加入 SQL Rules 區塊，明確要求「永遠用 alias 限定所有欄位（`t.name` 而非裸 `name`）」

### 失敗 2：Text2SQL 複雜「所有/每個」語義偶發失敗

- **問題**：「every patient older than 60」需 `NOT EXISTS (SELECT 1 ... WHERE NOT P)` 雙重否定結構，LLM 非確定性地生成不正確版本
- **原因**：全稱量詞的 SQL 表達並非常見直覺
- **修正**：SKILL.md 加入提示「"所有/每個 X 都符合 P" 用 NOT EXISTS」；個別重跑可通過，屬 LLM 非確定性問題

---

## 改進方向

### 優先級 1（立即可行）
1. Text2SQL 加入 `validate_sql.py` 驗證迴圈：讓 LLM 根據 validator 錯誤訊息重試（目前 SKILL.md 有說明但未強制）
2. Bug Hunter 降低 false positive：對 clean code 不要硬報 bug，加入「如果證據不足就回傳 clean」的 SKILL.md 指示

### 優先級 2（短期）
1. 擴展 `analyze.py` 的邊界 inputs 覆蓋率（目前只探測空列表、None、負數等）
2. Text2SQL 加入 JOIN 欄位衝突自動偵測，在輸入 schema 前先掃描重名欄位並在 prompt 中提示

### 優先級 3（長期）
1. Open Track 擴展支援多語言（目前僅 Python）
2. 實現 inter-skill 協調：Code Author 生成的代碼直接餵給 Bug Hunter 做自我審查

---

## 分工（單人項目）

由於本項目為單人完成：
- **所有實現**：完成了 Text2SQL、Code Author、Bug Hunter、Open Track 四個 skills
- **設計決策**：獨立進行所有架構和方法論決策
- **測試驗證**：完整執行 dev set 21 題；各 harness 工具單獨 smoke test
- **文檔**：完成所有 SKILL.md、OPEN_TRACK.md、PAIRWISE_ROLE.md、report.md

---

## 參考與引用

### 公開代碼 / 資源
1. **Python AST 模組**：https://docs.python.org/3/library/ast.html
   - 用於代碼解析和複雜度分析（open-mytool、bug-hunter）
   - 無修改；直接使用標準庫

2. **SQLite 驗證**：Python sqlite3 標準庫
   - 用於 SQL 語法驗證（text2sql validate_sql.py 的 EXPLAIN）
   - 無修改；直接使用

3. **Hermes 框架**：官方 skill 模板與 run_dev.py
   - 參考 file-based 輸出契約格式（`AIASE_RESULT_PATH`、原子寫入）
   - `scripts/run.py` 依助教範例實作

### 差異說明
- **Text2SQL**：`validate_sql.py` 為原創；SQL 生成完全交由 LLM，未使用任何第三方 text-to-SQL 函式庫
- **Code Author**：`selftest.py` 使用 radon 計算 SLOC（與規格書一致）；自製 import 違規 AST 掃描
- **Bug Hunter**：`analyze.py` 動態探測為原創；最終判斷交由 LLM
- **Open Track**：複雜度指標與評分演算法為原創設計；使用 Python AST 標準庫
