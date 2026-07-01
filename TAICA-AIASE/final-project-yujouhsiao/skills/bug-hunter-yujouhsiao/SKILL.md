---
name: bug-hunter-yujouhsiao
description: Pairwise Bug Hunter. Given code + task, produce a structured bug report and write the result via scripts/run.py (file-based output contract).
version: 2.0.0
metadata:
  hermes:
    tags: [bug, review, pairwise, aiase]
    category: aiase
    requires_toolsets: [terminal]
---

# bug-hunter (file-based output contract)



## When to Use
當使用者輸入 `/bug-hunter-yujouhsiao {json}` 時觸發。輸入含
`task_id` / `task_description` / `code` / `constraints` / `sample_inputs`。

## Procedure
1. 分析 `code`：先用靜態檢查 + 自動 edge-case 試跑抓低 hanging fruit，再對可疑點做語意分析。
2. 整理成 bug 清單（每個 bug：`line_start`/`line_end`/`severity`/`type`/`description`/`suggested_fix`）。
   合法 `type`：off_by_one · null_deref · type_error · logic_error · edge_case · api_misuse · inefficient · unhandled_input。
   `severity`：critical · high · medium · low。
3. **使用 `terminal` 工具（不要用 process/background 工具）**，以**絕對路徑**執行
   （`--bugs` 傳入一段 JSON 陣列字串）：
   ```
   python3 <skill_dir>/scripts/run.py --task_id "<task_id>" \
     --verdict "buggy" --confidence 0.75 \
     --bugs '[{"line_start":8,"line_end":8,"severity":"high","type":"edge_case","description":"...","suggested_fix":"..."}]'
   ```
4. 執行 `scripts/run.py` 把結果寫入結果檔（路徑取自 `AIASE_RESULT_PATH`，未設定則 `./aiase_result.json`）。
   **你不需要在對話訊息中再輸出或複述 JSON。**

## Pitfalls
- 對乾淨 code 不要硬報 bug（false positive 扣分）；`line_start` 是 1-indexed。
- `suggested_fix` 要具體，不要寫「Fix the bug」這種空話。

## Verification
結果檔存在、為合法 JSON object、含 `task_id` / `verdict` / `bugs`（陣列）/ `confidence`。
