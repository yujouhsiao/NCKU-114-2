---
name: open-mytool-yujouhsiao
description: Analyze Python code metrics (cyclomatic complexity, maintainability, nesting depth).
version: 1.0.0
metadata:
  hermes:
    tags: [code, metrics, analysis]
    category: open
---

# Open Track: Code Complexity Analyzer

## When to Use
When given Python source code and need to analyze code quality metrics including cyclomatic complexity, lines of code, function count, nesting levels, and maintainability score.

## Procedure
1. Parse the input JSON for `task_id` and `code`.
2. **使用 `terminal` 工具（不要用 process/background 工具）**，以**絕對路徑**執行
   （`<skill_dir>` 由 Hermes 以 `[Skill directory: ...]` 提供）：
   ```
   python3 <skill_dir>/scripts/run.py --task_id "<task_id>" --code "<code>"
   ```
3. `scripts/run.py` 用 AST 確定性分析代碼複雜度，並把結果寫入結果檔
   （路徑取自環境變數 `AIASE_RESULT_PATH`，未設定則寫工作目錄 `./aiase_result.json`）。
   **你不需要在對話訊息中再輸出或複述 JSON。**

## Pitfalls
- `--code` 傳入時 `\n` 會被 `run.py` 自動還原為換行符。
- 非 Python 代碼輸入會讓 AST 解析失敗，`run.py` 會回傳 `status: syntax_error` 而非 crash。
- 不要用 `process`/background 工具跑 script；要用同步的 `terminal` 工具。

## Verification
- Output JSON must include all metrics: cyclomatic_complexity, lines_of_code, function_count, max_nesting_level, maintainability_score.
- Assessment classification must be one of: highly_maintainable, maintainable, moderate_complexity, complex, highly_complex.
- Must handle syntax errors gracefully with status: syntax_error.

