## 1. Skill 簡介
Code Complexity Analyzer — 對任意 Python 原始碼做確定性 AST 分析，輸出圈複雜度、巢狀深度、可維護性分數與等級評估。

## 2. Skill 名稱與目錄
Skill 名稱: `open-mytool-yujouhsiao`
目錄: `skills/open-mytool-yujouhsiao/`
- `SKILL.md`: Hermes skill 指令
- `scripts/run.py`: 純確定性 AST 分析入口（寫結果檔，不印對話 JSON）

## 3. 呼叫方式

評分環境呼叫指令：
```
hermes chat --toolsets skills,terminal --yolo -Q -q '/open-mytool-yujouhsiao {"task_id":"ot_001","code":"def foo():\n    if True:\n        return 1"}'
```

Hermes 會根據 SKILL.md Procedure 呼叫 `scripts/run.py`，結果寫入 `AIASE_RESULT_PATH` 指定檔案。

### 輸入 JSON 格式
```json
{
  "task_id": "complexity_test_001",
  "code": "def calculate(n):\n    if n <= 1:\n        return 1\n    for i in range(n):\n        result = i * 2\n    return result"
}
```

### 實際輸出 JSON（`scripts/run.py` 寫入結果檔）
```json
{
  "task_id": "complexity_test_001",
  "result": {
    "status": "success",
    "metrics": {
      "cyclomatic_complexity": 3,
      "lines_of_code": 6,
      "function_count": 1,
      "max_nesting_level": 1,
      "maintainability_score": 100
    },
    "assessment": "highly_maintainable"
  },
  "confidence": 0.9
}
```

輸出欄位說明：
- `cyclomatic_complexity`: 決策點數（if/for/while/except 各計 1，基礎為 1）
- `lines_of_code`: 非空白、非純注釋行數
- `function_count`: 函數/方法定義數
- `max_nesting_level`: 最深巢狀層數（if/for/while 計入）
- `maintainability_score`: 0–100，cc > 10 每多 1 扣 5 分，nesting > 5 每多 1 扣 10 分
- `assessment`: `highly_maintainable` (≥85) / `maintainable` (≥70) / `moderate_complexity` (≥50) / `complex` (≥25) / `highly_complex` (<25)

## 4. 自定 Verifiable Scenario

### Ground Truth 來源
本 skill 使用 Python `ast` 標準庫做確定性靜態分析。對同一段 Python 程式碼，`ast.parse()` + `NodeVisitor` 的計算結果永遠唯一確定。Ground truth 即為 `scripts/run.py` 對該段程式碼的輸出值，可在任何環境下重現。

### 評分器自動判定方式
評分器讀取結果 JSON，對以下欄位做精確匹配（整數完全一致；`assessment` 字串完全一致；`status` 非 `success` 時按 scenario 要求判定）：
- `result.metrics.cyclomatic_complexity` 精確值
- `result.metrics.max_nesting_level` 精確值
- `result.assessment` 精確字串
- `result.status`（syntax error scenario）精確字串

### Anti-Hardcoding 論證
由於輸出是 Python AST 確定性計算結果，對任何未見過的 Python 程式碼仍可正確計算。Staff perturbation（任何合法/非法 Python 程式碼片段）的正確答案無法預先知道或猜測，只能靠真正執行 AST 分析。「只能對固定 task_id 輸出硬編碼答案」的 skill 對任何不同程式碼輸入均會給出錯誤數值，無法通過評分。

---

### Scenario 1：簡單函數（高可維護性）

**輸入 JSON：**
```json
{
  "task_id": "scenario_1",
  "code": "def greet(name):\n    return f'Hello, {name}'"
}
```

**確定性期望輸出：**
```json
{
  "task_id": "scenario_1",
  "result": {
    "status": "success",
    "metrics": {
      "cyclomatic_complexity": 1,
      "lines_of_code": 2,
      "function_count": 1,
      "max_nesting_level": 0,
      "maintainability_score": 100
    },
    "assessment": "highly_maintainable"
  },
  "confidence": 0.9
}
```

**驗證標準：** `cyclomatic_complexity == 1`，`max_nesting_level == 0`，`assessment == "highly_maintainable"`

---

### Scenario 2：高複雜度函數（中等可維護性）

**輸入 JSON：**
```json
{
  "task_id": "scenario_2",
  "code": "def parse_config(cfg, strict, verbose, validate, retry):\n    if cfg is None:\n        if strict:\n            raise ValueError('cfg required')\n        return {}\n    result = {}\n    for key in cfg:\n        if key == 'host':\n            if strict:\n                if not cfg[key]:\n                    if verbose:\n                        print('empty host')\n                    if retry:\n                        cfg[key] = 'localhost'\n                    else:\n                        return None\n            result['host'] = cfg[key]\n        elif key == 'port':\n            if validate:\n                if not isinstance(cfg[key], int):\n                    if strict:\n                        raise TypeError('port must be int')\n                    continue\n                if cfg[key] < 1 or cfg[key] > 65535:\n                    if strict:\n                        raise ValueError('invalid port')\n                    continue\n            result['port'] = cfg[key]\n        elif key == 'timeout':\n            if cfg[key] <= 0:\n                if strict:\n                    raise ValueError('timeout must be positive')\n                cfg[key] = 30\n            result['timeout'] = cfg[key]\n    return result"
}
```

**確定性期望輸出：**
```json
{
  "task_id": "scenario_2",
  "result": {
    "status": "success",
    "metrics": {
      "cyclomatic_complexity": 18,
      "lines_of_code": 35,
      "function_count": 1,
      "max_nesting_level": 6,
      "maintainability_score": 50
    },
    "assessment": "moderate_complexity"
  },
  "confidence": 0.9
}
```

**驗證標準：** `cyclomatic_complexity == 18`，`max_nesting_level == 6`，`maintainability_score == 50`，`assessment == "moderate_complexity"`

---

### Scenario 3：語法錯誤處理

**輸入 JSON：**
```json
{
  "task_id": "scenario_3",
  "code": "def foo(x\n    return x"
}
```

**確定性期望輸出：**
```json
{
  "task_id": "scenario_3",
  "result": {
    "status": "syntax_error",
    "error": "'(' was never closed (<unknown>, line 1)",
    "metrics": {}
  },
  "confidence": 0.3
}
```

**驗證標準：** `result.status == "syntax_error"`，`confidence == 0.3`，不得 crash 或拋出未捕獲例外

---

### 哪些輸入變化屬於同一任務能力
- 不同的函數名稱、不同的參數名稱：輸出根據真實 AST 計算，與名稱無關
- 等效但排列不同的 if-else 結構：complexity 由決策點數決定，不受順序影響
- 各種不同長度或複雜度的 Python 程式碼：skill 一律用同一公式計算，無需見過該程式碼

## 5. 預期失敗模式

### 失敗 1：LLM 未正確傳遞 code 參數給 scripts/run.py
- **原因**：Hermes LLM 在組合 CLI 指令時，可能截短或錯誤轉義多行字串
- **處理**：`scripts/run.py` 對 `\n` 做自動還原（`replace('\\n', '\n')`）；若 code 為空或缺失，AST parse 仍返回合法結果（0 complexity），不會 crash

### 失敗 2：非 Python 程式碼輸入（類 JavaScript / Java）
- **原因**：非 Python 語法導致 `ast.parse()` 拋出 `SyntaxError`
- **處理**：try-except 捕獲，返回 `status: "syntax_error"`，`confidence: 0.3`，不 crash

### 失敗 3：LLM 不呼叫 run.py 而直接在對話輸出計算結果
- **原因**：SKILL.md 指示不夠明確，LLM 嘗試自行計算而非呼叫確定性工具
- **處理**：SKILL.md Procedure 明確要求「使用 terminal 工具執行 scripts/run.py」，並說明「不需要在對話訊息中再輸出或複述 JSON」

## 6. 互動對象
本 skill 設計為獨立執行，不需要其他 skill 的輸出作為輸入。`scripts/run.py` 做純確定性 AST 分析，LLM 只負責解析輸入 JSON 並呼叫正確的 CLI 指令。

可選擴展互動：Bug Hunter skill 分析某段 code 後，可呼叫本 skill 評估該段 code 的複雜度，作為 bug 報告的補充指標（複雜度高的 code 更容易有 bug）。

## 7. Token Budget 估算

| Scenario | 操作 | 估計 Token |
|----------|------|-----------|
| 簡單代碼（< 20 LOC） | 讀 input → 呼叫 run.py → 完成 | ~300–500 |
| 中等代碼（20–100 LOC） | 讀 input → 呼叫 run.py → 完成 | ~500–1000 |
| 複雜代碼（> 100 LOC） | 讀 input → 呼叫 run.py → 完成 | ~1000–2000 |
| 語法錯誤 | 讀 input → 呼叫 run.py → 完成 | ~300–500 |

**設計說明：** 由於所有計算由確定性的 `scripts/run.py` 完成，LLM 只需要解析輸入並呼叫一次 CLI，Token 消耗穩定、低量。
