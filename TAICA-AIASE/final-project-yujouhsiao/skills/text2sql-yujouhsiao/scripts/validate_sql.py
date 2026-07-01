#!/usr/bin/env python3
"""
text2sql validate_sql.py: SQL 語法與欄位驗證工具

驗證給定的 SQL 是否：
1. 合法的 SQLite 語法（可以 EXPLAIN）
2. 引用的表與欄位存在於 schema 中
3. 不使用禁止的操作（如 DDL、DML、CTE 等）

用法：
  python validate_sql.py "<sql>" "<schema_ddl>"

輸出：
  JSON: {"ok": true/false, "error": "..."}
"""
import sqlite3
import sys
import json
import re

def check_forbidden_operations(sql: str):
    """檢查是否使用禁止的操作"""
    forbidden_patterns = [
        r'\b(CREATE|DROP|ALTER|INSERT|UPDATE|DELETE|PRAGMA|ATTACH|DETACH)\b',
        r'\bWITH\b',  # CTE
        r'\bWINDOW\b',  # named window clause
        r'\bOVER\s*\(',  # window function OVER(...)
    ]
    sql_upper = sql.upper()
    for pattern in forbidden_patterns:
        if re.search(pattern, sql_upper):
            return False, f"Forbidden operation detected: {pattern}"
    return True, ""

def validate_sql(sql: str, schema_ddl: str):
    """驗證 SQL 語法與合法性"""
    # 檢查禁止操作
    ok, err = check_forbidden_operations(sql)
    if not ok:
        return False, err
    
    # 嘗試在 in-memory SQLite 中執行 EXPLAIN
    con = sqlite3.connect(":memory:")
    try:
        # 建立 schema 但不插入資料
        con.executescript(schema_ddl)
        
        # 使用 EXPLAIN 驗證語法（不執行實際查詢）
        con.execute(f"EXPLAIN {sql}")
        
        return True, ""
    except sqlite3.Error as e:
        return False, str(e)
    finally:
        con.close()

def main():
    if len(sys.argv) < 3:
        result = {"ok": False, "error": "usage: python validate_sql.py '<sql>' '<schema_ddl>'"}
        print(json.dumps(result))
        sys.exit(1)
    
    sql = sys.argv[1]
    schema_ddl = sys.argv[2].replace('\\n', '\n')
    
    ok, err = validate_sql(sql, schema_ddl)
    result = {"ok": ok, "error": err}
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if ok else 2)

if __name__ == '__main__':
    main()
