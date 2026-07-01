import sqlite3
import json
import os
from typing import List, Optional, Dict
from models import ActivityRecord, Goal  # 確保 models.py 已更新 Goal 類別

class DataStorage:
    """
    資料存取層 (Data Access Layer)
    負責處理 SQLite 資料庫的讀寫，並支援從 v1.0 JSON 格式遷移數據。
    """
    def __init__(self, db_path: str = "balance_life.db", legacy_path: str = "data.json"):
        self.db_path = db_path
        self.legacy_path = legacy_path
        self._init_db()
        self._migrate_from_v1()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """初始化 SQLite 資料表：活動紀錄與個人目標"""
        with self._get_connection() as conn:
            # 建立活動紀錄表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS activities (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    name TEXT NOT NULL,
                    price REAL NOT NULL,
                    duration_min INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL
                )
            ''')
            # 建立個人目標表 (需求 A)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS goals (
                    key TEXT PRIMARY KEY,
                    value REAL NOT NULL
                )
            ''')
            conn.commit()

    def _migrate_from_v1(self):
        """向下相容：若偵測到舊版 data.json，自動遷移至 SQLite"""
        if os.path.exists(self.legacy_path):
            try:
                with open(self.legacy_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        # 避免重複匯入，使用 INSERT OR IGNORE
                        self.save_activity(ActivityRecord(**item))
                # 遷移成功後，將舊檔案重新命名備份，避免重複觸發
                os.rename(self.legacy_path, f"{self.legacy_path}.bak")
                print(f"[*] Data migrated from {self.legacy_path} successfully.")
            except Exception as e:
                print(f"Error during migration: {e}")

    # --- 活動紀錄相關功能 ---

    def save_activity(self, record: ActivityRecord):
        """存入一筆活動紀錄"""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO activities (id, category, name, price, duration_min, date, time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (record.id, record.category, record.name, record.price, 
                  record.duration_min, record.date, record.time))

    def list_activities(self, date: Optional[str] = None, category: Optional[str] = None) -> List[ActivityRecord]:
        """
        支援篩選的清單查詢 (需求 C)
        category: 傳入 'W', 'L', 'B' 等進行篩選，若為 None 則顯示全部。
        """
        query = "SELECT * FROM activities WHERE 1=1"
        params = []
        
        if date:
            query += " AND date = ?"
            params.append(date)
        if category:
            query += " AND category = ?"
            params.append(category)
            
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [ActivityRecord(id=r[0], category=r[1], name=r[2], price=r[3], 
                                   duration_min=r[4], date=r[5], time=r[6]) for r in rows]

    def delete_activity(self, record_id: str) -> bool:
        """根據 ID 刪除紀錄"""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM activities WHERE id = ?", (record_id,))
            return cursor.rowcount > 0

    # --- 目標設定相關功能 (需求 A) ---

    def save_goal(self, budget: float, workout_min: int):
        """儲存每日預算與運動目標"""
        with self._get_connection() as conn:
            conn.execute("INSERT OR REPLACE INTO goals (key, value) VALUES ('daily_budget', ?)", (budget,))
            conn.execute("INSERT OR REPLACE INTO goals (key, value) VALUES ('daily_workout', ?)", (workout_min,))

    def get_goals(self) -> Dict[str, float]:
        """讀取目前設定的目標值"""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT key, value FROM goals")
            return dict(cursor.fetchall())