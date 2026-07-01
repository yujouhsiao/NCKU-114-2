import sys
from datetime import datetime, timedelta
from typing import List, Optional
from models import ActivityRecord
from storage import DataStorage

class ActivityManager:
    """
    Balance Life CLI 的邏輯核心。
    處理活動紀錄的篩選、統計計算與刪除邏輯。
    """
    def __init__(self, storage: DataStorage):
        self.storage = storage

    def add_activity(self, name: str, category: str, price: float, duration: int, 
                     date: Optional[str] = None, time: Optional[str] = None):
        """
        建立新的活動紀錄並存入資料庫。
        """
        # 如果使用者沒輸入日期時間，models.py 會自動生成當下的
        record = ActivityRecord(
            name=name,
            category=category,
            price=price,
            duration_min=duration
        )
        if date: record.date = date
        if time: record.time = time
        
        # 執行 __post_init__ 驗證 (這會處理退出碼 1 & 2)
        record.__post_init__()
        
        self.storage.add_record(record)
        
        # 符合 SDD 測試案例 1 的 stdout 格式
        status_msg = f"($ {record.price})" if record.duration_min == 0 else f"({record.duration_min} min)"
        print(f"Added: [{record.id}] {record.name} {status_msg}")

    def list_activities(self, target_date: Optional[str] = None):
        """
        列出所有紀錄或指定日期的紀錄。
        """
        records = self.storage.load_all()
        
        if target_date:
            records = [r for r in records if r.date == target_date]
        
        if not records:
            print("No records found.")
            return

        # 專業表格輸出 (增加程式碼細節)
        print("-" * 65)
        print(f"{'ID':<8} {'Date':<12} {'Time':<8} {'Type':<12} {'Name':<15} {'Info'}")
        print("-" * 65)
        for r in records:
            info = f"${r.price:<7}" if r.category != "Workout" else f"{r.duration_min} min"
            print(f"{r.id:<8} {r.date:<12} {r.time:<8} {r.category:<12} {r.name:<15} {info}")
        print("-" * 65)

    def get_stats(self, period: str):
        """
        計算統計數據：總花費與總運動時長。
        符合 SDD 測試案例 3 的輸出要求。
        """
        records = self.storage.load_all()
        now = datetime.now()
        
        if period == "day":
            target_str = now.strftime("%Y-%m-%d")
            filtered = [r for r in records if r.date == target_str]
            label = f"Daily Total ({target_str})"
        elif period == "week":
            # 取得過去 7 天的紀錄
            seven_days_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d")
            filtered = [r for r in records if r.date >= seven_days_ago]
            label = "Weekly Total (Last 7 days)"
        else:
            print("Error: Invalid period. Use 'day' or 'week'.")
            return

        total_spend = sum(r.price for r in filtered)
        total_workout = sum(r.duration_min for r in filtered)

        print(f"=== {label} ===")
        print(f"Total Spending : ${total_spend:.2f}")
        print(f"Total Workout  : {total_workout} min")
        print("==========================")

    def delete_activity(self, record_id: str):
        """
        根據 ID 刪除紀錄。
        符合 SDD 錯誤處理：若找不到 ID 則以 退出碼 3 結束。
        """
        success = self.storage.delete_record_by_id(record_id)
        if success:
            print(f"Successfully deleted record [{record_id}].")
        else:
            # 實作 SDD 定義的錯誤碼 3
            print(f"Error: Record ID {record_id} not found.")
            sys.exit(3)