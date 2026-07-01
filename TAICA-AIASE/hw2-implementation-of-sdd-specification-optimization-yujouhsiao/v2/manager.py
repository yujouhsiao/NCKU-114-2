from typing import List, Optional
from models import ActivityRecord
from storage import DataStorage

class ActivityManager:
    """
    邏輯層 (Logic Layer)
    負責數據的運算、分組以及格式化輸出。
    """
    def __init__(self, storage: DataStorage):
        self.storage = storage
        self.type_map = {
            'B': 'Breakfast 🍳',
            'L': 'Lunch     🍱',
            'D': 'Dinner    🍜',
            'S': 'Snack     🍰',
            'W': 'Workout   💪'
        }

    def add_activity(self, category, name, price, duration):
        record = ActivityRecord(category=category, name=name, price=price, duration_min=duration)
        self.storage.save_activity(record)
        print(f"Added: [{record.id}] {record.name} ({record.category})")

    def list_records(self, date: Optional[str] = None, category_filter: Optional[str] = None, show_all: bool = False):
        """
        清單顯示邏輯 (需求 C & D)
        - 若有篩選或看全部：顯示流水帳 (v1.0 相容)
        - 若是看今日預設：顯示分類摘要格式 (v2.0 新功能)
        """
        records = self.storage.list_activities(date=date, category=category_filter)
        
        if not records:
            print(f"No records found for the specified criteria.")
            return

        # 需求 D: 判斷是否要切換為「分類摘要格式」
        if not show_all and not category_filter and date:
            self._display_summary_format(records, date)
        else:
            self._display_flat_format(records)

    def _display_flat_format(self, records: List[ActivityRecord]):
        """流水帳格式 (v1.0 相容)"""
        print(f"{'ID':<8} {'Type':<10} {'Name':<15} {'Value':<10}")
        print("-" * 45)
        for r in records:
            val = f"${r.price}" if r.category != 'W' else f"{r.duration_min}m"
            print(f"{r.id:<8} {r.category:<10} {r.name:<15} {val:<10}")

    def _display_summary_format(self, records: List[ActivityRecord], date: str):
        """需求 D: 以類別分組的摘要格式"""
        print(f"\n===== Daily Summary ({date}) =====")
        
        # 按類別分組資料
        grouped = {}
        for r in records:
            grouped.setdefault(r.category, []).append(r)

        for cat_code, cat_name in self.type_map.items():
            if cat_code in grouped:
                items = grouped[cat_code]
                subtotal_p = sum(item.price for item in items)
                subtotal_d = sum(item.duration_min for item in items)
                
                print(f"\n[{cat_name}]")
                for item in items:
                    detail = f"${item.price}" if cat_code != 'W' else f"{item.duration_min} min"
                    print(f"  - {item.name:<15} ({detail})")
                
                # 顯示小計
                if cat_code == 'W':
                    print(f"  >> Group Total: {subtotal_d} min")
                else:
                    print(f"  >> Group Total: ${subtotal_p}")
        print("=" * 35)

    def show_stats(self, period: str):
        """
        統計與目標對照 (需求 B)
        """
        import datetime
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        records = self.storage.list_activities(date=today if period == 'day' else None)
        goals = self.storage.get_goals()

        actual_price = sum(r.price for r in records)
        actual_workout = sum(r.duration_min for r in records if r.category == 'W')

        print(f"\n--- Statistics ({period}) ---")
        
        # 顯示金額統計與預算對照
        budget = goals.get('daily_budget')
        budget_str = f"/ ${budget}" if budget else ""
        status_p = ""
        if budget:
            status_p = "✅ OK" if actual_price <= budget else "⚠️ OVER BUDGET"
        print(f"Total Spent:   ${actual_price:<6} {budget_str:<8} {status_p}")

        # 顯示運動統計與目標對照
        target_w = goals.get('daily_workout')
        target_str = f"/ {target_w}m" if target_w else ""
        status_w = ""
        if target_w:
            status_w = "🎯 TARGET REACHED" if actual_workout >= target_w else "🏃 KEEP GOING"
        print(f"Total Workout: {actual_workout:<6} {target_str:<8} {status_w}")

    def set_goals(self, budget: float, workout: int):
        """需求 A: 設定個人目標"""
        self.storage.save_goal(budget, workout)
        print(f"Goals updated: Budget=${budget}, Workout={workout}min")

    def show_current_goals(self):
        """需求 A: 查看目前目標"""
        goals = self.storage.get_goals()
        if not goals:
            print("No goals set yet. Use 'goal' command to set them.")
        else:
            print(f"Current Goals -> Budget: ${goals.get('daily_budget')}, Workout: {goals.get('daily_workout')}min")