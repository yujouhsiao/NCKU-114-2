import json
import os
import sys
import shutil
from typing import List
from models import ActivityRecord

class DataStorage:
    """
    負責處理 Balance Life CLI 的資料存取與檔案完整性。
    實作資料備份機制與自動排序功能。
    """
    def __init__(self, file_path: str = "data.json"):
        self.file_path = file_path
        self.backup_path = file_path + ".bak"
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """
        初始化檢查：若資料檔不存在，則建立一個空的 JSON 清單。
        """
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def load_all(self) -> List[ActivityRecord]:
        """
        從 JSON 讀取所有紀錄。
        符合 SDD 錯誤處理：若檔案損毀，則輸出錯誤並以 退出碼 4 結束。
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
                return [ActivityRecord.from_dict(d) for d in data_list]
        except (json.JSONDecodeError, KeyError, TypeError):
            # 實作 SDD 定義的錯誤碼 4
            print("Error: Data file corrupted. Please check or delete data.json.")
            sys.exit(4)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            sys.exit(5)

    def save_all(self, records: List[ActivityRecord]):
        """
        將紀錄存回 JSON。
        包含自動排序功能：先按日期排序，同日期則按時間排序。
        包含備份機制：在寫入前先產生備份檔，防止寫入中斷導致資料遺失。
        """
        # 數據人必備：多欄位排序邏輯 (Lambda Sorting)
        sorted_records = sorted(
            records, 
            key=lambda x: (x.date, x.time)
        )

        # 1. 建立備份
        if os.path.exists(self.file_path):
            shutil.copy2(self.file_path, self.backup_path)

        # 2. 轉換為字典清單並寫入
        dict_records = [r.to_dict() for r in sorted_records]
        
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(dict_records, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving data: {e}")
            sys.exit(5)

    def add_record(self, record: ActivityRecord):
        """
        輔助方法：新增單筆紀錄並立即存檔。
        """
        records = self.load_all()
        records.append(record)
        self.save_all(records)

    def delete_record_by_id(self, record_id: str) -> bool:
        """
        輔助方法：根據 ID 刪除紀錄。
        符合 SDD 錯誤處理：若找不到 ID，會由調用端決定是否觸發退出碼 3。
        """
        records = self.load_all()
        initial_count = len(records)
        
        # 過濾掉指定 ID 的紀錄
        records = [r for r in records if r.id != record_id]
        
        if len(records) < initial_count:
            self.save_all(records)
            return True
        return False