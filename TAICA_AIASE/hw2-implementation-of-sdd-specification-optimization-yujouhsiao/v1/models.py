import uuid
import sys
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ActivityRecord:
    """
    Balance Life CLI 的核心資料模型。
    負責儲存單筆紀錄並進行嚴格的格式驗證。
    """
    name: str
    category: str  # B, L, D, S, W
    price: float = 0.0
    duration_min: int = 0
    # 預設使用當前日期與時間
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    time: str = field(default_factory=lambda: datetime.now().strftime("%H:%M"))
    # 自動生成 6 碼唯一識別碼
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:6])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """
        在物件初始化後，自動執行規格書 (SDD) 定義的驗證邏輯。
        """
        self._validate_values()
        self._validate_formats()
        self._standardize_category()

    def _validate_values(self):
        """
        驗證數值：金額與時長不可為負數。
        符合 SDD 錯誤處理：退出碼 1。
        """
        if self.price < 0 or self.duration_min < 0:
            print("Error: Value cannot be negative.")
            sys.exit(1)

    def _validate_formats(self):
        """
        驗證格式：日期與時間必須符合 YYYY-MM-DD 與 HH:MM。
        符合 SDD 錯誤處理：退出碼 2。
        """
        try:
            datetime.strptime(self.date, "%Y-%m-%d")
            datetime.strptime(self.time, "%H:%M")
        except ValueError:
            print("Error: Invalid format.")
            sys.exit(2)

    def _standardize_category(self):
        """
        將簡寫 (B/L/D/S/W) 轉換為完整名稱，增加資料可讀性。
        """
        mapping = {
            'B': 'Breakfast',
            'L': 'Lunch',
            'D': 'Dinner',
            'S': 'Snack',
            'W': 'Workout'
        }
        # 如果使用者輸入簡寫，自動轉換；否則保持原樣
        self.category = mapping.get(self.category.upper(), self.category)

    def to_dict(self) -> Dict[str, Any]:
        """將物件轉換為字典，方便存入 JSON 檔案"""
        return {
            "id": self.id,
            "category": self.category,
            "name": self.name,
            "price": self.price,
            "duration_min": self.duration_min,
            "date": self.date,
            "time": self.time,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActivityRecord':
        """從字典資料還原 ActivityRecord 物件"""
        record = cls(
            name=data["name"],
            category=data["category"],
            price=data["price"],
            duration_min=data["duration_min"],
            date=data["date"],
            time=data["time"]
        )
        # 覆蓋自動生成的 ID 與時間戳記，確保資料一致
        record.id = data["id"]
        record.created_at = data["created_at"]
        return record