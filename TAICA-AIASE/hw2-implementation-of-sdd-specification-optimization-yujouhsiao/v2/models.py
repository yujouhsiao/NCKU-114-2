import uuid
import sys
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ActivityRecord:
    """
    活動紀錄模型
    儲存飲食 (B/L/D/S) 或運動 (W) 的詳細資訊。
    """
    category: str
    name: str
    price: float = 0.0
    duration_min: int = 0
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:6])
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    time: str = field(default_factory=lambda: datetime.now().strftime("%H:%M"))

    def __post_init__(self):
        """驗證邏輯：確保數值不為負數 (向下相容 v1.0 規格)"""
        if self.price < 0 or self.duration_min < 0:
            print("Error: Value cannot be negative.")
            sys.exit(1)

@dataclass
class Goal:
    """
    個人化目標模型 (v2.0 新增需求)
    儲存每日預算上限與運動時長目標。
    """
    daily_budget: float
    daily_workout_min: int

    def __post_init__(self):
        """驗證邏輯：確保目標值不為負數"""
        if self.daily_budget < 0 or self.daily_workout_min < 0:
            print("Error: Goal value cannot be negative.")
            sys.exit(5)  # 根據 sdd_v2 規劃的退出碼 5