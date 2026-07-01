import argparse
import sys
from models import ActivityRecord
from storage import DataStorage
from manager import ActivityManager

def print_banner():
    """印出專業的 CLI 標題"""
    banner = """
    ==========================================
              BALANCE LIFE CLI v1.0
       Optimize Your Meals & Workouts 
    ==========================================
    """
    print(banner)

def main():
    # 1. 初始化底層組件
    storage = DataStorage("data.json")
    manager = ActivityManager(storage)

    # 2. 設定 argparse 主解析器
    parser = argparse.ArgumentParser(
        description="Balance Life CLI: 追蹤三餐花費與運動時長的利器"
    )
    subparsers = parser.add_subparsers(dest="command", help="可用指令")

    # --- Sub-command: add ---
    add_parser = subparsers.add_parser("add", help="新增一筆紀錄")
    add_parser.add_argument("--name", required=True, help="食物或運動名稱")
    add_parser.add_argument("--price", type=float, default=0.0, help="金額 (飲食必填)")
    add_parser.add_argument("--duration", type=int, default=0, help="運動時長 (分鐘)")
    add_parser.add_argument("--type", required=True, choices=['B', 'L', 'D', 'S', 'W'], 
                           help="類別: B(早), L(午), D(晚), S(點心), W(運動)")
    add_parser.add_argument("--date", help="日期 (YYYY-MM-DD), 預設今日")
    add_parser.add_argument("--time", help="時間 (HH:MM), 預設當下")

    # --- Sub-command: list ---
    list_parser = subparsers.add_parser("list", help="列出紀錄")
    list_parser.add_argument("--all", action="store_true", help="列出所有紀錄")
    list_parser.add_argument("--date", help="列出特定日期的紀錄 (YYYY-MM-DD)")

    # --- Sub-command: stats ---
    stats_parser = subparsers.add_parser("stats", help="統計數據")
    stats_parser.add_argument("--period", choices=["day", "week"], default="day", 
                             help="統計週期: day(今日) 或 week(過去 7 天)")

    # --- Sub-command: delete ---
    del_parser = subparsers.add_parser("delete", help="刪除紀錄")
    del_parser.add_argument("--id", required=True, help="要刪除的紀錄唯一 ID")

    # 3. 解析參數
    args = parser.parse_args()

    # 4. 根據指令執行相對應的邏輯
    if args.command == "add":
        manager.add_activity(
            name=args.name,
            category=args.type,
            price=args.price,
            duration=args.duration,
            date=args.date,
            time=args.time
        )
    
    elif args.command == "list":
        if args.all:
            manager.list_activities()
        elif args.date:
            manager.list_activities(target_date=args.date)
        else:
            # 預設列出今日
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            manager.list_activities(target_date=today)

    elif args.command == "stats":
        manager.get_stats(args.period)

    elif args.command == "delete":
        manager.delete_activity(args.id)

    else:
        # 如果沒輸入指令，印出 Banner 與 Help
        print_banner()
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(5)