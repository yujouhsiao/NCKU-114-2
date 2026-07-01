import argparse
import os
import sys
from datetime import datetime
from storage import DataStorage
from manager import ActivityManager

def main():
    # 確保檔案路徑正確鎖定在 v2/ 資料夾內
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "balance_life.db")
    legacy_json = os.path.join(base_dir, "data.json")

    storage = DataStorage(db_path, legacy_json)
    manager = ActivityManager(storage)

    parser = argparse.ArgumentParser(description="Balance Life CLI v2.0 - 飲食與運動健康管理工具")
    subparsers = parser.add_subparsers(dest="command", help="可用指令")

    # --- Add 指令 (向下相容 v1.0) ---
    parser_add = subparsers.add_parser("add", help="新增紀錄")
    parser_add.add_argument("--type", required=True, choices=['B', 'L', 'D', 'S', 'W'], help="類別: B/L/D/S/W")
    parser_add.add_argument("--name", required=True, help="名稱")
    parser_add.add_argument("--price", type=float, default=0.0, help="金額 (飲食類)")
    parser_add.add_argument("--duration", type=int, default=0, help="分鐘數 (運動類)")
    parser_add.add_argument("--date", help="日期 YYYY-MM-DD (預設今日)")
    parser_add.add_argument("--time", help="時間 HH:MM (預設現在)")

    # --- List 指令 (v2.0 強化版) ---
    parser_list = subparsers.add_parser("list", help="列出紀錄")
    parser_list.add_argument("--all", action="store_true", help="顯示所有歷史紀錄")
    parser_list.add_argument("--date", help="查詢特定日期 (YYYY-MM-DD)")
    parser_list.add_argument("--type", dest="category_filter", choices=['B', 'L', 'D', 'S', 'W'], help="依類別篩選")

    # --- Stats 指令 (v2.0 強化版) ---
    parser_stats = subparsers.add_parser("stats", help="統計與目標對照")
    parser_stats.add_argument("--period", choices=['day', 'week'], default='day', help="統計區間")

    # --- Delete 指令 (向下相容 v1.0) ---
    parser_delete = subparsers.add_parser("delete", help="刪除紀錄")
    parser_delete.add_argument("--id", required=True, help="紀錄的唯一識別 ID")

    # --- Goal 指令 (v2.0 全新功能) ---
    parser_goal = subparsers.add_parser("goal", help="設定或查看健康目標")
    parser_goal.add_argument("--budget", type=float, help="每日飲食預算上限")
    parser_goal.add_argument("--workout", type=int, help="每日運動目標時長(分鐘)")

    args = parser.parse_args()

    if args.command == "add":
        manager.add_activity(args.type, args.name, args.price, args.duration)

    elif args.command == "list":
        # 決定查詢日期：若沒選 --all 且沒給日期，預設為「今日」，以觸發摘要格式
        search_date = args.date
        if not args.all and not args.date:
            search_date = datetime.now().strftime("%Y-%m-%d")
        
        manager.list_records(date=search_date, category_filter=args.category_filter, show_all=args.all)

    elif args.command == "stats":
        manager.show_stats(args.period)

    elif args.command == "delete":
        if manager.storage.delete_activity(args.id):
            print(f"Successfully deleted record [{args.id}].")
        else:
            print(f"Error: Record ID {args.id} not found.")
            sys.exit(3)

    elif args.command == "goal":
        if args.budget is not None or args.workout is not None:
            # 如果有傳參數，執行「設定」
            # 這裡簡單處理：若只傳一個，另一個用目前的，或預設 0
            current = storage.get_goals()
            new_budget = args.budget if args.budget is not None else current.get('daily_budget', 0)
            new_workout = args.workout if args.workout is not None else current.get('daily_workout', 0)
            manager.set_goals(new_budget, new_workout)
        else:
            # 沒傳參數，執行「查看」
            manager.show_current_goals()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()