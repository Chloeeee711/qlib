# ============================================
# 系统启动器
# 统一管理所有模块
# ============================================

import os
import sys
from pathlib import Path

def main():
    print("="*70)
    print("HIGH-FREQUENCY TRADING SYSTEM")
    print("="*70)
    print("请选择要启动的模块:")
    print()
    print("[1] 训练/回测（使用历史数据）")
    print("     - 下载历史数据")
    print("     - 训练模型")
    print("     - 回测分析")
    print()
    print("[2] 实时数据更新（交易时间运行）")
    print("     - 实时路径: qlib_data_recent")
    print("     - 每分钟更新")
    print("     - 收盘测试: python tqsdk_realtime_updater.py --test")
    print("     - 开盘正式: python tqsdk_realtime_updater.py")
    print()
    print("[3] 预测调度器（14:40预测）")
    print("     - 使用实时数据")
    print("     - 收盘测试: python prediction_scheduler.py --test")
    print("     - 正式运行: python prediction_scheduler.py")
    print()
    print("[4] 邮件发送（14:45买入，10:46卖出）")
    print("     - 读取CSV信号")
    print("     - 定时发送邮件")
    print()
    print("[5] 快速测试（收盘后验证所有功能）")
    print()
    print("="*70)
    
    choice = input("\n请输入选择 (1-5): ").strip()
    
    if choice == "1":
        print("\n[START] 启动训练/回测...")
        print("[INFO] 使用历史数据路径: qlib_data")
        os.system("python highfreq_workflow.py")
        
    elif choice == "2":
        print("\n[START] 启动实时数据更新...")
        print("[INFO] 使用实时数据路径: qlib_data_recent")
        mode = input("选择模式 (1=测试, 2=正式): ").strip()
        if mode == "1":
            os.system("python tqsdk_realtime_updater.py --test")
        else:
            os.system("python tqsdk_realtime_updater.py")
        
    elif choice == "3":
        print("\n[START] 启动预测调度器...")
        print("[INFO] 使用实时数据路径: qlib_data_recent")
        mode = input("选择模式 (1=测试, 2=正式): ").strip()
        if mode == "1":
            os.system("python prediction_scheduler.py --test")
        else:
            os.system("python prediction_scheduler.py")
        
    elif choice == "4":
        print("\n[START] 启动邮件发送器...")
        print("[INFO] 读取: daily_predictions.csv")
        os.system("python email_sender.py")
        
    elif choice == "5":
        print("\n[TEST] 快速测试模式...")
        print("="*70)
        print("[STEP 1] 测试实时数据更新器...")
        os.system("python tqsdk_realtime_updater.py --test")
        print("\n[STEP 2] 测试预测调度器...")
        os.system("python prediction_scheduler.py --test")
        print("\n[OK] 所有测试完成！")
        
    else:
        print("[ERROR] 无效选择")
    
    print("\n" + "="*70)
    print("[INFO] 使用说明:")
    print("  - 历史数据路径: qlib_data (用于训练)")
    print("  - 实时数据路径: qlib_data_recent (用于预测)")
    print("  - 收盘后使用 --test 模式验证")
    print("  - 开盘后正式运行")
    print("="*70)

if __name__ == "__main__":
    main()