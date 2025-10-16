#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, '.')

# 初始化qlib
import qlib
from qlib.config import REG_CN
from qlib.data import D

# 使用天勤数据路径
KQ_DATA_DIR = r"C:\Users\ASUS\qlib_data"

# 强制单线程，避免多进程问题
from qlib.config import C
C.kernels = 1
C.joblib_backend = "threading"

qlib.init(provider_uri=KQ_DATA_DIR, region=REG_CN)

def debug_data_loading():
    print("=== 调试数据加载 ===")
    
    try:
        # 1. 检查可用的股票
        print("1. 检查可用的股票...")
        instruments = D.instruments("all")
        print(f"可用股票数量: {len(instruments)}")
        print(f"股票类型: {type(instruments)}")
        print(f"股票内容: {instruments}")
        
        # 2. 检查特定股票的数据 - 使用实际存在的股票
        test_inst = "SH000300"  # 使用沪深300指数
        print(f"\n2. 检查股票 {test_inst} 的数据...")
        
        # 3. 尝试加载数据
        print("3. 尝试加载OHLCV数据...")
        df = D.features(
            [test_inst], 
            ["$open", "$high", "$low", "$close", "$volume"],
            start_time="2025-05-06", 
            end_time="2025-05-07", 
            freq="1min"
        )
        
        if df is not None and not df.empty:
            print(f"✅ 数据加载成功，形状: {df.shape}")
            print(f"索引类型: {type(df.index)}")
            print(f"列名: {df.columns.tolist()}")
            print(f"前5行数据:")
            print(df.head())
        else:
            print("❌ 数据加载失败或为空")
            
        # 4. 检查时间范围
        print("\n4. 检查时间范围...")
        try:
            cal = D.calendar(start_time="2025-05-06", end_time="2025-05-07", freq="1min")
            print(f"交易日历: {cal}")
        except Exception as e:
            print(f"交易日历获取失败: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_data_loading()
