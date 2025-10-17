#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, '.')

# 初始化qlib
import qlib
from qlib.config import REG_CN
from qlib.data.dataset.loader import QlibDataLoader

# 使用天勤数据路径
KQ_DATA_DIR = r"C:\Users\ASUS\qlib_data"

# 强制单线程，避免多进程问题
from qlib.config import C
C.kernels = 1
C.joblib_backend = "threading"

qlib.init(provider_uri=KQ_DATA_DIR, region=REG_CN)

def debug_data_loader():
    print("=== 调试 DataLoader ===")
    
    try:
        # 1. 测试简单的特征配置
        simple_config = {
            "feature": (["$open", "$high", "$low", "$close", "$volume"], ["$open", "$high", "$low", "$close", "$volume"]),
            "label": (["Ref($close, -1) / $close - 1"], ["LABEL0"]),
        }
        
        print("1. 测试简单的特征配置...")
        data_loader = QlibDataLoader(
            config=simple_config,
            swap_level=False,
            freq="1min",
            inst_processors=[],
        )
        
        # 测试数据加载
        print("2. 测试数据加载...")
        data = data_loader.load(
            instruments=['SH000300'],
            start_time='2025-05-06',
            end_time='2025-05-07'
        )
        
        if data is not None and not data.empty:
            print(f"✅ 数据加载成功，形状: {data.shape}")
            print(f"列名: {data.columns.tolist()}")
            print(f"索引类型: {type(data.index)}")
            print(f"前5行数据:")
            print(data.head())
        else:
            print("❌ 数据加载失败或为空")
            
        return True
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_data_loader()




