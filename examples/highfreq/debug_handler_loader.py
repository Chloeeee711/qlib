#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, '.')

# 初始化qlib
import qlib
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config

# 使用天勤数据路径
KQ_DATA_DIR = r"C:\Users\ASUS\qlib_data"

# 强制单线程，避免多进程问题
from qlib.config import C
C.kernels = 1
C.joblib_backend = "threading"

qlib.init(provider_uri=KQ_DATA_DIR, region=REG_CN)

from simple_handler import SimpleHighFreqHandler

def debug_handler_loader():
    print("=== 调试 Handler DataLoader ===")
    
    try:
        # 1. 创建handler
        print("1. 创建 SimpleHighFreqHandler...")
        h = SimpleHighFreqHandler(
            instruments=['SH000300'],
            start_time='2025-05-06',
            end_time='2025-05-07'
        )
        print("✅ Handler创建成功")
        
        # 2. 检查data_loader配置
        print("2. 检查data_loader配置...")
        print(f"data_loader类型: {type(h.data_loader)}")
        print(f"data_loader配置: {h.data_loader}")
        
        # 3. 直接测试data_loader.load()
        print("3. 直接测试data_loader.load()...")
        data = h.data_loader.load(
            instruments=['SH000300'],
            start_time='2025-05-06',
            end_time='2025-05-07'
        )
        
        if data is not None and not data.empty:
            print(f"✅ data_loader.load()成功，形状: {data.shape}")
            print(f"列名: {data.columns.tolist()}")
            print(f"索引类型: {type(data.index)}")
        else:
            print("❌ data_loader.load()失败或为空")
            
        # 4. 检查get_feature_config()
        print("4. 检查get_feature_config()...")
        config = h.get_feature_config()
        print(f"特征配置: {config}")
        
        return True
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_handler_loader()

