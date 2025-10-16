#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import qlib
from qlib.constant import REG_CN
from qlib.data.dataset import DatasetH
from qlib.utils import init_instance_by_config
from highfreq_handler import HighFreqHandler
from highfreq_ops import IntradayWindowVWAP, DayShift
import pandas as pd

# 设置数据路径
KQ_DATA_DIR = r"C:\Users\ASUS\qlib_data"

# 初始化 Qlib
print("🔧 初始化 Qlib...")
qlib.init(provider_uri=KQ_DATA_DIR, region=REG_CN)
print("✅ Qlib 初始化完成")

# 注册自定义算子
print("🔧 注册自定义算子...")
from qlib.data.ops import Operators
from highfreq_ops import DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut, DayFirst
Operators.register([DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut, DayFirst, IntradayWindowVWAP, DayShift])
print("✅ 自定义算子注册完成")

# 测试配置
test_config = {
    "class": "HighFreqHandler",
    "module_path": "highfreq_handler",
    "kwargs": {
        "instruments": ["SH000300"],
        "start_time": "2025-05-06",
        "end_time": "2025-05-07",
        "drop_raw": False,
    }
}

print("🔧 创建 HighFreqHandler...")
try:
    handler = init_instance_by_config(test_config)
    print("✅ HighFreqHandler 创建成功")
    
    # 直接测试自定义标签计算
    print("\n🔧 直接测试自定义标签计算...")
    custom_labels = handler._load_custom_labels(["SH000300"], "2025-05-06", "2025-05-07")
    print(f"🔍 自定义标签形状: {custom_labels.shape}")
    print(f"🔍 自定义标签列名: {custom_labels.columns.tolist()}")
    print(f"🔍 自定义标签非NaN数量: {custom_labels.notna().sum().sum()}")
    
    if custom_labels.notna().sum().sum() > 0:
        print("✅ 自定义标签计算成功！")
        print(f"🔍 自定义标签统计: {custom_labels.describe()}")
        print(f"🔍 自定义标签示例值:")
        print(custom_labels.dropna().head())
    else:
        print("❌ 自定义标签全为NaN")
        print(f"🔍 自定义标签前5行:")
        print(custom_labels.head())
    
    # 检查handler的_data
    print(f"\n🔍 Handler _data 形状: {handler._data.shape}")
    print(f"🔍 Handler _data 列名: {handler._data.columns.tolist()}")
    
    # 检查标签列
    label_cols = [col for col in handler._data.columns if col[0] == 'label']
    print(f"🔍 标签列: {label_cols}")
    
    if label_cols:
        label_data = handler._data[label_cols[0]]
        print(f"🔍 标签数据形状: {label_data.shape}")
        print(f"🔍 标签非NaN数量: {label_data.notna().sum().sum()}")
        
        if label_data.notna().sum().sum() > 0:
            print("✅ Handler标签计算成功！")
            print(f"🔍 标签统计信息:")
            print(label_data.describe())
        else:
            print("❌ Handler标签全为NaN")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 测试完成！")

