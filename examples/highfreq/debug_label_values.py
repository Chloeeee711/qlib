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
import numpy as np

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
    print(f"🔍 自定义标签非NaN数量: {custom_labels.notna().sum().sum()}")
    
    # 检查共同索引的标签值
    common_index = handler._data.index.intersection(custom_labels.index)
    print(f"🔍 共同索引数量: {len(common_index)}")
    
    if len(common_index) > 0:
        common_labels = custom_labels.loc[common_index, "LABEL0"]
        print(f"🔍 共同索引的标签非NaN数量: {common_labels.notna().sum()}")
        print(f"🔍 共同索引的标签示例值:")
        print(common_labels.dropna().head())
        
        # 检查标签值是否有效
        if common_labels.notna().sum() > 0:
            print("✅ 共同索引的标签有有效值")
            print(f"🔍 标签值范围: {common_labels.min()} 到 {common_labels.max()}")
        else:
            print("❌ 共同索引的标签全为NaN")
            print(f"🔍 共同索引: {common_index[:5]}")
            print(f"🔍 自定义标签在共同索引处的值:")
            print(custom_labels.loc[common_index[:5], "LABEL0"])
    
    # 检查handler的标签列
    label_cols = [col for col in handler._data.columns if col[0] == 'label']
    if label_cols:
        label_data = handler._data[label_cols[0]]
        print(f"\n🔍 Handler标签数据形状: {label_data.shape}")
        print(f"🔍 Handler标签非NaN数量: {label_data.notna().sum().sum()}")
        
        if label_data.notna().sum().sum() > 0:
            print("✅ Handler标签有有效值")
            print(f"🔍 Handler标签统计: {label_data.describe()}")
        else:
            print("❌ Handler标签全为NaN")
            print(f"🔍 Handler标签前5行:")
            print(label_data.head())
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 测试完成！")




