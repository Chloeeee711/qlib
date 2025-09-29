#!/usr/bin/env python3
"""
比较不同处理器配置对 FactorZoo 因子的影响
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add the qlib source code to Python path
qlib_path = Path.cwd().parent  # Go up one level from examples to qlib root
if str(qlib_path) not in sys.path:
    sys.path.insert(0, str(qlib_path))

import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset import DatasetH

# Initialize qlib
provider_uri = '~/.qlib/qlib_data/cn_data'
qlib.init(provider_uri=provider_uri, region=REG_CN)

print("=== 比较不同处理器配置对 FactorZoo 因子的影响 ===")

# 定义 Alpha158FZ7 类 - 使用自定义处理器
class Alpha158FZ7_Custom(Alpha158):
    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
            "rolling": {"windows": [5, 10, 20, 30, 60]},
            "custom_fz": {
                "feature": [
                    "Max($high, 240) / Min($low, 240) - 1",
                    "Mad($high, 240) / $close",
                    "Std($close / Ref($close, 4) - 1, 240)",
                    "Kurt($close / Ref($close, 4) - 1, 240)",
                    "Corr(Ref($high, 1), $volume, 237)",
                    "Max($close, 240)",
                    "Min($close / Ref($close, 7) - 1, 240)",
                ],
                "windows": [240],
            },
        }
        return Alpha158DL.get_feature_config(conf)

    def get_processors(self):
        # 自定义处理器 - 更严格的标准化
        return [
            {"class": "DropnaLabel"},
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "feature"}},
        ]

# 定义 Alpha158FZ7 类 - 使用默认处理器
class Alpha158FZ7_Default(Alpha158):
    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
            "rolling": {"windows": [5, 10, 20, 30, 60]},
            "custom_fz": {
                "feature": [
                    "Max($high, 240) / Min($low, 240) - 1",
                    "Mad($high, 240) / $close",
                    "Std($close / Ref($close, 4) - 1, 240)",
                    "Kurt($close / Ref($close, 4) - 1, 240)",
                    "Corr(Ref($high, 1), $volume, 237)",
                    "Max($close, 240)",
                    "Min($close / Ref($close, 7) - 1, 240)",
                ],
                "windows": [240],
            },
        }
        return Alpha158DL.get_feature_config(conf)

    # 不重写 get_processors，使用默认处理器

# 测试配置
data_handler_config = {
    'start_time': '2008-01-01',
    'end_time': '2020-08-01',
    'fit_start_time': '2008-01-01',
    'fit_end_time': '2014-12-31',
    'instruments': 'csi300',
}

def analyze_factor_distribution(handler_class, name):
    """分析因子分布"""
    print(f"\n=== {name} ===")
    
    try:
        # 创建处理器
        handler = handler_class(**data_handler_config)
        
        # 获取特征配置
        fields, names = handler.get_feature_config()
        
        # 找到 FactorZoo 因子
        fz_indices = [i for i, name in enumerate(names) if 'FZ_' in name]
        fz_fields = [fields[i] for i in fz_indices]
        fz_names = [names[i] for i in fz_indices]
        
        print(f"FactorZoo 因子数量: {len(fz_names)}")
        print(f"FactorZoo 因子: {fz_names}")
        
        # 创建数据集（只用于分析，不实际加载数据）
        dataset = DatasetH(handler=handler, segments={
            "train": ("2008-01-01", "2014-12-31"),
        })
        
        # 获取处理器信息
        processors = handler.get_processors()
        print(f"处理器配置: {len(processors)} 个")
        for i, proc in enumerate(processors):
            print(f"  {i+1}. {proc}")
            
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

# 分析不同配置
print("1. 原始 Alpha158 (无 FactorZoo)")
analyze_factor_distribution(Alpha158, "原始 Alpha158")

print("\n2. Alpha158FZ7 + 默认处理器")
analyze_factor_distribution(Alpha158FZ7_Default, "Alpha158FZ7 + 默认处理器")

print("\n3. Alpha158FZ7 + 自定义处理器")
analyze_factor_distribution(Alpha158FZ7_Custom, "Alpha158FZ7 + 自定义处理器")

print("\n=== 分析完成 ===")

# 分析可能的问题
print("\n=== 可能的问题分析 ===")
print("1. 处理器顺序问题:")
print("   - 自定义处理器: DropnaLabel -> RobustZScoreNorm -> Fillna -> CSRankNorm")
print("   - 默认处理器: ProcessInf -> ZScoreNorm -> Fillna")
print("   - 问题: 自定义处理器可能过度标准化")

print("\n2. 标准化方法问题:")
print("   - RobustZScoreNorm: 使用中位数和MAD，对异常值更鲁棒")
print("   - ZScoreNorm: 使用均值和标准差，对异常值敏感")
print("   - CSRankNorm: 横截面排名标准化，可能改变因子分布")

print("\n3. 建议的优化方案:")
print("   - 使用与原始 Alpha158 相同的处理器")
print("   - 或者调整自定义处理器的顺序和参数")
print("   - 考虑 FactorZoo 因子的特性选择合适的标准化方法")
