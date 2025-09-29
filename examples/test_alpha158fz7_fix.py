#!/usr/bin/env python3
"""
测试 Alpha158FZ7 修复后的效果
"""

import sys
from pathlib import Path

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

# 定义 Alpha158FZ7 类
class Alpha158FZ7(Alpha158):
    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
            "rolling": {"windows": [5, 10, 20, 30, 60]},
            "custom_fz": {
                "feature": [
                    "Max($high, 240) / Min($low, 240) - 1",
                    "Mad($high, 240) / $close",
                    "UpStd($close / Ref($close, 4) - 1, 240)",
                    "Kurt($close / Ref($close, 4) - 1, 240)",
                    "Corr(Ref($high, 1), $volume, 237)",
                    "Peak($close, 240)",
                    "Min($close / Ref($close, 7) - 1, 240)",
                ],
                "windows": [240],
            },
        }
        return Alpha158DL.get_feature_config(conf)

    def get_processors(self):
        return [
            {"class": "DropnaLabel"},
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "feature"}},
        ]

# 测试配置
data_handler_config = {
    'start_time': '2008-01-01',
    'end_time': '2020-08-01',
    'fit_start_time': '2008-01-01',
    'fit_end_time': '2014-12-31',
    'instruments': 'csi300',
}

print("=== 测试 Alpha158FZ7 修复效果 ===")

# 创建数据集
try:
    a158_fz7 = Alpha158FZ7(**data_handler_config)
    dataset = DatasetH(handler=a158_fz7, segments={
        "train": ("2008-01-01", "2014-12-31"),
        "valid": ("2015-01-01", "2016-12-31"),
        "test":  ("2017-01-01", "2020-08-01"),
    })
    
    # 获取特征列
    cols = dataset.get_cols()
    print(f"✓ 总特征数: {len(cols)}")
    
    # 检查 FactorZoo 因子
    factorzoo_factors = [col for col in cols if 'FZ_' in col]
    print(f"✓ FactorZoo 因子数: {len(factorzoo_factors)}")
    
    # 检查其他因子
    other_factors = [col for col in cols if not col.startswith('FZ_')]
    print(f"✓ 其他因子数: {len(other_factors)}")
    
    print(f"\n预期: 158 (原始) + 7 (FactorZoo) = 165 总计")
    print(f"实际: {len(other_factors)} (原始) + {len(factorzoo_factors)} (FactorZoo) = {len(cols)} 总计")
    
    # 显示 FactorZoo 因子
    if factorzoo_factors:
        print(f"\nFactorZoo 因子列表:")
        for i, factor in enumerate(factorzoo_factors, 1):
            print(f"  {i}. {factor}")
    else:
        print("\n❌ 没有找到 FactorZoo 因子！")
        
    # 显示一些其他因子作为对比
    print(f"\n其他因子示例 (前10个):")
    for i, factor in enumerate(other_factors[:10], 1):
        print(f"  {i}. {factor}")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试完成 ===")
