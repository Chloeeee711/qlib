#!/usr/bin/env python3
"""
测试 notebook 中的 Alpha158FZ7 配置
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

# Initialize qlib
provider_uri = '~/.qlib/qlib_data/cn_data'
qlib.init(provider_uri=provider_uri, region=REG_CN)

print("=== 测试 Notebook 中的 Alpha158FZ7 配置 ===")

# 定义与 notebook 中完全相同的 Alpha158FZ7 类
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
                    "Max($close, 240)",    # peak无法识别 改为max
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

try:
    # 创建 Alpha158FZ7 实例
    a158_fz7 = Alpha158FZ7(**data_handler_config)
    
    # 获取特征配置
    feat_conf = a158_fz7.get_feature_config()
    print(f"✓ 特征配置类型: {type(feat_conf)}")
    print(f"✓ 特征配置长度: {len(feat_conf)}")
    
    if isinstance(feat_conf, tuple) and len(feat_conf) == 2:
        fields, names = feat_conf
        print(f"✓ 字段数: {len(fields)}")
        print(f"✓ 名称数: {len(names)}")
        
        # 检查 FactorZoo 因子
        factorzoo_factors = [name for name in names if 'FZ_' in name]
        print(f"✓ FactorZoo 因子数: {len(factorzoo_factors)}")
        
        if factorzoo_factors:
            print(f"\n✅ FactorZoo 因子列表:")
            for i, factor in enumerate(factorzoo_factors, 1):
                print(f"  {i}. {factor}")
        else:
            print("\n❌ 没有找到 FactorZoo 因子！")
            
        # 检查其他因子
        other_factors = [name for name in names if not name.startswith('FZ_')]
        print(f"\n✓ 其他因子数: {len(other_factors)}")
        print(f"✓ 总因子数: {len(names)}")
        
        # 显示一些其他因子作为对比
        print(f"\n其他因子示例 (前10个):")
        for i, factor in enumerate(other_factors[:10], 1):
            print(f"  {i}. {factor}")
            
    else:
        print(f"❌ 特征配置格式不正确: {feat_conf}")
        
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试完成 ===")
