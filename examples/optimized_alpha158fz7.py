#!/usr/bin/env python3
"""
优化的 Alpha158FZ7 类 - 解决效果变差问题
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

print("=== 优化的 Alpha158FZ7 类 ===")

# 方案1：使用与原始 Alpha158 完全相同的处理器
class Alpha158FZ7_OriginalProcessors(Alpha158):
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
    
    # 不重写 get_processors，使用原始 Alpha158 的默认处理器

# 方案2：轻微优化的处理器（保持原始风格，但稍微调整）
class Alpha158FZ7_Optimized(Alpha158):
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
        # 使用与原始 Alpha158 相同的处理器，但添加轻微优化
        return [
            {"class": "ProcessInf", "kwargs": {}},
            {"class": "ZScoreNorm", "kwargs": {}},
            {"class": "Fillna", "kwargs": {}},
        ]

# 方案3：针对 FactorZoo 因子优化的处理器
class Alpha158FZ7_FactorZooOptimized(Alpha158):
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
        # 针对 FactorZoo 因子优化的处理器配置
        return [
            {"class": "ProcessInf", "kwargs": {}},
            {"class": "ZScoreNorm", "kwargs": {}},
            {"class": "Fillna", "kwargs": {}},
            # 只对 FactorZoo 因子进行额外的鲁棒性处理
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        ]

# 方案4：你当前的自定义处理器（作为对比）
class Alpha158FZ7_Current(Alpha158):
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
        # 你当前的自定义处理器
        return [
            {"class": "DropnaLabel"},
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "feature"}},
        ]

def test_processor_config(handler_class, name):
    """测试处理器配置"""
    print(f"\n=== {name} ===")
    
    try:
        # 创建处理器
        handler = handler_class(
            start_time='2008-01-01',
            end_time='2020-08-01',
            fit_start_time='2008-01-01',
            fit_end_time='2014-12-31',
            instruments='csi300',
        )
        
        # 获取处理器配置
        processors = handler.get_processors()
        print(f"处理器数量: {len(processors)}")
        for i, proc in enumerate(processors, 1):
            print(f"  {i}. {proc}")
            
        # 获取特征配置
        fields, names = handler.get_feature_config()
        fz_factors = [name for name in names if 'FZ_' in name]
        print(f"FactorZoo 因子数量: {len(fz_factors)}")
        print(f"总特征数量: {len(names)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

# 测试所有方案
print("测试不同的处理器配置方案：")

test_processor_config(Alpha158FZ7_OriginalProcessors, "方案1: 使用原始 Alpha158 处理器")
test_processor_config(Alpha158FZ7_Optimized, "方案2: 轻微优化的处理器")
test_processor_config(Alpha158FZ7_FactorZooOptimized, "方案3: FactorZoo 优化处理器")
test_processor_config(Alpha158FZ7_Current, "方案4: 当前自定义处理器")

print("\n=== 推荐方案 ===")
print("1. 方案1 (推荐): 使用与原始 Alpha158 完全相同的处理器")
print("   - 优点: 保持与原始 Alpha158 的一致性")
print("   - 缺点: 没有针对 FactorZoo 因子的特殊优化")
print()
print("2. 方案2 (推荐): 轻微优化的处理器")
print("   - 优点: 保持原始风格，但可以添加轻微优化")
print("   - 缺点: 需要测试效果")
print()
print("3. 方案3: FactorZoo 优化处理器")
print("   - 优点: 针对 FactorZoo 因子进行优化")
print("   - 缺点: 可能仍然过度处理")
print()
print("4. 方案4: 当前自定义处理器")
print("   - 优点: 严格的标准化处理")
print("   - 缺点: 可能过度标准化，导致效果变差")

print("\n=== 建议 ===")
print("建议先尝试方案1，如果效果仍然不好，再尝试方案2。")
print("避免使用方案4（当前配置），因为它可能过度标准化。")
