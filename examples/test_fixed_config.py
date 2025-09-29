#!/usr/bin/env python3
"""
测试修复后的 Alpha158FZ7 配置
"""

import sys
from pathlib import Path

# Add the qlib source code to Python path
qlib_path = Path.cwd().parent  # Go up one level from examples to qlib root
if str(qlib_path) not in sys.path:
    sys.path.insert(0, str(qlib_path))

import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.loader import Alpha158DL

# Initialize qlib
provider_uri = '~/.qlib/qlib_data/cn_data'
qlib.init(provider_uri=provider_uri, region=REG_CN)

print("=== 测试修复后的 Alpha158FZ7 配置 ===")

# 修复后的配置 - 将 UpStd 改为 Std
conf = {
    "kbar": {},
    "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
    "rolling": {"windows": [5, 10, 20, 30, 60]},
    "custom_fz": {
        "feature": [
            "Max($high, 240) / Min($low, 240) - 1",
            "Mad($high, 240) / $close",
            "Std($close / Ref($close, 4) - 1, 240)",  # 修复：将 UpStd 改为 Std
            "Kurt($close / Ref($close, 4) - 1, 240)",
            "Corr(Ref($high, 1), $volume, 237)",
            "Max($close, 240)",    # peak无法识别 改为max
            "Min($close / Ref($close, 7) - 1, 240)",
        ],
        "windows": [240],
    },
}

try:
    # 测试特征配置生成
    fields, names = Alpha158DL.get_feature_config(conf)
    
    print(f"✓ 总特征数: {len(fields)}")
    print(f"✓ 总名称数: {len(names)}")
    
    # 检查 FactorZoo 因子
    factorzoo_factors = [name for name in names if 'FZ_' in name]
    print(f"✓ FactorZoo 因子数: {len(factorzoo_factors)}")
    
    # 检查其他因子
    other_factors = [name for name in names if not name.startswith('FZ_')]
    print(f"✓ 其他因子数: {len(other_factors)}")
    
    print(f"\n预期: 158 (原始) + 7 (FactorZoo) = 165 总计")
    print(f"实际: {len(other_factors)} (原始) + {len(factorzoo_factors)} (FactorZoo) = {len(names)} 总计")
    
    # 显示 FactorZoo 因子
    if factorzoo_factors:
        print(f"\n✅ FactorZoo 因子列表:")
        for i, factor in enumerate(factorzoo_factors, 1):
            print(f"  {i}. {factor}")
    else:
        print("\n❌ 没有找到 FactorZoo 因子！")
        
    # 显示一些其他因子作为对比
    print(f"\n其他因子示例 (前10个):")
    for i, factor in enumerate(other_factors[:10], 1):
        print(f"  {i}. {factor}")
        
    # 显示 FactorZoo 因子对应的表达式
    print(f"\nFactorZoo 因子对应的表达式:")
    fz_keywords = ['Max($high', 'Mad($high', 'Std', 'Kurt', 'Corr', 'Max($close', 'Min($close']
    fz_expressions = [f for f in fields if any(fz in f for fz in fz_keywords)]
    for i, (name, field) in enumerate(zip(factorzoo_factors, fz_expressions), 1):
        print(f"  {i}. {name}: {field}")
        
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试完成 ===")
