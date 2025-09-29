#!/usr/bin/env python3
"""
调试为什么只有6个因子而不是7个
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

print("=== 调试 FactorZoo 因子数量问题 ===")

# 定义你原始的7个因子
original_factors = [
    "Max($high, 240) / Min($low, 240) - 1",
    "Mad($high, 240) / $close", 
    "Std($close / Ref($close, 4) - 1, 240)",
    "Kurt($close / Ref($close, 4) - 1, 240)",
    "Corr(Ref($high, 1), $volume, 237)",
    "Max($close, 240)",
    "Min($close / Ref($close, 7) - 1, 240)",
]

print(f"原始定义的因子数量: {len(original_factors)}")
for i, factor in enumerate(original_factors, 1):
    print(f"  {i}. {factor}")

# 创建配置
conf = {
    "kbar": {},
    "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
    "rolling": {"windows": [5, 10, 20, 30, 60]},
    "custom_fz": {
        "feature": original_factors,
        "windows": [240],
    },
}

print(f"\n配置中的因子数量: {len(conf['custom_fz']['feature'])}")

# 调用 Alpha158DL.get_feature_config
print("\n调用 Alpha158DL.get_feature_config...")
fields, names = Alpha158DL.get_feature_config(conf)

print(f"\n返回的 fields 数量: {len(fields)}")
print(f"返回的 names 数量: {len(names)}")

# 查找 custom_fz 相关的因子
print("\n查找 custom_fz 相关的因子:")
custom_fz_fields = []
custom_fz_names = []

for i, (field, name) in enumerate(zip(fields, names)):
    # 检查是否是我们的自定义因子
    is_custom = False
    for orig_factor in original_factors:
        if field == orig_factor:
            is_custom = True
            break
    
    if is_custom:
        custom_fz_fields.append(field)
        custom_fz_names.append(name)
        print(f"  {len(custom_fz_fields)}. {name}")
        print(f"      表达式: {field}")

print(f"\n找到的 custom_fz 因子数量: {len(custom_fz_fields)}")

# 检查是否有遗漏的因子
print("\n检查遗漏的因子:")
for i, orig_factor in enumerate(original_factors, 1):
    if orig_factor in custom_fz_fields:
        print(f"  ✅ {i}. {orig_factor}")
    else:
        print(f"  ❌ {i}. {orig_factor} - 未找到!")

# 检查是否有重复的因子
print("\n检查重复的因子:")
seen_fields = set()
duplicates = []
for field in custom_fz_fields:
    if field in seen_fields:
        duplicates.append(field)
    else:
        seen_fields.add(field)

if duplicates:
    print(f"发现重复的因子: {duplicates}")
else:
    print("没有发现重复的因子")

# 检查命名是否有问题
print("\n检查命名:")
for i, (field, name) in enumerate(zip(custom_fz_fields, custom_fz_names), 1):
    print(f"  {i}. 表达式: {field}")
    print(f"      名称: {name}")
    print(f"      长度: {len(name)}")
    print()

# 检查是否有空字符串或None
print("检查空值:")
for i, (field, name) in enumerate(zip(custom_fz_fields, custom_fz_names), 1):
    if not field or field.strip() == "":
        print(f"  ❌ 第{i}个因子表达式为空!")
    if not name or name.strip() == "":
        print(f"  ❌ 第{i}个因子名称为空!")

print("\n=== 调试完成 ===")
