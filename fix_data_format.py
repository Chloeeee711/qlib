#!/usr/bin/env python3
"""修复数据文件格式：在数据前面添加 start_index"""
import numpy as np
from pathlib import Path
import shutil

DATA_DIR = Path("/home/intern0/qlib_data_ipynb/features")
START_INDEX = 0  # 数据从索引0开始

def fix_data_file(file_path):
    """修复单个数据文件：在数据前面添加 start_index"""
    # 读取原始数据
    data = np.fromfile(file_path, dtype='<f')
    
    # 检查第一个值是否是索引（应该是0或很小的整数）
    first_value = data[0]
    
    # 如果第一个值看起来像价格数据（大于100），说明缺少索引
    if first_value > 100:
        # 备份原文件
        backup_file = file_path.with_suffix('.bin.bak2')
        if not backup_file.exists():
            shutil.copy2(file_path, backup_file)
        
        # 在数据前面添加 start_index
        fixed_data = np.hstack([START_INDEX, data]).astype('<f')
        fixed_data.tofile(file_path)
        return True
    return False

# 修复所有数据文件
fixed_count = 0
total_count = 0

for stock_dir in DATA_DIR.iterdir():
    if not stock_dir.is_dir():
        continue
    
    for bin_file in stock_dir.glob("*.1min.bin"):
        total_count += 1
        if fix_data_file(bin_file):
            fixed_count += 1
            if fixed_count <= 5:
                print(f"✅ 已修复: {bin_file.parent.name}/{bin_file.name}")

print(f"\n修复完成: {fixed_count}/{total_count} 个文件")




