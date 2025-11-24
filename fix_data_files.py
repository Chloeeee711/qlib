#!/usr/bin/env python3
"""修复数据文件：跳过第一行的0"""
import numpy as np
from pathlib import Path
import shutil

DATA_DIR = Path("/home/intern0/qlib_data_ipynb/features")

def fix_data_file(file_path):
    """修复单个数据文件"""
    data = np.fromfile(file_path, dtype=np.float32)
    
    # 如果第一个值是0，跳过它
    if len(data) > 0 and data[0] == 0:
        # 备份原文件
        backup_file = file_path.with_suffix('.bin.bak')
        if not backup_file.exists():
            shutil.copy2(file_path, backup_file)
        
        # 写入修复后的数据（跳过第一行）
        data[1:].tofile(file_path)
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
                print(f"✅ 已修复: {bin_file.name}")

print(f"\n修复完成: {fixed_count}/{total_count} 个文件")




