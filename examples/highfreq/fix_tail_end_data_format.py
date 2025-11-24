#!/usr/bin/env python3
"""
修复 tail_end 数据文件格式：在数据前面添加 start_index
根据图片中的解决方案：如果数据文件的第一个值是价格数据（如3925.89），
需要在前面添加 start_index（通常是0）
"""
import numpy as np
from pathlib import Path
import shutil

DATA_DIR = Path("/home/intern0/qlib_data_tail_end/features")
START_INDEX = 0  # 数据从索引0开始（如果数据从日历第一个时间点开始）

def fix_data_file(file_path):
    """修复单个数据文件：在数据前面添加 start_index"""
    # 读取原始数据（只读前几个值用于判断）
    data = np.fromfile(file_path, dtype='<f4')
    
    if len(data) < 2:
        return False
    
    first_value = data[0]
    second_value = data[1]
    
    # 判断第一个值是否为索引
    # 如果第一个值是0或很小的整数（<10），说明格式正确
    if first_value == 0.0 or (0 <= first_value < 10 and first_value == int(first_value)):
        return False  # 格式正确，不需要修复
    
    # 如果第一个值看起来像价格数据（>10且与第二个值接近），说明缺少索引
    if first_value > 10 and abs(first_value - second_value) < abs(first_value * 0.1):
        # 备份原文件
        backup_file = file_path.with_suffix('.bin.bak')
        if not backup_file.exists():
            shutil.copy2(file_path, backup_file)
            print(f"📦 已备份: {backup_file.name}")
        
        # 在数据前面添加 start_index
        fixed_data = np.hstack([START_INDEX, data]).astype('<f4')
        fixed_data.tofile(file_path)
        return True
    
    return False

def main():
    """修复所有数据文件"""
    if not DATA_DIR.exists():
        print(f"❌ 数据目录不存在: {DATA_DIR}")
        return
    
    print(f"🔧 开始检查并修复数据文件格式: {DATA_DIR}")
    print(f"   如果第一个值是价格数据（>10），将在前面添加 start_index={START_INDEX}")
    
    fixed_count = 0
    total_count = 0
    already_correct = 0
    
    for stock_dir in DATA_DIR.iterdir():
        if not stock_dir.is_dir() or stock_dir.is_symlink():
            continue
        
        for bin_file in stock_dir.glob("*.1min.bin"):
            total_count += 1
            if fix_data_file(bin_file):
                fixed_count += 1
                if fixed_count <= 10:
                    print(f"✅ 已修复: {stock_dir.name}/{bin_file.name}")
            else:
                already_correct += 1
    
    print(f"\n📊 修复完成:")
    print(f"   总检查文件数: {total_count}")
    print(f"   已修复文件数: {fixed_count}")
    print(f"   格式正确文件数: {already_correct}")
    
    if fixed_count > 0:
        print(f"\n🎉 数据文件格式修复完成！")
    else:
        print(f"\n✅ 所有文件格式都正确，无需修复")

if __name__ == "__main__":
    main()

