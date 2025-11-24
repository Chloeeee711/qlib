#!/usr/bin/env python3
"""
修复 tail_end 数据集的文件夹大小写问题
根据图片中的解决方案：Qlib使用instrument.lower()查找文件夹，但实际文件夹是大写的
需要创建符号链接，将小写文件夹名映射到大写文件夹名
"""
import os
from pathlib import Path

FEATURES_DIR = Path("/home/intern0/qlib_data_tail_end/features")

def fix_folder_case_mismatch():
    """创建小写文件夹名的符号链接指向大写文件夹"""
    if not FEATURES_DIR.exists():
        print(f"❌ 特征目录不存在: {FEATURES_DIR}")
        return
    
    print(f"🔧 开始修复文件夹大小写问题: {FEATURES_DIR}")
    
    created_count = 0
    skipped_count = 0
    error_count = 0
    
    # 遍历所有大写文件夹
    for folder in FEATURES_DIR.iterdir():
        if not folder.is_dir():
            continue
        
        original_name = folder.name
        lowercase_name = original_name.lower()
        
        # 如果已经是大写，创建小写符号链接
        if original_name != lowercase_name:
            lowercase_path = FEATURES_DIR / lowercase_name
            
            # 如果小写路径已存在（可能是符号链接或真实文件夹），跳过
            if lowercase_path.exists():
                if lowercase_path.is_symlink():
                    # 检查符号链接是否指向正确的大写文件夹
                    try:
                        target = lowercase_path.resolve()
                        if target == folder.resolve():
                            skipped_count += 1
                            continue
                        else:
                            # 符号链接指向错误，删除并重新创建
                            lowercase_path.unlink()
                            print(f"🔄 更新符号链接: {lowercase_name} -> {original_name}")
                    except Exception as e:
                        print(f"⚠️ 检查符号链接失败 {lowercase_name}: {e}")
                        error_count += 1
                        continue
                else:
                    # 已存在真实文件夹，跳过
                    skipped_count += 1
                    continue
            
            # 创建符号链接
            try:
                lowercase_path.symlink_to(original_name)
                created_count += 1
                if created_count <= 10:  # 只显示前10个
                    print(f"✅ 创建符号链接: {lowercase_name} -> {original_name}")
            except Exception as e:
                print(f"❌ 创建符号链接失败 {lowercase_name}: {e}")
                error_count += 1
    
    print(f"\n📊 修复完成:")
    print(f"   创建符号链接: {created_count} 个")
    print(f"   跳过（已存在）: {skipped_count} 个")
    print(f"   失败: {error_count} 个")
    
    if created_count > 0:
        print(f"🎉 文件夹大小写问题已修复！")

if __name__ == "__main__":
    fix_folder_case_mismatch()

