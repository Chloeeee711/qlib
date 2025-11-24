# ============================================
# 按年份分段下载数据
# 可以手动控制下载进度，中断后可以从某个年份继续
# ============================================

import sys
import os
from pathlib import Path
import subprocess

# 配置
PROJECT_ROOT = Path(r"E:\qlib")
SCRIPT = PROJECT_ROOT / "scripts" / "data_collector" / "KQ" / "KQdownloader.py"
SOURCE_DIR = r"E:\kq_raw_data_20-25_500"
TARGET_DIR = r"E:\qlib_data_20-25_500"
STOCK_POOL_CSV = r"E:\qlib\examples\highfreq\sorted_high_preclose_ratio_2025.csv"
USERNAME = "xclight"
PASSWORD = "xclight666"
BENCHMARK = "SSE.000300"
BENCHMARK_DIR = r"E:\qlib_data_20-25_500\benchmark"

# Windows multiprocessing 修复：减少并发数
# 如果遇到 BrokenProcessPool 错误，可以尝试将 MAX_WORKERS 设置为 1 或 2
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "2"))  # 默认 2，减少 Windows 进程池问题

# 年份列表（2020-2025）
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

print("="*70)
print("按年份分段下载数据")
print("="*70)
print(f"股票池: {STOCK_POOL_CSV}")
print(f"年份范围: {YEARS[0]} ~ {YEARS[-1]}")
print(f"源目录: {SOURCE_DIR}")
print(f"目标目录: {TARGET_DIR}")
print(f"并发数: {MAX_WORKERS} (Windows 建议 1-2，避免 BrokenProcessPool 错误)")
print("💡 提示: 如果遇到进程池错误，可以设置环境变量 MAX_WORKERS=1")
print("="*70 + "\n")

# 询问用户从哪一年开始
print("可用年份:")
for i, year in enumerate(YEARS, 1):
    print(f"  {i}. {year}年")

print("\n选项:")
print("  - 输入年份数字（如 2020）: 下载该年份")
print("  - 输入 'all': 自动下载所有年份")
print("  - 输入 'continue': 从第一个未完成的年份开始")
print("  - 输入 'q': 退出")

choice = input("\n请选择: ").strip().lower()

if choice == 'q':
    print("退出")
    exit(0)

# 确定要下载的年份
years_to_download = []

if choice == 'all':
    years_to_download = YEARS
    print(f"\n将下载所有年份: {years_to_download}")
elif choice == 'continue':
    # 检查已下载的数据，找出第一个未完成的年份
    from datetime import datetime
    import pandas as pd
    
    print("\n检查已下载数据...")
    source_path = Path(SOURCE_DIR)
    if source_path.exists():
        # 检查一个示例文件
        sample_files = list(source_path.glob("*.csv"))
        if sample_files:
            try:
                sample_df = pd.read_csv(sample_files[0], usecols=['datetime'], parse_dates=['datetime'])
                if not sample_df.empty:
                    latest_year = sample_df['datetime'].max().year
                    print(f"已下载数据最新年份: {latest_year}")
                    # 从下一年开始
                    start_idx = YEARS.index(latest_year) + 1 if latest_year in YEARS else 0
                    years_to_download = YEARS[start_idx:]
                    print(f"将从 {years_to_download[0] if years_to_download else '无'} 年开始下载")
                else:
                    years_to_download = YEARS
            except:
                years_to_download = YEARS
        else:
            years_to_download = YEARS
    else:
        years_to_download = YEARS
elif choice.isdigit():
    year = int(choice)
    if year in YEARS:
        years_to_download = [year]
        print(f"\n将下载: {year}年")
    else:
        print(f"错误: {year} 不在可用年份列表中")
        exit(1)
else:
    print("无效选择")
    exit(1)

if not years_to_download:
    print("所有年份都已下载完成！")
    exit(0)

# 逐个年份下载
MAX_RETRIES = 3  # 最大重试次数
RETRY_DELAY = 5  # 重试延迟（秒）

for year in years_to_download:
    print(f"\n{'='*70}")
    print(f"开始下载 {year} 年数据")
    print(f"{'='*70}\n")
    
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    # 构建命令（使用fire库，参数格式）
    cmd = [
        sys.executable, str(SCRIPT), 'run',
        '--source_dir', SOURCE_DIR,
        '--target_dir', TARGET_DIR,
        '--csv_stock_pool', STOCK_POOL_CSV,
        '--start', start_date,
        '--end', end_date,
        '--interval', '1min',
        '--username', USERNAME,
        '--password', PASSWORD,
        '--benchmark', BENCHMARK,
        '--benchmark_dir', BENCHMARK_DIR,
        '--year_segment', str(year),
        '--append_mode', 'True',  # 追加模式
        '--max_workers', str(MAX_WORKERS)  # 传递并发数
    ]
    
    print(f"执行命令:")
    print(f"  {' '.join(cmd)}\n")
    
    # 执行下载（带重试机制）
    success = False
    retry_count = 0
    
    while retry_count < MAX_RETRIES and not success:
        try:
            if retry_count > 0:
                print(f"🔄 第 {retry_count} 次重试...")
                import time
                time.sleep(RETRY_DELAY)
            
            # 设置环境变量，减少 Windows multiprocessing 问题
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            # 限制并发数，减少进程间通信压力
            env['JOBLIB_TEMP_FOLDER'] = str(Path(TARGET_DIR).parent / 'temp_joblib')
            env['MAX_WORKERS'] = str(MAX_WORKERS)  # 传递并发数给子进程
            
            result = subprocess.run(
                cmd, 
                check=False,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            if result.returncode == 0:
                print(f"\n✅ {year} 年数据下载完成")
                success = True
            else:
                print(f"\n❌ {year} 年数据下载失败 (退出码: {result.returncode})")
                
                # 检查是否是 multiprocessing 错误
                if "BrokenProcessPool" in result.stdout or "AssertionError" in result.stdout:
                    print("⚠️  检测到进程池错误（Windows multiprocessing 常见问题）")
                    print("💡 建议：")
                    print("   1. 减少并发数（修改 dump_bin.py 中的 works 参数）")
                    print("   2. 分批下载（按季度或月份）")
                    print("   3. 使用 ThreadPoolExecutor 替代 ProcessPoolExecutor")
                
                # 显示部分错误输出
                error_lines = result.stdout.split('\n')
                error_snippet = [line for line in error_lines if any(
                    keyword in line for keyword in ['Error', 'Exception', 'Traceback', 'BrokenProcessPool']
                )]
                if error_snippet:
                    print("\n错误摘要:")
                    for line in error_snippet[:10]:  # 只显示前10行
                        print(f"  {line}")
                
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    print(f"\n将在 {RETRY_DELAY} 秒后重试...")
                else:
                    print(f"\n❌ 已达到最大重试次数 ({MAX_RETRIES})，放弃下载 {year} 年数据")
                    print("是否继续下载下一年的数据？(y/n): ", end='')
                    continue_choice = input().strip().lower()
                    if continue_choice != 'y':
                        print("已停止")
                        break
                    
        except KeyboardInterrupt:
            print(f"\n\n⚠️  用户中断了 {year} 年的下载")
            print("下次运行可以选择 'continue' 从当前年份继续")
            break
        except Exception as e:
            print(f"\n❌ 下载 {year} 年数据时出错: {e}")
            retry_count += 1
            if retry_count < MAX_RETRIES:
                print(f"\n将在 {RETRY_DELAY} 秒后重试...")
            else:
                print("是否继续下载下一年的数据？(y/n): ", end='')
                continue_choice = input().strip().lower()
                if continue_choice != 'y':
                    print("已停止")
                    break

print(f"\n{'='*70}")
print("分段下载完成")
print(f"{'='*70}")

