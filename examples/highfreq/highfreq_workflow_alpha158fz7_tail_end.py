# ============================================
# 实时交易信号生成系统
# 功能：首次初始化历史数据 + 增量更新 + 14:40预测并保存CSV
# ============================================

from pathlib import Path
import os
import sys
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import struct
import schedule

# 确保 qlib 源码目录在 Python 路径中（必须在导入 qlib 之前）
QLIB_ROOT = "/home/intern0/qlib"
if QLIB_ROOT not in sys.path:
    sys.path.insert(0, QLIB_ROOT)

# ====== 内存优化：设置环境变量（必须在导入qlib之前） ======
# 将缓存目录设置到E盘（如果有），避免占用C盘空间
cache_dir = None
for drive in ['E:', 'D:', 'C:']:
    if os.path.exists(drive):
        cache_dir = os.path.join(drive, 'qlib_cache')
        try:
            os.makedirs(cache_dir, exist_ok=True)
            # 测试写入权限
            test_file = os.path.join(cache_dir, 'test.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            break
        except:
            continue

if cache_dir is None:
    cache_dir = os.path.join(os.path.expanduser('~'), '.qlib', 'cache')
    os.makedirs(cache_dir, exist_ok=True)

# 设置环境变量，限制joblib的并行数和缓存位置
os.environ['JOBLIB_TEMP_FOLDER'] = cache_dir
# 使用单进程模式（最大程度降低内存峰值）
os.environ['OMP_NUM_THREADS'] = '1'  # 限制OpenMP线程数
os.environ['MKL_NUM_THREADS'] = '1'  # 限制MKL线程数
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # 限制NumExpr线程数
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # 限制VecLib线程数
# 进一步减少缓存大小
os.environ['JOBLIB_MAX_NBYTES'] = '1M'  # 最小缓存

# 设置临时目录到E盘（如果可用）
for drive in ['E:', 'D:', 'C:']:
    if os.path.exists(drive):
        tmp_dir = os.path.join(drive, 'tmp')
        try:
            os.makedirs(tmp_dir, exist_ok=True)
            os.environ['TMP'] = tmp_dir
            os.environ['TEMP'] = tmp_dir
            break
        except:
            continue

print(f"🔧 内存优化环境变量已设置:")
print(f"   - 缓存目录: {cache_dir}")
print(f"   - 临时目录: {os.environ.get('TMP', '默认')}")
print(f"   - 单进程模式: OMP=1, MKL=1, NUMEXPR=1")
print(f"   - Joblib缓存: {os.environ.get('JOBLIB_MAX_NBYTES', '默认')}")

# =============== 第一部分：天勤数据下载模块 ===============
# 检查是否跳过历史数据下载（增量更新不受影响）
SKIP_HISTORICAL_DOWNLOAD = True  # 设置为True跳过历史数据下载，False重新下载

def download_historical_data():
    """下载历史数据：2025-01-01到今天"""
    nb_cwd = Path.cwd()
    if (nb_cwd / 'scripts').exists() and (nb_cwd / 'qlib').exists():
        PROJECT_ROOT = nb_cwd
    else:
        PROJECT_ROOT = nb_cwd
        if PROJECT_ROOT.name.lower() == 'highfreq' and PROJECT_ROOT.parent.name.lower() == 'examples':
            PROJECT_ROOT = PROJECT_ROOT.parents[1]

    SCRIPT = PROJECT_ROOT / "scripts" / "data_collector" / "KQ" / "KQdownloader.py"
    RAW_DIR = PROJECT_ROOT.parent / "kq_raw_data_tail_end"
    QLIB_DIR = PROJECT_ROOT.parent / "qlib_data_tail_end"
    POOL_CSV = Path("/home/intern0/qlib/examples/highfreq/tail_end_candidates.csv")

    START = "2025-01-01"
    END = datetime.now().strftime("%Y-%m-%d")
    INTERVAL = "1min"
    USERNAME = "xclight"
    PASSWORD = "xclight666"
    LIMIT_NUMS = None

    BENCHMARK_CODE = "SSE.000300"
    BENCHMARK_DIR = QLIB_DIR / "benchmark"
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_FILE = BENCHMARK_DIR / f"{BENCHMARK_CODE}.csv"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    QLIB_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("📦 下载天勤历史数据")
    print("="*70)
    print(f"📅 日期范围: {START} ~ {END}")
    print(f"📊 数据频率: {INTERVAL}")
    print(f"📂 输出目录: {QLIB_DIR}")
    print("="*70 + "\n")

    cmd = [
        sys.executable, str(SCRIPT), 'run',
        '--source_dir', str(RAW_DIR),
        '--target_dir', str(QLIB_DIR),
        '--csv_stock_pool', str(POOL_CSV),
        '--start', START,
        '--end', END,
        '--interval', INTERVAL,
        '--username', USERNAME,
        '--password', PASSWORD,
        '--benchmark', BENCHMARK_CODE,
        '--benchmark_dir', str(BENCHMARK_DIR)
    ]
    if LIMIT_NUMS:
        cmd += ['--limit_nums', str(LIMIT_NUMS)]

    print("🚀 开始下载...\n")
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        for line in p.stdout:
            print(line, end='')
        rc = p.wait()
        if rc != 0:
            print(f"⚠️ 下载完成但有错误，返回码: {rc}")

    print(f"\n✅ 数据下载完成: {QLIB_DIR}")
    print("="*70 + "\n")

# 调用下载函数（首次初始化）
if SKIP_HISTORICAL_DOWNLOAD:
    print("⏭️ 跳过历史数据下载，使用现有数据...")
    print("💡 增量更新功能仍然可用")
else:
    print("🔧 初始化：下载历史数据...")
    download_historical_data()

# 继续原有代码（删除重复导入）

# ====== 天勤数据目录 - 历史数据路径（用于训练） ======
KQ_DATA_DIR = Path("/home/intern0/qlib_data_tail_end")  # 历史数据根目录
RAW_DIR = Path("/home/intern0/kq_raw_data_tail_end")  # 历史原始数据
SKIP_FEATURE_REWRITE = True  # 重要：避免在本脚本内重写 features，改用独立转换脚本

# ====== 工具函数 ======
def market_prefix_convert(code: str):
    """sse.xxx / szse.xxx / SSE_000300 → SHxxx / SZxxx (文件夹名用大写，支持点号和下划线)"""
    code = code.upper()
    # 支持点号分隔：SSE.000300
    if code.startswith("SSE."):
        return "SH" + code.split(".")[1]
    elif code.startswith("SZSE."):
        return "SZ" + code.split(".")[1]
    # 支持下划线分隔：SSE_000300
    elif code.startswith("SSE_"):
        return "SH" + code.split("_")[1]
    elif code.startswith("SZSE_"):
        return "SZ" + code.split("_")[1]
    else:
        return code.upper()

def market_prefix_convert_upper(code: str):
    """sse.xxx / szse.xxx / SSE_000300 → SHxxx / SZxxx (all.txt用大写，支持点号和下划线)"""
    code = code.upper()
    # 支持点号分隔：SSE.000300
    if code.startswith("SSE."):
        return "SH" + code.split(".")[1]
    elif code.startswith("SZSE."):
        return "SZ" + code.split(".")[1]
    # 支持下划线分隔：SSE_000300
    elif code.startswith("SSE_"):
        return "SH" + code.split("_")[1]
    elif code.startswith("SZSE_"):
        return "SZ" + code.split("_")[1]
    else:
        return code.upper()

# 测试转换函数
print("🧪 测试转换函数:")
test_codes = ["sse.000300", "szse.000001", "SSE.600519", "SZSE.000858", "SSE_000300", "SZSE_000001"]
for code in test_codes:
    converted_folder = market_prefix_convert(code)
    converted_file = market_prefix_convert_upper(code)
    print(f"   {code} → 文件夹: {converted_folder}, all.txt: {converted_file}")

def write_bin(series: pd.Series, path: Path):
    """写入 float32 二进制"""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = series.astype(np.float32).to_numpy()
    with open(path, "wb") as f:
        f.write(struct.pack(f"{len(arr)}f", *arr))

def read_bin(fp):
    return np.fromfile(fp, dtype=np.float32)

# ====== 1. 日历文件保持不变 ======
print(f"✅ 日历文件已存在: {KQ_DATA_DIR / 'calendars' / '1min.txt'}")

# ====== 2. instruments/all.txt 转换 ======
inst_src = KQ_DATA_DIR / "instruments" / "all.txt"

# 使用更灵活的读取方式处理格式问题
print(f"🔍 检查股票池文件格式: {inst_src}")

# 先读取原始内容查看格式
with open(inst_src, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
print(f"   文件总行数: {len(lines)}")
print(f"   前3行原始内容:")
for i in range(min(3, len(lines))):
    print(f"     {repr(lines[i].strip())}")

# 手动解析每行，处理引号和格式问题
parsed_data = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    
    # 分割字段，处理引号问题
    parts = line.split('\t')
    if len(parts) >= 3:
        instrument = parts[0].strip()
        start_time = parts[1].strip()
        end_time = parts[2].strip().strip('"').strip()  # 去除引号和空格
        
        # 清理end_time中的多余内容
        if '"' in end_time:
            end_time = end_time.split('"')[0].strip()
        
        parsed_data.append([instrument, start_time, end_time])
    else:
        print(f"⚠️ 跳过格式异常的行: {line}")

# 创建DataFrame
inst_df = pd.DataFrame(parsed_data, columns=["instrument", "start", "end"])

print(f"📝 解析完成，数据形状: {inst_df.shape}")
print(f"   前5行解析结果:")
for i in range(min(5, len(inst_df))):
    print(f"     {inst_df.iloc[i].tolist()}")

# 转换股票代码格式（all.txt用大写）
inst_df["instrument"] = inst_df["instrument"].apply(market_prefix_convert_upper)

# 保存转换后的文件
inst_df.to_csv(inst_src, sep="\t", header=False, index=False)
print(f"✅ 股票池格式已转换: {inst_src}")
print(f"   转换后形状: {inst_df.shape}")
print(f"   前5支股票: {inst_df['instrument'].head().tolist()}")

# ====== 3. features 特征字段补齐 ======
#修复后 原始写入bin第一行数据为0 现在改成直接从csv读取写入bin
feature_src = KQ_DATA_DIR / "features"
csv_dir = RAW_DIR  # CSV 文件目录

# 删除benchmark文件夹（如果存在）
benchmark_dir = KQ_DATA_DIR / "benchmark"
if benchmark_dir.exists():
    import shutil
    shutil.rmtree(benchmark_dir)
    print(f"🗑️ 已删除benchmark文件夹: {benchmark_dir}")

print(f"📁 开始处理特征目录: {feature_src}")
if SKIP_FEATURE_REWRITE:
    print("⏭️ 跳过 features 重建：已通过 convert_kq_to_qlib.py 正确生成 .bin（含 date_index）。")
    print("💡 如需重新构建，请将 SKIP_FEATURE_REWRITE 设为 False 再运行本段。")
else:
    pass

# 先清空features目录（避免重复文件夹）
"""
🧹 清空features目录（将重新生成所有文件夹）...
if feature_src.exists():
    existing_folders = [d for d in feature_src.iterdir() if d.is_dir()]
    print(f"   发现 {len(existing_folders)} 个现有文件夹，开始删除...")
    
    import shutil
    deleted_count = 0
    failed_count = 0
    
    for folder in existing_folders:
        try:
            shutil.rmtree(folder)
            deleted_count += 1
            if deleted_count <= 5:  # 只显示前5个
                print(f"   🗑️ 删除: {folder.name}")
        except PermissionError as e:
            failed_count += 1
            print(f"   ❌ 删除失败 {folder.name}: 权限错误，可能被占用")
        except Exception as e:
            failed_count += 1
            print(f"   ❌ 删除失败 {folder.name}: {e}")
    
    if deleted_count > 5:
        print(f"   ... 还有 {deleted_count - 5} 个文件夹已删除")
    
    print(f"   ✅ 清理完成: 成功删除 {deleted_count} 个, 失败 {failed_count} 个")
    
    if failed_count > 0:
        print(f"   ⚠️ 警告: 有 {failed_count} 个文件夹删除失败，可能影响数据转换")
        print(f"   建议: 关闭可能占用这些文件夹的程序后重新运行")
else:
    feature_src.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ features目录已创建")
"""

# 处理每个股票的特征文件：已改为外部独立脚本生成，这里不再重写
print("📁 特征文件采用外部转换产物（convert_kq_to_qlib.py），本脚本不进行重建。")

# ====== 4. 验证数据转换结果 ======
print("\n🔍 开始验证数据转换结果...")

# 验证日历文件
calendar_file = KQ_DATA_DIR / "calendars" / "1min.txt"
if calendar_file.exists():
    with open(calendar_file, 'r') as f:
        calendar_lines = f.readlines()
    print(f"✅ 日历文件验证: {len(calendar_lines)} 个交易日")
    print(f"   最早交易日: {calendar_lines[0].strip()}")
    print(f"   最晚交易日: {calendar_lines[-1].strip()}")
else:
    print("❌ 日历文件不存在")

# 验证股票池文件
instruments_file = KQ_DATA_DIR / "instruments" / "all.txt"
if instruments_file.exists():
    inst_df_check = pd.read_csv(instruments_file, header=None, sep='\t')
    print(f"✅ 股票池文件验证: {len(inst_df_check)} 支股票")
    print(f"   前5支股票: {inst_df_check.iloc[:5, 0].tolist()}")
else:
    print("❌ 股票池文件不存在")

# 验证特征文件
feature_dir = KQ_DATA_DIR / "features"
if feature_dir.exists():
    stock_dirs = [d for d in feature_dir.iterdir() if d.is_dir()]
    print(f"✅ 特征目录验证: {len(stock_dirs)} 个股票目录")
    
    # 检查第一个股票的特征文件
    if stock_dirs:
        first_stock = stock_dirs[0]
        feature_files = list(first_stock.glob("*.1min.bin"))
        print(f"   示例股票 {first_stock.name} 的特征文件:")
        for f in feature_files:
            file_size = f.stat().st_size
            data_points = file_size // 4  # float32 = 4 bytes
            print(f"     {f.name}: {data_points} 个数据点")
            
        # 验证数据内容
        try:
            close_data = read_bin(first_stock / "close.1min.bin")
            print(f"    {first_stock.name} close 数据范围: {close_data.min():.2f} ~ {close_data.max():.2f}")
            print(f"    数据点数: {len(close_data)}")
        except Exception as e:
            print(f"   ❌ 读取数据失败: {e}")
else:
    print("❌ 特征目录不存在")

# 验证benchmark文件夹（已删除）
benchmark_dir = KQ_DATA_DIR / "benchmark"
if benchmark_dir.exists():
    print("⚠️ Benchmark文件夹仍然存在")
else:
    print("✅ Benchmark文件夹已删除（符合Qlib标准）")

# 最终验证总结
print(f"\n📊 数据转换验证总结:")
print(f"   数据目录: {KQ_DATA_DIR}")
print(f"   日历文件: {'✅' if calendar_file.exists() else '❌'}")
print(f"   股票池文件: {'✅' if instruments_file.exists() else '❌'}")
print(f"   特征目录: {'✅' if feature_dir.exists() else '❌'}")
print(f"   Benchmark文件夹: {'❌' if benchmark_dir.exists() else '✅'} (已删除)")

# 检查文件夹命名格式
print(f"\n🔍 检查文件夹命名格式:")
if feature_dir.exists():
    stock_dirs = [d for d in feature_dir.iterdir() if d.is_dir()]
    if stock_dirs:
        print(f"   前5个文件夹名称:")
        for i, d in enumerate(stock_dirs[:5]):
            print(f"     {i+1}. {d.name}")
        
        # 检查是否符合Qlib标准格式（文件夹用大写）
        non_standard = [d.name for d in stock_dirs if not (d.name.startswith('SH') or d.name.startswith('SZ'))]
        if non_standard:
            print(f"   ⚠️ 发现非标准格式文件夹: {non_standard[:5]}")
        else:
            print(f"   ✅ 所有文件夹都符合Qlib标准格式 (SH/SZ开头，大写)")
    else:
        print("   ❌ 没有找到股票文件夹")

# 检查是否有任何文件被修改
import time
current_time = time.time()
recent_files = []

for root, dirs, files in os.walk(KQ_DATA_DIR):
    for file in files:
        file_path = Path(root) / file
        if file_path.stat().st_mtime > current_time - 300:  # 5分钟内的文件
            recent_files.append(str(file_path))

if recent_files:
    print(f"\n🔄 最近5分钟内修改的文件 ({len(recent_files)} 个):")
    for f in recent_files[:10]:  # 只显示前10个
        print(f"   {f}")
    if len(recent_files) > 10:
        print(f"   ... 还有 {len(recent_files) - 10} 个文件")
else:
    print(f"\n⚠️ 警告: 最近5分钟内没有文件被修改，可能数据转换未成功执行")

print("\n🎯 验证完成！如果所有项目都显示 ✅，说明数据转换成功。")


from pathlib import Path

# 注意：这个重命名代码块应该只在数据预处理时运行一次
# 如果文件夹已经是大写格式，应该跳过此步骤
FEATURES_DIR = Path("/home/intern0/qlib_data_tail_end/features")  # 使用Tail-End数据路径

def _normalize_feature_dir_name(name: str) -> str:
    """将 features 目录名规范为 Qlib 期望格式（SH/SZ 开头，无点号）。"""
    if not name:
        return name
    norm = name.strip().upper()
    if norm.startswith("SSE."):
        return f"SH{norm.split('.', 1)[1]}"
    if norm.startswith("SZSE."):
        return f"SZ{norm.split('.', 1)[1]}"
    if norm.startswith("SH.") or norm.startswith("SZ."):
        return norm.replace(".", "")
    return norm

if FEATURES_DIR.exists():
    print(f"\n🔄 开始检查并重命名文件夹为标准格式: {FEATURES_DIR}")
    
    rename_count = 0
    skip_count = 0
    error_count = 0
    
    # 先收集所有需要重命名的目录（避免迭代时修改）
    rename_list = []
    for folder in FEATURES_DIR.iterdir():
        if folder.is_dir():
            original_name = folder.name
            final_name = _normalize_feature_dir_name(original_name)
            
            # 如果已经符合规范，跳过
            if original_name == final_name:
                skip_count += 1
                continue
            
            final_path = folder.parent / final_name
            if final_path.exists():
                print(f"⚠️ 跳过: {original_name} -> {final_name} (目标已存在)")
                skip_count += 1
                continue
            
            rename_list.append((folder, final_path))
    
    print(f"📋 需要重命名: {len(rename_list)} 个, 已符合规范/跳过: {skip_count} 个")
    
    # 执行重命名操作
    for folder, final_path in rename_list:
        try:
            original_name = folder.name
            final_name = final_path.name
            
            if os.path.exists(str(folder)):
                folder.rename(final_path)
                print(f"✅ 重命名: {original_name} -> {final_name}")
                rename_count += 1
            else:
                print(f"⚠️ 跳过: {original_name} (文件夹不存在)")
                
        except PermissionError as e:
            print(f"❌ 重命名失败 {folder.name}: 文件被占用或无权限 - {e}")
            error_count += 1
            continue
        except Exception as e:
            print(f"❌ 重命名失败 {folder.name}: {e}")
            error_count += 1
            continue
    
    print(f"\n📊 重命名完成: 成功 {rename_count} 个, 跳过 {skip_count} 个, 失败 {error_count} 个")
    if rename_count > 0:
        print("🎉 features 文件夹重命名完成！")
else:
    print(f"⚠️ 特征目录不存在: {FEATURES_DIR}")

# ====== 修复文件夹大小写问题：创建小写符号链接 ======
# Qlib的FileFeatureStorage使用instrument.lower()查找文件夹，但实际文件夹是大写的
# 需要创建符号链接，将小写文件夹名映射到大写文件夹名
if FEATURES_DIR.exists():
    print(f"\n🔧 开始修复文件夹大小写问题（创建小写符号链接）: {FEATURES_DIR}")
    
    symlink_count = 0
    symlink_skipped = 0
    symlink_error = 0
    
    for folder in FEATURES_DIR.iterdir():
        if not folder.is_dir() or folder.is_symlink():
            continue
        
        original_name = folder.name
        lowercase_name = original_name.lower()
        
        # 如果文件夹名包含大写字母，创建小写符号链接
        if original_name != lowercase_name:
            lowercase_path = FEATURES_DIR / lowercase_name
            
            # 如果小写路径已存在，检查是否正确
            if lowercase_path.exists():
                if lowercase_path.is_symlink():
                    try:
                        target = lowercase_path.resolve()
                        if target == folder.resolve():
                            symlink_skipped += 1
                            continue
                        else:
                            # 符号链接指向错误，删除并重新创建
                            lowercase_path.unlink()
                    except Exception:
                        symlink_error += 1
                        continue
                else:
                    # 已存在真实文件夹，跳过
                    symlink_skipped += 1
                    continue
            
            # 创建符号链接
            try:
                lowercase_path.symlink_to(original_name)
                symlink_count += 1
                if symlink_count <= 5:  # 只显示前5个
                    print(f"✅ 创建符号链接: {lowercase_name} -> {original_name}")
            except Exception as e:
                print(f"❌ 创建符号链接失败 {lowercase_name}: {e}")
                symlink_error += 1
    
    print(f"📊 符号链接修复完成: 创建 {symlink_count} 个, 跳过 {symlink_skipped} 个, 失败 {symlink_error} 个")
    if symlink_count > 0:
        print("🎉 文件夹大小写问题已修复！")

# ============================================
# 从这里开始可以单独运行训练测试回测部分
# 跳过前面的数据下载和预处理
# ============================================

# ====== 文件后缀配置（用于区分不同脚本的输出文件） ======
OUTPUT_FILE_SUFFIX = "_alpha158fz7_tail_end"  # 此脚本的输出文件后缀

# 注册自定义算子（来自 examples/highfreq/highfreq_ops.py）
import importlib, highfreq_ops as _hf
_hf = importlib.reload(_hf)
import sys
sys.path.append("/home/intern0/qlib/examples/highfreq")
from qlib.data.ops import Operators, register_all_ops
from qlib.config import C
from highfreq_ops import Cut as HFCut, IntradayWindowVWAP as HFVWAP, DayShift as HFShift

# 当前进程注册，覆盖内置同名实现
Operators.register([HFCut, HFVWAP, HFShift])

# 写入全局配置，便于并行/子进程可见；仅刷新算子注册
C.custom_ops = [
    {"class": "Cut", "module_path": "highfreq_ops"},
    {"class": "IntradayWindowVWAP", "module_path": "highfreq_ops"},
    {"class": "DayShift", "module_path": "highfreq_ops"},
]
register_all_ops(C)
print("Registered custom ops: Cut, IntradayWindowVWAP, DayShift from highfreq_ops.")


import os
import sys
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.utils import flatten_dict, init_instance_by_config
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.model.gbdt import LGBModel
from qlib.backtest import backtest
from workflow import HighfreqWorkflow
from highfreq_ops import IntradayWindowVWAP, DayShift
from qlib.data.ops import Operators


# 确保当前路径包含 workflow.py
WORKDIR = "/home/intern0/qlib/examples/highfreq"
sys.path.insert(0, WORKDIR)  # 使用 insert(0) 确保优先级

# 清除可能的模块缓存，确保使用最新的 handler
import importlib
if 'highfreq_handler_alpha158fz7' in sys.modules:
    importlib.reload(sys.modules['highfreq_handler_alpha158fz7'])
    print("🔄 已重新加载 highfreq_handler_alpha158fz7 模块")

# 导入自定义操作符
from highfreq_ops import get_calendar_day, DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut,DayFirst,IntradayWindowVWAP, DayShift, If, Gt, Lt


# 配置自定义操作符
SPEC_CONF = {"custom_ops": [DayLast, DayFirst,FFillNan, BFillNan, Date, Select, IsNull , DayShift ,Cut,IntradayWindowVWAP, If, Gt, Lt],"expression_cache": None}

print("✅ 自定义操作符已配置")

# 天勤数据配置
KQ_DATA_DIR = "/home/intern0/qlib_data_tail_end"
STOCK_POOL_CSV = "/home/intern0/qlib/examples/highfreq/tail_end_candidates.csv"

# 天勤数据时间段配置
START_TIME = "2025-01-02 09:30:00"
END_TIME = "2025-10-27 14:57:00"  # 修改：测试集截止到09-29，14:57避免回测越界
TRAIN_END_TIME = "2025-07-15 14:57:00"  # 1-4月用于训练，14:57避免回测越界
TEST_START_TIME = "2025-07-16 09:30:00"  # 5-9月用于测试和回测

print(f"🔧 天勤数据配置:")
print(f"📂 数据目录: {KQ_DATA_DIR}")
print(f"📋 股票池: {STOCK_POOL_CSV}")
print(f"📅 训练时间段: {START_TIME} ~ {TRAIN_END_TIME}")
print(f"📅 测试时间段: {TEST_START_TIME} ~ {END_TIME}")


# 强制使用自定义路径，避免 HIGH_FREQ_CONFIG 的默认路径覆盖
print("🔧 强制设置自定义数据路径...")

# 方法1：直接设置配置，不使用 HIGH_FREQ_CONFIG
CUSTOM_QLIB_CONFIG = {
    "provider_uri": KQ_DATA_DIR,  # 强制使用天勤数据路径
    "dataset_cache": None,
    "expression_cache": "DiskExpressionCache",
    "region": REG_CN,
    **SPEC_CONF  # 添加自定义操作符配置
}

print(f"📂 强制使用数据路径: {KQ_DATA_DIR}")
print(f"🔧 配置内容: {CUSTOM_QLIB_CONFIG}")

# 尝试初始化，如果失败则使用备用方案
try:
    qlib.init(**CUSTOM_QLIB_CONFIG, joblib_backend="threading")
    print(f"✅ Qlib 已初始化，使用天勤数据: {KQ_DATA_DIR}")
except Exception as e:
    print(f"⚠️ 自定义路径初始化失败: {e}")
    print("🔄 尝试备用方案：使用默认配置但指定数据路径...")
    
    # 备用方案：使用 HIGH_FREQ_CONFIG 但强制覆盖路径
    from qlib.config import HIGH_FREQ_CONFIG
    BACKUP_CONFIG = HIGH_FREQ_CONFIG.copy()
    BACKUP_CONFIG["provider_uri"] = KQ_DATA_DIR
    BACKUP_CONFIG.update(SPEC_CONF)
    
    qlib.init(**BACKUP_CONFIG)
    print(f"✅ 使用备用配置初始化成功")

print(f"✅ 自定义操作符已加载")

# 验证数据路径是否正确生效

from qlib.config import C, QlibConfig
C.maxtasksperchild = 1  # 避免长进程累积资源

if isinstance(C.provider_uri, str):
    C.provider_uri = QlibConfig.DataPathManager.format_provider_uri(C.provider_uri)
    print(f"🔧 provider_uri 已转换为字典格式: {C.provider_uri}")
elif isinstance(C.provider_uri, dict):
    print(f"🔧 provider_uri 格式正确（字典）: {C.provider_uri}")
else:
    print(f"🔧 provider_uri 格式: {type(C.provider_uri)} = {C.provider_uri}")
    try:
        C.provider_uri = QlibConfig.DataPathManager.format_provider_uri(C.provider_uri)
        print(f"🔧 provider_uri 已转换为字典格式: {C.provider_uri}")
    except Exception as e:
        print(f"⚠️ 无法转换 provider_uri: {e}")

if isinstance(C.provider_uri, dict):
    actual_path = C.provider_uri.get('__DEFAULT_FREQ', C.provider_uri)
    print(f"🔍 验证：当前 Qlib 数据路径（字典格式）: {C.provider_uri}")
    print(f"   实际路径: {actual_path}")
    if str(actual_path) != KQ_DATA_DIR:
        print(f"⚠️ 警告：数据路径不匹配！")
        print(f"  期望路径: {KQ_DATA_DIR}")
        print(f"  实际路径: {actual_path}")
        print("💡 建议检查天勤数据转换是否完整")
    else:
        print(f"✅ 数据路径匹配正确")
else:
    print(f"🔍 验证：当前 Qlib 数据路径: {C.provider_uri}")


TEST_STOCK_COUNT = int(os.environ.get("TEST_STOCK_COUNT", "124"))

stock_pool_df = pd.read_csv(STOCK_POOL_CSV)
stock_codes_raw = stock_pool_df["code"].tolist()
print(f"📊 股票池包含 {len(stock_codes_raw)} 行（原始格式）")

# 去重：保留每个股票代码的第一次出现
stock_codes_raw_unique = stock_pool_df["code"].drop_duplicates().tolist()
if len(stock_codes_raw) != len(stock_codes_raw_unique):
    print(f"   ⚠️ 发现重复股票，去重后: {len(stock_codes_raw_unique)} 支（原 {len(stock_codes_raw)} 行）")
else:
    print(f"   ✅ 无重复股票: {len(stock_codes_raw_unique)} 支")
print(f"   前5个原始代码: {stock_codes_raw_unique[:5]}")

def convert_code_format(code):
    """统一股票代码格式"""
    code = str(code).strip().upper()
    if '.' in code:
        market, stock_code = code.split('.', 1)
        if market in ('SH', 'SSE'):
            return f"SH{stock_code}"
        if market in ('SZ', 'SZSE'):
            return f"SZ{stock_code}"
    if code.startswith(('SH', 'SZ')):
        return code
    if code.startswith(('6', '9')):
        return f"SH{code}"
    if code.startswith(('0', '3')):
        return f"SZ{code}"
    return code

# 对去重后的股票代码进行格式转换
candidate_codes = [convert_code_format(code) for code in stock_codes_raw_unique]
print(f"   转换后前5个代码: {candidate_codes[:5]}")

if TEST_STOCK_COUNT > 0:
    candidate_codes = candidate_codes[:TEST_STOCK_COUNT]
    print(f"🧪 测试模式（候选集）：仅保留前 {len(candidate_codes)} 支股票（目标 {TEST_STOCK_COUNT}）")

from qlib.data import D
from pathlib import Path
import pickle

FILTERED_MARKET_PATH = Path("filtered_market_tail_end.pkl")  # 使用Tail-End专用的缓存文件名
FORCE_REBUILD_MARKET = os.environ.get("FORCE_REBUILD_MARKET", "0").lower() in ("1", "true", "yes")
market_needs_refresh = FORCE_REBUILD_MARKET or (not FILTERED_MARKET_PATH.exists())
stock_codes = []

if market_needs_refresh or len(stock_codes) == 0:
    if not market_needs_refresh:
        print(f"⚠️ 缓存股票池为空，自动触发重新筛选...")
    market_needs_refresh = True
    stock_codes = candidate_codes
    print("⏳ filtered_market_tail_end.pkl 不存在/为空或强制重建，使用候选股票集合")
else:
    with open(FILTERED_MARKET_PATH, "rb") as f:
        stock_codes = pickle.load(f)
    print(f"✅ 载入缓存股票集合: {len(stock_codes)} 支 (来源 {FILTERED_MARKET_PATH})")
    if TEST_STOCK_COUNT > 0 and len(stock_codes) > TEST_STOCK_COUNT:
        stock_codes = stock_codes[:TEST_STOCK_COUNT]
        print(f"🧪 测试模式：缓存集合仅保留前 {len(stock_codes)} 支股票（目标 {TEST_STOCK_COUNT}）")

# 可选：按需抽查数据
RUN_DATA_PROBE = os.environ.get("RUN_DATA_PROBE", "0").lower() in ("1", "true", "yes")
if RUN_DATA_PROBE:
    instruments_file = Path(KQ_DATA_DIR) / "instruments" / "all.txt"
    if not instruments_file.exists():
        raise FileNotFoundError(instruments_file)
    lines = instruments_file.read_text().splitlines()
    codes = [line.split("\t")[0].strip().upper() for line in lines if line.strip()]
    start_time = "2025-01-02 09:30:00"
    end_time = "2025-10-27 14:57:00"
    fields = ["$close"]
    print(f"🔍 RUN_DATA_PROBE=1，抽查 {len(codes)} 支股票的数据可用性（默认前20支）")
    ok_codes, bad_codes = [], []
    max_test = int(os.environ.get("DATA_PROBE_LIMIT", "20"))
    for idx, code in enumerate(codes):
        if idx >= max_test:
            break
        try:
            print(f"   测试 {code}...", end=' ')
            df = D.features([code], fields, start_time=start_time, end_time=end_time, freq="1min")
            if df.empty:
                bad_codes.append(code)
                print("❌ 空数据")
            else:
                ok_codes.append(code)
                print(f"✅ 形状: {df.shape}")
        except Exception as e:
            bad_codes.append(code)
            print(f"❌ 错误: {e}")
    print(f"🔍 抽查完成: ✅ {len(ok_codes)} / ❌ {len(bad_codes)} （总样本 {max_test}）")
else:
    print("⏭️ 跳过逐只数据探测（设置 RUN_DATA_PROBE=1 可开启抽查）")

#直接使用workflow.py中的函数定义

wf = HighfreqWorkflow()

wf.start_time = START_TIME
wf.train_end_time = TRAIN_END_TIME
wf.test_start_time = TEST_START_TIME
wf.end_time = END_TIME


# 拷贝 handler 配置并指定 provider_uri
handler_config = wf.DATA_HANDLER_CONFIG0.copy()
handler_config["instruments"] = stock_codes
#handler_config["provider_uri"] = KQ_DATA_DIR   # ⚠️ 关键
handler_config["infer_processors"] = [
    {"class": "HighFreqNorm", "module_path": "highfreq_processor",
     "kwargs": {"fit_start_time": wf.start_time,
                "fit_end_time": wf.train_end_time}}
]


#训练
from qlib.workflow import R
from qlib.utils import init_instance_by_config
from qlib.contrib.model.gbdt import LGBModel
import pandas as pd

RECORDER_STARTED = False

def ensure_recorder():
    """确保 MLflow recorder 已启动并返回实例"""
    global RECORDER_STARTED
    try:
        recorder = R.get_recorder()
        if recorder is None:
            raise RuntimeError("recorder is None")
        return recorder
    except Exception:
        R.start(experiment_name="HF_MIN_BACKTEST", recorder_name="mlflow_recorder")
        RECORDER_STARTED = True
        return R.get_recorder()


# dataset task
# 确保使用正确的 handler 模块
import importlib
import os
handler_module_path = os.path.join(WORKDIR, "highfreq_handler_alpha158fz7.py")
if os.path.exists(handler_module_path):
    print(f"✅ 找到 handler 文件: {handler_module_path}")
    # 强制重新加载模块
    if 'highfreq_handler_alpha158fz7' in sys.modules:
        importlib.reload(sys.modules['highfreq_handler_alpha158fz7'])
        print("🔄 已重新加载 highfreq_handler_alpha158fz7 模块")
else:
    print(f"❌ 未找到 handler 文件: {handler_module_path}")

dataset_task = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "HighFreqHandler",   ## 使用我们修改过的 handler
            "module_path": "highfreq_handler_alpha158fz7",  # 使用 Alpha158 + FZ7 版本
            "kwargs": handler_config,
        },
        "segments": {
            "train": (wf.start_time, wf.train_end_time),
            "test": (wf.test_start_time, wf.end_time),
        },
        #"infer_processors": handler_config["infer_processors"],
        #"learn_processors": handler_config["infer_processors"],
        
    },
}


handler_cfg = dataset_task['kwargs']['handler']


# 修正时间段 和自定义数据时间段一致 否则会用默认路径时间 重要！！！
handler_cfg['kwargs'].update({
    'start_time': '2025-01-02 09:30:00',
    'end_time': '2025-10-27 14:57:00',  # 修改：测试集截止到09-29，14:57避免回测越界
    'fit_start_time': '2025-01-02 09:30:00',
    'fit_end_time': '2025-07-15 14:57:00'  # 修改：14:57避免回测越界
})

dataset_task['kwargs']['handler'] = handler_cfg

# 模型 task
_use_gpu = str(os.environ.get("USE_GPU", "0")).lower() in ("1", "true", "yes")
_lgbm_kwargs = {
    "loss": "mse",
    "learning_rate": 0.1,
    "n_estimators": 100,
    "num_leaves": 31,
    "min_child_samples": 20,    # 增加最小样本数
    "reg_alpha": 0.1,           # 添加L1正则化
    "reg_lambda": 0.1,          # 添加L2正则化
}
if _use_gpu:
    # 仅在设置 USE_GPU=1 时启用；若安装为CPU版LightGBM可能报错
    _lgbm_kwargs.update({
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
    })
    print("⚙️ LightGBM: 已启用 GPU 参数（需要 GPU 版 LightGBM）")
else:
    print("⚙️ LightGBM: 使用 CPU（设置 USE_GPU=1 可启用 GPU）")

model_task = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": _lgbm_kwargs,
}

# 整体 task
task = {
    "dataset": dataset_task,
    "model": model_task,
}


from qlib.utils import init_instance_by_config

# 在创建 handler 之前，验证模块路径
print(f"\n🔍 验证 handler 配置:")
print(f"   class: {handler_cfg.get('class')}")
print(f"   module_path: {handler_cfg.get('module_path')}")
print(f"   期望的模块: highfreq_handler_alpha158fz7")

# 尝试直接导入验证
try:
    import highfreq_handler_alpha158fz7 as hf_alpha158
    print(f"✅ 成功导入 highfreq_handler_alpha158fz7")
    print(f"   模块路径: {hf_alpha158.__file__}")
    print(f"   HighFreqHandler 类: {hf_alpha158.HighFreqHandler}")
    
    # 测试 get_feature_config（使用未绑定方法）
    # 创建一个最小化的实例来测试
    import types
    test_instance = types.SimpleNamespace()
    test_cfg = hf_alpha158.HighFreqHandler.get_feature_config(test_instance)
    feature_count = len(test_cfg['feature'][0])
    print(f"   ✅ get_feature_config() 返回特征数量: {feature_count}")
    if feature_count == 180:
        print(f"   ✅ 特征数量正确（180个：15 基础 + 158 Alpha158 + 7 FactorZoo）")
    else:
        print(f"   ⚠️ 特征数量不正确，期望180个，实际{feature_count}个")
        print(f"   前20个特征名称: {test_cfg['feature'][1][:20]}")
except Exception as e:
    print(f"❌ 导入或测试 highfreq_handler_alpha158fz7 失败: {e}")
    import traceback
    traceback.print_exc()

handler = init_instance_by_config(handler_cfg)
print(f"\n✅ Handler 创建完成: {type(handler)}")
print(f"   Handler 模块: {type(handler).__module__}")


from qlib.data.dataset import DatasetH

if market_needs_refresh:
    # 用同一个 handler，构造一个只含训练段的临时 DatasetH（很快）
    tmp_dataset = DatasetH(
        handler=handler,
        segments={'train': (START_TIME, TRAIN_END_TIME)},
    )

    # 只取训练期的 label
    train_df = tmp_dataset.prepare(['train'], col_set=['label'])[0]['label']
    labels = train_df[[c for c in train_df.columns if str(c).upper().startswith('LABEL')][0]]

    df = labels.reset_index()
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    per_day_cnt = df.groupby(['instrument', 'date'])[labels.name].apply(lambda s: s.notna().sum())
    total_days = len(sorted(df['date'].unique()))
    valid_days = (per_day_cnt == 240).groupby('instrument').sum()
    coverage = (valid_days / float(total_days)).fillna(0.0)

    stock_codes = coverage[coverage >= 0.95].index.tolist()
    FILTERED_MARKET_PATH.write_bytes(pickle.dumps(stock_codes))
    print(f'覆盖率≥95%的标的数: {len(stock_codes)} / 全部: {coverage.shape[0]}')
    print(f'✅ 筛选结果已保存 -> {FILTERED_MARKET_PATH}')

    del tmp_dataset, train_df, labels, df, coverage
else:
    print(f"✅ 使用缓存股票池：{len(stock_codes)} 支（设置 FORCE_REBUILD_MARKET=1 可重新筛选）")

if TEST_STOCK_COUNT > 0 and len(stock_codes) > TEST_STOCK_COUNT:
    stock_codes = stock_codes[:TEST_STOCK_COUNT]
    print(f"🧪 测试模式：最终仅保留前 {len(stock_codes)} 支股票（目标 {TEST_STOCK_COUNT}）")

print(f"📌 最终股票列表大小: {len(stock_codes)}")
print(stock_codes[:10])

from qlib.utils import init_instance_by_config

dataset = init_instance_by_config(dataset_task)

# 1) 强制把筛后的股票写回 handler
if hasattr(handler, 'set_instruments'):
    handler.set_instruments(stock_codes)
else:
    handler.instruments = stock_codes  # 大多数 DataHandler 支持直接覆盖

# 2) 清理 handler 的缓存数据，避免沿用旧全集数据
for buf in ['_data','_learn','_infer','_data_l','_data_i']:
    if hasattr(handler, buf):
        setattr(handler, buf, None)

# 3) 重新加载（只一次）
handler.setup_data()

# ========== 关键内存优化：分阶段构造数据集 ==========
# 只用训练段构造 dataset，训练完成后释放，再构造测试段
dataset_train = DatasetH(
    handler=handler,
    instruments=stock_codes,
    segments={'train': (START_TIME, TRAIN_END_TIME)},
)

# 训练前先做最小验证（仅 train）
from qlib.data.dataset.handler import DataHandlerLP
train_df = dataset_train.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
xtrain = train_df["feature"]; ytrain = train_df["label"]

# 后续预测/回测使用的对象名保持为 dataset_clean（用测试段）
dataset_clean = None

# 5) 训练后释放训练数据集，构造测试数据集（仅 test 段）
import gc as _gc_after_train
del train_df
_gc_after_train.collect()

# 为测试阶段单独构造 dataset（共享同一个 handler，避免重复加载）
dataset_clean = DatasetH(
    handler=handler,
    instruments=stock_codes,
    segments={'test': (TEST_START_TIME, END_TIME)},
)

# 构造测试集特征标签（仅 test）
test_df = dataset_clean.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
xtest = test_df["feature"]; ytest = test_df["label"]

if xtrain.size == 0 or xtest.size == 0:
    raise ValueError("❌ 数据为空！请检查股票池与 provider_uri 是否匹配天勤数据")
else:
    print(f"✅ 数据检查通过: 训练集 {xtrain.shape}, 测试集 {xtest.shape}")
    print(f"\n🔍 训练数据质量检查:")
    print(f"   训练特征形状: {xtrain.shape}")
    print(f"   训练标签形状: {ytrain.shape}")
    train_nan_count = xtrain.isnull().sum().sum()
    test_nan_count = xtest.isnull().sum().sum()
    print(f"   训练集NaN值数量: {train_nan_count}")
    print(f"   测试集NaN值数量: {test_nan_count}")
    if train_nan_count < xtrain.size * 0.5:
        print(f"   训练特征值范围: {xtrain.min().min():.6f} 到 {xtrain.max().max():.6f}")
        print(f"   训练特征平均值: {xtrain.mean().mean():.6f}")
        print(f"   训练特征标准差: {xtrain.std().std():.6f}")
    else:
        print("   ⚠️ 训练特征包含过多NaN值")
    if isinstance(ytrain, pd.DataFrame):
        label_col = ytrain.columns[0]; label_data = ytrain[label_col]
        print(f"   标签值范围: {label_data.min():.6f} 到 {label_data.max():.6f}")
        print(f"   标签平均值: {label_data.mean():.6f}")
        print(f"   标签标准差: {label_data.std():.6f}")
        print(f"   标签唯一值数量: {label_data.nunique()}")
    print("✅ 数据质量检查完成")

from qlib.utils import init_instance_by_config

# 在训练前/准备完成后，尝试提取并保存 HighFreqNorm 的标准化参数（用于实时预测严格对齐）
try:
    norm_params = None
    # handler 可能持有 infer_processors / learn_processors 已实例化的 Processor 列表
    proc_lists = []
    for attr in ("infer_processors", "learn_processors"):
        if hasattr(handler, attr):
            procs = getattr(handler, attr)
            if isinstance(procs, (list, tuple)):
                proc_lists.extend(list(procs))
    # 兼容某些实现将处理器挂在内部字段
    for backup_attr in ("_infer_processors", "_learn_processors"):
        if hasattr(handler, backup_attr):
            procs = getattr(handler, backup_attr)
            if isinstance(procs, (list, tuple)):
                proc_lists.extend(list(procs))
    # 找到 HighFreqNorm 实例
    for proc in proc_lists:
        try:
            cls_name = getattr(proc, "__class__", type(proc)).__name__
            if cls_name == "HighFreqNorm":
                # 读取训练时统计
                fmed = getattr(proc, "feature_med", {}) or {}
                fstd = getattr(proc, "feature_std", {}) or {}
                # 仅保存 price/volume 两组关键统计
                norm_params = {
                    "price": {"med": float(fmed.get("price", 0.0)), "mad": float(fstd.get("price", 1.0))},
                    "volume": {"med": float(fmed.get("volume", 0.0)), "mad": float(fstd.get("volume", 1.0))},
                }
                break
        except Exception:
            continue
    if norm_params is not None:
        try:
            ensure_recorder()
            R.save_objects(norm_params=norm_params)
            print(f"💾 已保存标准化参数到 recorder: {norm_params}")
        except Exception as e:
            print(f"⚠️ 保存标准化参数时出错: {e}")
    else:
        print("⚠️ 未找到 HighFreqNorm 处理器，无法保存标准化参数")
except Exception as _e_norm:
    print(f"⚠️ 保存标准化参数时出错: {_e_norm}")

# 确保MLflow run已启动（如果还没有）
recorder = ensure_recorder()

#dataset = init_instance_by_config(task["dataset"])
model = init_instance_by_config(task["model"])

# 开始训练
print("🚀 开始模型训练...")
print(f"   模型类型: {type(model).__name__}")
print(f"   训练样本数: {len(xtrain)}")
print(f"   特征数量: {xtrain.shape[1]}")

model.fit(dataset_train)

# 保存模型
R.save_objects(model=model)

print("✅ 模型训练完成")

# LightGBM 特征重要性
try:
    importance_gain = model.get_feature_importance(importance_type="gain")
    importance_split = model.get_feature_importance(importance_type="split")
    importance_split_aligned = importance_split.reindex(importance_gain.index).fillna(0.0)
    feature_importance_df = pd.DataFrame(
        {
            "feature": importance_gain.index,
            "importance_gain": importance_gain.values,
            "importance_split": importance_split_aligned.values,
        }
    ).sort_values(by="importance_gain", ascending=False)

    top_k = feature_importance_df.head(20)
    print("\n🔝 LightGBM 特征重要性（Top 20，Gain）：")
    print(top_k.to_string(index=False))

    importance_csv_path = Path(f"feature_importance{OUTPUT_FILE_SUFFIX}.csv")
    feature_importance_df.to_csv(importance_csv_path, index=False, encoding="utf-8-sig")
    print(f"💾 特征重要性已保存到 {importance_csv_path.resolve()}")

    try:
        R.save_objects(feature_importance=feature_importance_df)
        print("✅ 特征重要性已保存到 MLflow recorder")
    except Exception as save_err:
        print(f"⚠️ 无法保存特征重要性到 recorder: {save_err}")
except Exception as imp_err:
    print(f"⚠️ 计算特征重要性失败: {imp_err}")

# ========== FactorZoo 因子 IC 测试 ==========
print("\n" + "="*70)
print("📊 FactorZoo 因子 IC 测试")
print("="*70)

try:
    from scipy.stats import pearsonr
    
    # 7 个 FactorZoo 因子名称
    fz_factor_names = [
        "FZ_MAX_HIGH_MIN_LOW_RATIO",
        "FZ_MAD_HIGH_CLOSE_RATIO",
        "FZ_STD_CLOSE_RET_4",
        "FZ_KURT_CLOSE_RET_4",
        "FZ_CORR_HIGH_VOLUME",
        "FZ_MAX_CLOSE",
        "FZ_MIN_CLOSE_RET_7",
    ]
    
    # 重新获取训练数据（因为 train_df 可能已被删除）
    print("🔍 重新加载训练数据用于 IC 测试...")
    train_data_for_ic = dataset_train.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    
    # 获取训练数据的原始特征（标准化之前）
    # 需要从 handler 中获取原始数据，或者从 dataset 中获取
    print("🔍 提取 FactorZoo 因子原始值...")
    
    # 方法：从训练数据中提取特征列，找到对应的 FZ 因子列
    train_features_cols = train_data_for_ic["feature"].columns
    
    # 查找 FZ 因子对应的列索引
    fz_col_indices = {}
    for fz_name in fz_factor_names:
        # 在列名中查找（可能是 MultiIndex 或普通 Index）
        for idx, col in enumerate(train_features_cols):
            col_str = str(col)
            # 处理 MultiIndex 列名
            if isinstance(col, tuple):
                col_str = str(col[-1])  # 取最后一层
            if fz_name in col_str:
                fz_col_indices[fz_name] = idx
                break
    
    print(f"✅ 找到 {len(fz_col_indices)} 个 FactorZoo 因子列")
    for fz_name, col_idx in fz_col_indices.items():
        print(f"   {fz_name}: Column_{col_idx}")
    
    if len(fz_col_indices) == 0:
        print("⚠️ 未找到 FactorZoo 因子列，跳过 IC 测试")
    else:
        # 提取特征和标签数据
        X_train = train_data_for_ic["feature"]
        y_train = train_data_for_ic["label"]
        
        # 获取标签列（通常是 LABEL0）
        label_col = [c for c in y_train.columns if str(c).upper().startswith('LABEL')][0]
        y_values = y_train[label_col]
        
        # 按日期分组计算 IC
        print("📊 按日期分组计算 IC...")
        
        # 获取日期索引
        if isinstance(X_train.index, pd.MultiIndex):
            datetime_idx = X_train.index.get_level_values('datetime')
            dates = pd.to_datetime(datetime_idx).date
        else:
            dates = pd.to_datetime(X_train.index).date
        
        # 存储每日 IC 结果
        ic_results = {fz_name: [] for fz_name in fz_col_indices.keys()}
        ic_dates = []
        
        # 按日期分组
        unique_dates = sorted(set(dates))
        print(f"   总交易日数: {len(unique_dates)}")
        
        for date in unique_dates:
            date_mask = dates == date
            if date_mask.sum() < 10:  # 至少需要 10 个样本
                continue
            
            X_date = X_train[date_mask]
            y_date = y_values[date_mask]
            
            # 对齐索引（确保 X 和 y 的索引一致）
            common_idx = X_date.index.intersection(y_date.index)
            if len(common_idx) < 10:
                continue
            
            X_date_aligned = X_date.loc[common_idx]
            y_date_aligned = y_date.loc[common_idx]
            
            # 计算每个 FZ 因子的当日 IC
            for fz_name, col_idx in fz_col_indices.items():
                if col_idx >= X_date_aligned.shape[1]:
                    continue
                
                factor_values = X_date_aligned.iloc[:, col_idx]
                
                # 去除 NaN 和无穷值
                valid_mask = factor_values.notna() & np.isfinite(factor_values) & y_date_aligned.notna() & np.isfinite(y_date_aligned)
                if valid_mask.sum() < 10:
                    continue
                
                factor_clean = factor_values[valid_mask]
                y_clean = y_date_aligned[valid_mask]
                
                # 计算 Pearson 相关系数
                try:
                    ic, p_value = pearsonr(factor_clean.values, y_clean.values)
                    if np.isfinite(ic):
                        ic_results[fz_name].append(ic)
                except Exception:
                    continue
            
            if any(len(ic_results[fz_name]) > 0 for fz_name in fz_col_indices.keys()):
                ic_dates.append(date)
        
        # 计算 IC 统计指标
        print("\n📈 FactorZoo 因子 IC 统计结果:")
        print("="*70)
        
        ic_stats = []
        for fz_name in fz_col_indices.keys():
            ic_series = np.array(ic_results[fz_name])
            if len(ic_series) == 0:
                continue
            
            ic_mean = np.mean(ic_series)
            ic_std = np.std(ic_series)
            icir = ic_mean / ic_std if ic_std > 0 else 0.0
            ic_win_rate = (ic_series > 0).sum() / len(ic_series)  # IC 胜率
            ic_abs_mean = np.mean(np.abs(ic_series))
            
            ic_stats.append({
                "factor": fz_name,
                "IC_mean": ic_mean,
                "IC_std": ic_std,
                "ICIR": icir,
                "IC_win_rate": ic_win_rate,
                "IC_abs_mean": ic_abs_mean,
                "valid_days": len(ic_series),
            })
        
        # 输出统计结果
        if ic_stats:
            ic_stats_df = pd.DataFrame(ic_stats)
            ic_stats_df = ic_stats_df.sort_values(by="IC_mean", ascending=False)
            
            print(ic_stats_df.to_string(index=False))
            
            # 保存到 CSV
            ic_csv_path = Path(f"fz_factor_ic_stats{OUTPUT_FILE_SUFFIX}.csv")
            ic_stats_df.to_csv(ic_csv_path, index=False, encoding="utf-8-sig")
            print(f"\n💾 IC 统计结果已保存到 {ic_csv_path.resolve()}")
            
            # 保存到 MLflow
            try:
                R.save_objects(fz_factor_ic_stats=ic_stats_df)
                print("✅ IC 统计结果已保存到 MLflow recorder")
            except Exception as save_err:
                print(f"⚠️ 无法保存 IC 统计到 recorder: {save_err}")
            
            # 保存每日 IC 序列（用于进一步分析）
            ic_daily_df = pd.DataFrame({
                'date': ic_dates[:len(ic_results[list(fz_col_indices.keys())[0]])]
            })
            for fz_name in fz_col_indices.keys():
                if len(ic_results[fz_name]) > 0:
                    ic_daily_df[fz_name] = ic_results[fz_name][:len(ic_daily_df)]
            
            ic_daily_csv_path = Path(f"fz_factor_ic_daily{OUTPUT_FILE_SUFFIX}.csv")
            ic_daily_df.to_csv(ic_daily_csv_path, index=False, encoding="utf-8-sig")
            print(f"💾 每日 IC 序列已保存到 {ic_daily_csv_path.resolve()}")
        else:
            print("⚠️ 未计算出有效的 IC 统计结果")
        
        print("="*70)

except Exception as ic_err:
    print(f"⚠️ FactorZoo 因子 IC 测试失败: {ic_err}")
    import traceback
    traceback.print_exc()

# 模型训练后的验证
print(f"\n🔍 模型训练验证:")
try:
    # 用数据集对象验证（方案A）：避免对 DataFrame 直接 predict
    sample_pred = model.predict(dataset_clean)
    if isinstance(sample_pred, pd.DataFrame):
        sample_series = sample_pred.iloc[:, 0]
    else:
        sample_series = pd.Series(sample_pred)

    print(f"   预测样本数量: {len(sample_series)}")
    print(f"   预测值范围: {sample_series.min():.6f} 到 {sample_series.max():.6f}")
    print(f"   预测值标准差: {sample_series.std():.6f}")

    if sample_series.std() < 1e-10:
        print("   ⚠️ 警告：模型预测值都相同！")
    else:
        print("   ✅ 模型预测正常")

except Exception as e:
    print(f"   ❌ 模型验证失败: {e}")

#生成信号并记录
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import risk_analysis

# 确保实验记录已启动（如果还没有启动）
recorder = ensure_recorder()
print("✅ Recorder 已就绪")

# 再次保存一次标准化参数（确保在 recorder 启动后已落盘）
try:
    _norm_params_2 = None
    _proc_lists2 = []
    for _attr in ("infer_processors", "learn_processors", "_infer_processors", "_learn_processors"):
        if hasattr(handler, _attr):
            _procs = getattr(handler, _attr)
            if isinstance(_procs, (list, tuple)):
                _proc_lists2.extend(list(_procs))
    for _p in _proc_lists2:
        try:
            _cls = getattr(_p, "__class__", type(_p)).__name__
            if _cls == "HighFreqNorm":
                _fmed = getattr(_p, "feature_med", {}) or {}
                _fstd = getattr(_p, "feature_std", {}) or {}
                _norm_params_2 = {
                    "price": {"med": float(_fmed.get("price", 0.0)), "mad": float(_fstd.get("price", 1.0))},
                    "volume": {"med": float(_fmed.get("volume", 0.0)), "mad": float(_fstd.get("volume", 1.0))},
                }
                break
        except Exception:
            continue
    if _norm_params_2 is not None:
        R.save_objects(norm_params=_norm_params_2)
        print(f"💾 已保存标准化参数(确保recorder已启动): {_norm_params_2}")
    else:
        print("⚠️ 未能在当前 handler 上获取到 HighFreqNorm，用于保存参数的步骤被跳过")
except Exception as _e_norm2:
    print(f"⚠️ 保存标准化参数(启动后)时出错: {_e_norm2}")

# 将模型在 dataset 上的预测保存为信号
sr = SignalRecord(model, dataset_clean, recorder)
sr.generate()

# 修复：只生成14:40时间点的信号
print("🔧 生成14:40时间点的信号...")

# 先获取完整的预测结果
pred_full = model.predict(dataset_clean)

print(f"📊 完整预测结果数量: {len(pred_full)}")

# 处理DataFrame格式的pred_full
if isinstance(pred_full, pd.DataFrame):
    s_full = pred_full.iloc[:, 0]
else:
    s_full = pred_full

print(f"📊 完整预测结果统计:")
print(f"   最小值: {s_full.min():.6f}")
print(f"   最大值: {s_full.max():.6f}")
print(f"   平均值: {s_full.mean():.6f}")
print(f"   标准差: {s_full.std():.6f}")
print(f"   唯一值数量: {len(s_full.unique())}")

# 检查预测结果是否异常
if s_full.std() < 1e-10:
    print("⚠️ 警告：所有预测值都相同！")
    print("   可能原因：")
    print("   1. 模型没有正确训练")
    print("   2. 特征数据有问题")
    print("   3. 数据预处理有问题")
    
    # 检查前几个预测值
    print(f"   前10个预测值: {s_full.head(10).tolist()}")
else:
    print("✅ 预测结果正常，存在差异")

# 检查时间分布
if isinstance(s_full.index, pd.MultiIndex) and "datetime" in s_full.index.names:
    datetime_values = s_full.index.get_level_values('datetime')
    unique_times = datetime_values.unique()
    print(f"📊 完整预测时间点数量: {len(unique_times)}")
    print(f"📊 时间点范围: {unique_times.min()} 到 {unique_times.max()}")
    
    # 检查每个时间点的信号分布
    for time_point in sorted(unique_times)[:5]:  # 只检查前5个时间点
        time_mask = datetime_values == time_point
        time_signals = s_full[time_mask]
        print(f"   {time_point}: 数量={len(time_signals)}, 范围={time_signals.min():.6f}到{time_signals.max():.6f}")

# 过滤出14:40时间点的信号
if isinstance(s_full.index, pd.MultiIndex) and "datetime" in s_full.index.names:
    datetime_index = s_full.index.get_level_values('datetime')
    time_mask = datetime_index.time == pd.Timestamp('14:40').time()
    pred_14_40 = s_full[time_mask]
    print(f"📊 14:40时间点信号数量: {len(pred_14_40)}")
    
    if len(pred_14_40) > 0:
        print(f"📊 14:40信号统计:")
        print(f"   最小值: {pred_14_40.min():.6f}")
        print(f"   最大值: {pred_14_40.max():.6f}")
        print(f"   平均值: {pred_14_40.mean():.6f}")
        print(f"   标准差: {pred_14_40.std():.6f}")
        print(f"   唯一值数量: {len(pred_14_40.unique())}")
        
        # 检查每个日期的信号分布
        if isinstance(pred_14_40.index, pd.MultiIndex):
            # 将 numpy 日期数组转为 Pandas Index 再去重
            datetime_values = pred_14_40.index.get_level_values('datetime')
            unique_dates = pd.Index(datetime_values.date).unique()
            print(f"📊 14:40信号日期数量: {len(unique_dates)}")
            
            for date in sorted(list(unique_dates))[:3]:  # 只检查前3个日期
                date_values = datetime_values.date
                date_mask = (date_values == date)
                date_signals = pred_14_40[date_mask]
                print(f"   {date}: 数量={len(date_signals)}, 范围={date_signals.min():.6f}到{date_signals.max():.6f}")
    else:
        print("❌ 没有找到14:40时间点的信号")
        pred_14_40 = s_full  # 使用完整信号作为备选
else:
    print("⚠️ 无法过滤时间点，使用完整信号")
    pred_14_40 = s_full

# 修复后的代码：
import numpy as np
import pandas as pd

# 处理DataFrame格式的pred_14_40
if isinstance(pred_14_40, pd.DataFrame):
    # 如果是DataFrame，取第一列
    s = pred_14_40.iloc[:, 0]
else:
    s = pred_14_40

# 保证是Series
if not isinstance(s, pd.Series):
    s = pd.Series(s)

raw_signal_1440 = s.copy()

# 现在进行rank化
if isinstance(s.index, pd.MultiIndex) and "datetime" in s.index.names:
    # 按日期分组排名
    pred_ranked = s.groupby(level="datetime", group_keys=False).rank(pct=True) * 2 - 1
else:
    # 直接排名
    pred_ranked = s.rank(pct=True) * 2 - 1

# 检查排名结果
print(f"📊 排名前信号值范围: {s.min():.6f} 到 {s.max():.6f}")
print(f"📊 排名后信号值范围: {pred_ranked.min():.6f} 到 {pred_ranked.max():.6f}")

# 如果排名后所有值都相同，说明原始信号有问题
if pred_ranked.std() < 1e-10:
    print("⚠️ 排名后信号值都相同，使用原始信号值")
    pred_processed = s  # 使用原始信号值
else:
    pred_processed = pred_ranked

print(f"✅ 14:40信号生成完成，信号数量: {len(pred_processed)}")
if isinstance(pred_processed.index, pd.MultiIndex) and "datetime" in pred_processed.index.names:
    datetime_values = pred_processed.index.get_level_values('datetime')
    print(f"📊 信号时间范围: {datetime_values.min()} 到 {datetime_values.max()}")
    print(f"📊 信号时间点数量: {len(datetime_values.unique())}")
    print(f"📊 信号时间点: {sorted(datetime_values.unique())}")
    
    # 添加信号值统计
    print(f"📊 信号值统计:")
    print(f"   最小值: {pred_processed.min():.6f}")
    print(f"   最大值: {pred_processed.max():.6f}")
    print(f"   平均值: {pred_processed.mean():.6f}")
    print(f"   标准差: {pred_processed.std():.6f}")
    
    # 检查信号值分布
    positive_signals = pred_processed[pred_processed > 0]
    negative_signals = pred_processed[pred_processed < 0]
    print(f"   正值信号数量: {len(positive_signals)}")
    print(f"   负值信号数量: {len(negative_signals)}")
    
    # 检查阈值以上的信号
    threshold_signals = pred_processed[pred_processed >= 0.0]  # 更新阈值
    print(f"   阈值(0.0)以上信号数量: {len(threshold_signals)}")
    
    if len(threshold_signals) > 0:
        print(f"   阈值以上信号: {threshold_signals.head(10).to_dict()}")
    else:
        print("   ⚠️ 没有信号超过阈值0.0")
        
else:
    print(f"📊 信号索引类型: {type(pred_processed.index)}")
    print(f"📊 信号值统计:")
    print(f"   最小值: {pred_processed.min():.6f}")
    print(f"   最大值: {pred_processed.max():.6f}")
    print(f"   平均值: {pred_processed.mean():.6f}")
    print(f"   标准差: {pred_processed.std():.6f}")

# ====== 计算样本外 IC 衰减曲线 ======
try:
    signal_instruments = sorted(set(raw_signal_1440.index.get_level_values("instrument")))
    ic_horizons = [1, 5, 10, 20, 40, 60, 90, 120, 180, 240]
    compute_ic_decay_curve(raw_signal_1440, ic_horizons, signal_instruments)
except Exception as ic_decay_err:
    print(f"⚠️ IC 衰减曲线计算失败: {ic_decay_err}")

import pandas as pd
import numpy as np
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.strategy.base import BaseStrategy

class WorkingLowFreqStrategy(BaseStrategy):
    """可工作的低频日交易策略"""

    def __init__(
        self,
        signal_time="14:40",
        buy_time="14:45",
        sell_time="10:46",
        topk=50,
        signal=None,
        signal_threshold=0.0,  # 降低阈值，让更多信号通过
        n_drop=3,  
        rebalance_ratio=0.2,
        max_position_ratio=0.8,
        min_trade_amount=10000,
        max_trade_amount=10000000,
        max_sell_ratio=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.signal_time = pd.Timestamp(signal_time).time()
        self.buy_time = pd.Timestamp(buy_time).time()
        self.sell_time = pd.Timestamp(sell_time).time()
        self.topk = int(topk)
        self.signal = signal
        self.signal_threshold = float(signal_threshold)
        self.n_drop = int(n_drop)
        self.rebalance_ratio = float(rebalance_ratio)
        self.max_position_ratio = float(max_position_ratio)
        self.min_trade_amount = float(min_trade_amount)
        self.max_trade_amount = float(max_trade_amount)
        self.max_sell_ratio = float(max_sell_ratio)
        
        self.daily_signals = {}

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        current_time = trade_start_time.time()
        current_date = trade_start_time.date()
        
        # 1) 14:40 记录信号
        if current_time == self.signal_time:
            self._record_signal(current_date, trade_start_time, trade_end_time)
            return TradeDecisionWO([], self)

        # 2) 14:45 买入
        if current_time == self.buy_time:
            if current_date in self.daily_signals:
                return self._generate_buy_decision(
                    self.daily_signals[current_date], 
                    trade_start_time, 
                    trade_end_time
                )
            return TradeDecisionWO([], self)

        # 3) 10:46 卖出
        if current_time == self.sell_time:
            return self._generate_sell_decision(trade_start_time, trade_end_time)

        return TradeDecisionWO([], self)

    def _record_signal(self, current_date, trade_start_time, trade_end_time):
        """记录14:40的信号"""
        try:
            if isinstance(self.signal, pd.Series):
                if isinstance(self.signal.index, pd.MultiIndex):
                    date_mask = self.signal.index.get_level_values('datetime').date == current_date
                    if date_mask.any():
                        pred_score_all = self.signal[date_mask]
                        datetime_level = pred_score_all.index.get_level_values('datetime')
                        time_mask_14_40 = datetime_level.time == pd.Timestamp('14:40').time()
                        
                        if time_mask_14_40.any():
                            pred_score_14_40 = pred_score_all[time_mask_14_40]
                            
                            if isinstance(pred_score_14_40.index, pd.MultiIndex):
                                instrument_level = pred_score_14_40.index.get_level_values('instrument')
                                pred_score = pd.Series(pred_score_14_40.values, index=instrument_level)
                            else:
                                pred_score = pred_score_14_40
                            
                            self.daily_signals[current_date] = pred_score
                            print(f"📊 14:40信号记录: {current_date}, 信号数量: {len(pred_score)}")
                            print(f"   信号值范围: {pred_score.min():.6f} 到 {pred_score.max():.6f}")
                            print(f"   阈值以上信号: {len(pred_score[pred_score >= 0.0])}")
                        else:
                            print(f"⚠️ 14:40时间点无信号: {current_date}")
                    else:
                        print(f"⚠️ 当日无信号数据: {current_date}")
        except Exception as e:
            print(f"❌ 信号记录错误: {e}")
    
    def _safe_deal_price(self, stock: str, start_time, end_time, direction) -> float:
        """安全获取成交价格"""
        try:
            price_data = self.trade_exchange.get_deal_price(
                stock_id=stock,
                start_time=start_time,
                end_time=end_time,
                direction=direction,
            )
            price = None
            if isinstance(price_data, (int, float, np.number)):
                price = float(price_data)
            elif hasattr(price_data, "iloc") and len(price_data) > 0:
                price = float(price_data.iloc[-1])
            elif hasattr(price_data, "__len__") and len(price_data) > 0:
                price = float(price_data[-1])
            if price is None or not np.isfinite(price) or price <= 0:
                return np.nan
            return price
        except Exception:
            return np.nan

    def _generate_buy_decision(self, pred_score, trade_start_time, trade_end_time):
        """分批买入"""
        print(f"🔍 买入决策: 信号数量={len(pred_score) if pred_score is not None else 0}")
        
        if pred_score is None or len(pred_score) == 0:
            print("❌ 买入决策: 无信号数据")
            return TradeDecisionWO([], self)
        
        if isinstance(pred_score.index, pd.MultiIndex):
            instrument_level = pred_score.index.get_level_values('instrument')
            pred_score = pd.Series(pred_score.values, index=instrument_level)
        
        print(f"📊 买入前信号统计: 数量={len(pred_score)}, 范围={pred_score.min():.6f}到{pred_score.max():.6f}")
        
        if self.signal_threshold > 0:
            pred_score = pred_score[pred_score >= self.signal_threshold]
            print(f"📊 阈值过滤后: 数量={len(pred_score)}")
        
        if len(pred_score) == 0:
            print("❌ 买入决策: 阈值过滤后无信号")
            return TradeDecisionWO([], self)
        
        top_idx = pred_score.sort_values(ascending=False).head(self.topk).index.tolist()
        if not top_idx:
            print("❌ 买入决策: 无有效股票")
            return TradeDecisionWO([], self)
        
        print(f"✅ 买入决策: 选择{len(top_idx)}支股票")
        print(f"   股票代码: {top_idx[:5]}...")  # 显示前5个

        orders = []
        cash = self.trade_position.get_cash()
        current_positions = set(self.trade_position.get_stock_list())
        total_assets = cash
        for s in current_positions:
            amt = self.trade_position.get_stock_amount(s)
            if amt > 0:
                p = self._safe_deal_price(s, trade_start_time, trade_end_time, OrderDir.BUY)
                if np.isfinite(p):
                    total_assets += amt * p

        new_stocks = [s for s in top_idx if s not in current_positions]
        if new_stocks:
            rebalance_cash = total_assets * self.rebalance_ratio
            weight_per_new_stock = rebalance_cash / len(new_stocks)
            
            for s in new_stocks:
                price = self._safe_deal_price(s, trade_start_time, trade_end_time, OrderDir.BUY)
                if not np.isfinite(price) or price <= 0:
                    continue
                target_value = max(self.min_trade_amount, min(weight_per_new_stock, self.max_trade_amount))
                shares = int(target_value / price)
                if shares < 1:
                    continue
                orders.append(Order(
                    stock_id=s, 
                    amount=shares, 
                    direction=OrderDir.BUY,
                    start_time=trade_start_time, 
                    end_time=trade_end_time, 
                    factor=1.0
                ))

        existing_stocks = [s for s in top_idx if s in current_positions]
        if existing_stocks:
            rebalance_cash2 = total_assets * self.rebalance_ratio * 0.5
            weight_per_existing = rebalance_cash2 / len(existing_stocks)
            
            for s in existing_stocks:
                price = self._safe_deal_price(s, trade_start_time, trade_end_time, OrderDir.BUY)
                if not np.isfinite(price) or price <= 0:
                    continue
                target_value = max(self.min_trade_amount, min(weight_per_existing, self.max_trade_amount))
                shares = int(target_value / price)
                if shares < 1:
                    continue
                orders.append(Order(
                    stock_id=s, 
                    amount=shares, 
                    direction=OrderDir.BUY,
                    start_time=trade_start_time, 
                    end_time=trade_end_time, 
                    factor=1.0
                ))

        return TradeDecisionWO(orders, self)

    def _generate_sell_decision(self, trade_start_time, trade_end_time):
        """分批卖出"""
        orders = []
        current_positions = list(self.trade_position.get_stock_list())
        
        if not current_positions:
            return TradeDecisionWO([], self)

        for s in current_positions:
            amount = self.trade_position.get_stock_amount(s)
            if amount <= 0:
                continue
            price = self._safe_deal_price(s, trade_start_time, trade_end_time, OrderDir.SELL)
            if not np.isfinite(price) or price <= 0:
                continue
            sell_amount = max(1, int(amount * min(self.rebalance_ratio, self.max_sell_ratio)))
            orders.append(Order(
                stock_id=s, 
                amount=sell_amount, 
                direction=OrderDir.SELL,
                start_time=trade_start_time, 
                end_time=trade_end_time, 
                factor=1.0
            ))
        
        return TradeDecisionWO(orders, self)


# ====== 计算等权隔夜收益 Benchmark ======
# 使用和 label 完全一致的收益计算方式（仅在 10:46 记录隔夜收益）
def calculate_equal_weight_overnight_benchmark(
    stock_codes: list,
    start_time: str,
    end_time: str,
) -> pd.Series:
    """
    计算等权隔夜收益 benchmark（10:46 记入昨日隔夜收益，其余时间为 0）
    """
    from qlib.data import D

    print("\n" + "=" * 70)
    print("📊 计算等权隔夜收益Benchmark")
    print("=" * 70)
    print(f"   股票池: {len(stock_codes)} 支")
    print(f"   时间范围: {start_time} ~ {end_time}")
    
    # 获取所有股票的数据
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    all_data = D.features(
        stock_codes,
        fields,
        start_time=start_time,
        end_time=end_time,
        freq="1min"
    )
    
    if all_data.empty:
        raise ValueError("无法获取benchmark数据，all_data为空")
    
    print(f"   数据形状: {all_data.shape}")
    
    # 存储每支股票的隔夜收益
    stock_overnight_returns = {}
    
    for inst in stock_codes:
        try:
            # 获取单支股票数据
            if isinstance(all_data.index, pd.MultiIndex):
                if inst not in all_data.index.get_level_values("instrument").unique():
                    continue
                df = all_data.xs(inst, level="instrument", drop_level=True)
            else:
                df = all_data.loc[inst] if inst in all_data.index else None
                if df is None:
                    continue
            
            if df.empty:
                continue
            
            # 时间窗口掩码（和handler中完全一致）
            dt = pd.to_datetime(df.index)
            t_int = dt.hour * 100 + dt.minute
            # 下午窗口 [14:41, 14:50) - 当日下午
            mask_afternoon = (t_int >= 1441) & (t_int < 1450)
            # 上午窗口 [10:11, 10:20) - 当日上午
            mask_morning = (t_int >= 1011) & (t_int < 1020)
            
            # 典型价格与成交量（和handler中完全一致）
            tp = (df["$open"] + df["$high"] + df["$low"] + df["$close"]) / 4.0
            vol = df["$volume"]
            
            # 按"日期"聚合成日级 VWAP（和handler中完全一致）
            day_key = pd.Index(dt.date)
            def window_vwap(mask):
                num = (tp.where(mask, 0.0) * vol.where(mask, 0.0)).groupby(day_key).sum()
                den = vol.where(mask, 0.0).groupby(day_key).sum()
                wv = (num / den.replace(0.0, np.nan)).astype("float64")
                wv = wv.ffill()  # 前向填充
                return wv
            
            afternoon_vwap = window_vwap(mask_afternoon)
            morning_vwap = window_vwap(mask_morning)
            
            # 隔夜收益率计算（和handler中完全一致）
            # 当日标签 = 次日上午VWAP / 当日下午VWAP - 1
            morning_next_day = morning_vwap.shift(-1)  # 次日上午VWAP对齐到当日
            label_vwap = morning_next_day / afternoon_vwap - 1
            
            # 截断异常值（和handler中完全一致）
            label_vwap = label_vwap.clip(
                lower=label_vwap.quantile(0.01),
                upper=label_vwap.quantile(0.99)
            )
            
            # 关键修复：只在10:46时间点记录隔夜收益，其他时间点为0
            # 策略逻辑：
            #   - 在日期D的14:45买入
            #   - 在日期D+1的10:46卖出
            #   - 收益 = 日期D+1上午VWAP / 日期D下午VWAP - 1
            # label_vwap[日期D] = 日期D+1上午VWAP / 日期D下午VWAP - 1
            # 所以在日期D+1的10:46卖出时，应该记录label_vwap[日期D]的值
            
            # 创建分钟级收益Series，初始化为0
            broadcast = pd.Series(0.0, index=df.index, dtype="float64")
            
            # 获取所有10:46时间点
            dt = pd.to_datetime(df.index)
            sell_time_mask = (dt.hour == 10) & (dt.minute == 46)
            sell_times = df.index[sell_time_mask]
            
            # 为每个10:46时间点分配对应的隔夜收益
            # 在日期D的10:46卖出时，记录的是label_vwap[日期D-1]（从日期D-1下午到日期D上午的收益）
            sorted_dates = sorted(label_vwap.index)
            for sell_time in sell_times:
                sell_date = pd.to_datetime(sell_time).date()
                prev_date = None
                for date in reversed(sorted_dates):
                    if date < sell_date:
                        prev_date = date
                        break
                if prev_date is not None and prev_date in label_vwap.index:
                    broadcast.loc[sell_time] = label_vwap.loc[prev_date]
            
            stock_overnight_returns[inst] = broadcast
            
        except Exception as e:
            print(f"   ⚠️ {inst} 计算失败: {e}")
            continue
    
    if not stock_overnight_returns:
        raise ValueError("没有成功计算任何股票的隔夜收益")
    print(f"   ✅ 成功计算 {len(stock_overnight_returns)} 支股票的隔夜收益")
    
    # 将所有股票的隔夜收益对齐到相同的时间索引
    # 使用所有数据的公共时间索引（分钟级）
    if isinstance(all_data.index, pd.MultiIndex):
        common_index = all_data.index.get_level_values("datetime").unique()
    else:
        common_index = all_data.index
    
    aligned_returns = []
    for _, ret_series in stock_overnight_returns.items():
        aligned_returns.append(ret_series.reindex(common_index))

    if aligned_returns:
        returns_df = pd.concat(aligned_returns, axis=1)
        returns_df.columns = list(stock_overnight_returns.keys())
        equal_weight_benchmark = returns_df.mean(axis=1)
        equal_weight_benchmark = equal_weight_benchmark.ffill().fillna(0.0)
    else:
        equal_weight_benchmark = pd.Series(0.0, index=common_index)
    
    print(f"   ✅ Benchmark计算完成: {len(equal_weight_benchmark)} 个时间点")
    print(
        f"   收益范围: {equal_weight_benchmark.min():.6f} ~ {equal_weight_benchmark.max():.6f}"
    )
    print(f"   平均收益: {equal_weight_benchmark.mean():.6f}")
    
    return equal_weight_benchmark

# 计算等权隔夜收益benchmark
print("\n" + "="*70)
print("📊 计算等权隔夜收益Benchmark")
print("="*70)

benchmark_series = calculate_equal_weight_overnight_benchmark(
    stock_codes=stock_codes,
    start_time=dataset_clean.segments["test"][0],
    end_time=END_TIME,
)

port_analysis_config_minute = {
    "strategy": {
        "class": "WorkingLowFreqStrategy",  # 使用自定义策略
        "module_path": "__main__",  # 在notebook中定义，使用__main__
        "kwargs": {
            "signal": pred_processed,
            "signal_time": "14:40",  # 14:40 生成信号
            "buy_time": "14:45",     # 14:45 买入
            "sell_time": "10:46",    # 10:46 卖出
            "topk": 50,  # 减少股票数量便于测试
            "n_drop": 5,
            # 新增参数
            "rebalance_ratio": 0.2,      # 每次调仓30%
            "max_position_ratio": 0.8,   # 最大仓位80%
            "min_trade_amount": 10000,   # 最小交易1万
            "max_trade_amount": 5000000, # 最大交易500万
            "max_sell_ratio": 0.2,
            #"allowed_buy_windows": [("14:41", "14:50"), ("09:31", "10:20")],
            #"allowed_sell_windows": [("09:31", "10:20"), ("14:40", "14:58")],
            #"window_execution": "first",
        },
    },
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "1min",  # 保持分钟级，但策略控制交易时机
            "generate_portfolio_metrics": True,
        },
    },
    "backtest": {
        "start_time": dataset_clean.segments["test"][0],
        "end_time": None,
        "account": 1e9,
        "benchmark": benchmark_series,  # 使用自定义等权隔夜收益benchmark
        "exchange_kwargs": {
            "freq": "1min",
            "deal_price": "close",  
            "limit_threshold": 0.095,   
            "open_cost": 0.0003,      
            "close_cost": 0.001,  
            "min_cost": 5,              
            "impact_cost": 0,      
           
        },
    },
}

import pandas as pd
from qlib.data.data import Cal


def compute_ic_decay_curve(signal_series, horizons, codes):
    """计算并绘制信号与未来收益的 IC 衰减曲线。

    参数
    ----
    signal_series : pd.Series
        MultiIndex (instrument, datetime) 的预测信号（未排名的原始值）。
    horizons : List[int]
        以“分钟”为单位的未来时间跨度。
    codes : List[str]
        股票代码列表，用于拉取价格。
    """
    import matplotlib.pyplot as plt

    if not isinstance(signal_series.index, pd.MultiIndex) or "datetime" not in signal_series.index.names:
        raise ValueError("signal_series 需要 MultiIndex，且包含 datetime 这一层。")

    horizons = sorted(set(int(h) for h in horizons if h > 0))
    if not horizons:
        raise ValueError("horizons 为空，请至少提供一个正整数。")

    # 信号透视表：index=datetime, columns=instrument
    signal_df = signal_series.rename("signal").reset_index().pivot(index="datetime", columns="instrument", values="signal").sort_index()
    signal_df.columns = signal_df.columns.astype(str)
    signal_dates = signal_df.index

    # 加载分钟级交易日历，准备定位 horizon
    cal_index = pd.DatetimeIndex(Cal.load_calendar(freq="1min", future=False))
    cal_pos = {dt: idx for idx, dt in enumerate(cal_index)}
    pos_list = [cal_pos[dt] for dt in signal_dates if dt in cal_pos]
    if not pos_list:
        raise ValueError("信号时间点未在交易日历中找到，请检查数据。")

    max_h = max(horizons)
    start_pos = min(pos_list)
    end_pos_needed = min(max(pos_list) + max_h, len(cal_index) - 1)
    start_dt_fetch = cal_index[start_pos]
    end_dt_fetch = cal_index[end_pos_needed]

    # 拉取分钟收盘价
    price_raw = D.features(
        codes,
        ["$close"],
        start_time=start_dt_fetch.strftime("%Y-%m-%d %H:%M:%S"),
        end_time=end_dt_fetch.strftime("%Y-%m-%d %H:%M:%S"),
        freq="1min",
    )
    if price_raw.empty:
        raise ValueError("从 qlib 获取价格数据失败，price_raw 为空。")

    price_raw.columns = [col.replace("$", "").lower() for col in price_raw.columns]
    price_df = price_raw.reset_index().pivot(index="datetime", columns="instrument", values="close").sort_index()
    price_df.columns = price_df.columns.astype(str)

    ic_records = []
    horizon_stats = []

    for horizon in horizons:
        horizon_ics = []
        for dt in signal_dates:
            if dt not in cal_pos:
                continue
            start_pos = cal_pos[dt]
            target_pos = start_pos + horizon
            if target_pos >= len(cal_index):
                continue
            dt_future = cal_index[target_pos]
            if dt not in price_df.index or dt_future not in price_df.index:
                continue

            price_now = price_df.loc[dt]
            price_future = price_df.loc[dt_future]
            returns = (price_future / price_now) - 1.0

            df_tmp = pd.concat([signal_df.loc[dt], returns], axis=1, keys=["signal", "ret"]).dropna()
            if len(df_tmp) < 5:
                continue

            ic_value = df_tmp["signal"].corr(df_tmp["ret"])
            if pd.notna(ic_value):
                horizon_ics.append((dt, ic_value))
                ic_records.append({"datetime": dt, "horizon_min": horizon, "ic": ic_value})

        if horizon_ics:
            ic_values = [val for _, val in horizon_ics]
            mean_ic = float(np.mean(ic_values))
            std_ic = float(np.std(ic_values, ddof=0))
            horizon_stats.append({
                "horizon_min": horizon,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "icir": mean_ic / std_ic if std_ic > 0 else 0.0,
                "positive_ratio": float(np.mean(np.array(ic_values) > 0)),
                "valid_days": len(ic_values),
            })

    ic_daily_df = pd.DataFrame(ic_records)
    ic_stats_df = pd.DataFrame(horizon_stats)

    if not ic_stats_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(ic_stats_df["horizon_min"], ic_stats_df["mean_ic"], marker="o", label="Mean IC")
        plt.fill_between(
            ic_stats_df["horizon_min"],
            ic_stats_df["mean_ic"] - ic_stats_df["std_ic"],
            ic_stats_df["mean_ic"] + ic_stats_df["std_ic"],
            color="skyblue",
            alpha=0.3,
            label="±1 Std"
        )
        plt.axhline(0, color="red", linestyle="--", linewidth=1)
        plt.title("IC Decay Curve (14:40 Signal)", fontsize=16)
        plt.xlabel("Horizon (minutes)", fontsize=13)
        plt.ylabel("Information Coefficient", fontsize=13)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"ic_decay_curve{OUTPUT_FILE_SUFFIX}.png", dpi=300)
        plt.show()

    ic_stats_df.to_csv(f"ic_decay_stats{OUTPUT_FILE_SUFFIX}.csv", index=False, encoding="utf-8-sig")
    ic_daily_df.to_csv(f"ic_decay_daily{OUTPUT_FILE_SUFFIX}.csv", index=False, encoding="utf-8-sig")

    print("\n📉 IC 衰减统计已生成并保存：")
    if not ic_stats_df.empty:
        print(ic_stats_df.to_string(index=False))
    else:
        print("⚠️ 未能计算出有效的 IC 值（可能因可用样本过少或价格缺失）。")

    return ic_stats_df, ic_daily_df


# 1) 预测的最后一分钟

dt_max = pred_processed.index.get_level_values("datetime").max()
dt_max = pd.Timestamp(dt_max)

# 2) 加载分钟交易日历并转成 DatetimeIndex
cal_list = Cal.load_calendar(freq="1min", future=False)  # 返回 list-like
cal_1min = pd.DatetimeIndex(cal_list)                    # 关键：转为 DatetimeIndex

# 3) 用 searchsorted 找到 dt_max 之后的"下一个交易分钟"；若没有，则裁剪到最后一分钟
pos = cal_1min.searchsorted(dt_max, side="right")
if pos >= len(cal_1min):
    # 没有更多分钟数据，回测终止在已有的最后一分钟
    # 二选一：
    next_min = dt_max             # 就停在预测的最后一分钟
    # next_min = cal_1min[-1]     # 或者停在日历中的最后一分钟（等价于 dt_max）
else:
    next_min = cal_1min[pos]

# 4) 设置回测 end_time
port_analysis_config_minute ["backtest"]["end_time"] = next_min

#显示设置生成日频报告
port_analysis_config_minute ['generate_portfolio_metrics'] = True


#运行回测并生成报告
par = PortAnaRecord(recorder, port_analysis_config_minute, risk_analysis_freq="1min")

port_analysis_config_minute["strategy"]["kwargs"]["signal"] = pred_processed
sig = port_analysis_config_minute["strategy"]["kwargs"]["signal"]
print(type(sig), sig.min(), sig.max(), sig.std())
par.generate()

# 4) 读取并展示关键结果（简版）
report = recorder.load_object("portfolio_analysis/report_normal_1min.pkl") 
analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1min.pkl")  # 风险指标汇总

print("=== 核心指标（含成本，日频）===")
# 常见关键信息：超额收益、有无成本、IR、年化等（字段名可能随版本略有差异）
try:
    er_wo_cost = risk_analysis(report["return"] - report["bench"])
    er_w_cost = risk_analysis(report["return"] - report["bench"] - report["cost"])
    print("超额（无成本）:\n", er_wo_cost.round(4).to_string())
    print("\n超额（含成本）:\n", er_w_cost.round(4).to_string())
except Exception:
    print(analysis_df.round(4).to_string())

# 5) 可选：简单预览收益曲线头部
print("\n=== 回测收益曲线（头部）===")
print(report.head())       

import matplotlib.pyplot as plt
import pandas as pd


# 1) Compute cumulative returns
# 1) Compute cumulative returns
import numpy as np
report = report.astype('float32', copy=False)
arr_wo_cost = (1 + (report['return'] - report['bench'])).to_numpy(dtype='float32')
arr_w_cost = (1 + (report['return'] - report['bench'] - report['cost'])).to_numpy(dtype='float32')
arr_bench = (1 + report['bench']).to_numpy(dtype='float32')

def block_cumprod(arr, chunk=5000):
    out = np.empty_like(arr)
    N = len(arr)
    prev = 1.0
    for i in range(0, N, chunk):
        seg = arr[i:i+chunk]
        seg_cp = np.cumprod(seg) * prev
        out[i:i+chunk] = seg_cp
        prev = seg_cp[-1] if len(seg_cp) else prev
    return out - 1

cum_return_wo_cost = pd.Series(block_cumprod(arr_wo_cost, chunk=5000), index=report.index)
cum_return_w_cost = pd.Series(block_cumprod(arr_w_cost, chunk=5000), index=report.index)
cum_bench = pd.Series(block_cumprod(arr_bench, chunk=5000), index=report.index)

# 2) Plot
plt.figure(figsize=(12, 6))
plt.plot(cum_return_wo_cost, label='Excess Return (No Cost)', color='green')
plt.plot(cum_return_w_cost, label='Excess Return (With Cost)', color='red', alpha=0.8)
plt.plot(cum_bench, label='Benchmark', color='blue', linestyle='--')

plt.title("Daily Backtest Cumulative Returns", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Cumulative Return", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig(f"backtest_curve{OUTPUT_FILE_SUFFIX}.png", dpi=150)
plt.close()

if RECORDER_STARTED:
    try:
        R.end()
        RECORDER_STARTED = False
        print("🛑 已结束当前 MLflow 运行")
    except Exception as e:
        print(f"⚠️ 结束 MLflow 运行时出错: {e}")

# ============================================
# 第三部分：14:40预测模块
# ============================================

# ====== 数据完整性检查 ======
def check_data_integrity():
    """检查数据完整性"""
    from qlib.data import D
    
    print("\n🔍 开始数据完整性检查...")
    
    # 检查几个关键股票的数据
    test_codes = ['SH000300', 'SH600519', 'SZ000001']
    
    for code in test_codes:
        try:
            data = D.features([code], ["$close"], 
                           start_time="2025-01-02 09:30:00", 
                           end_time="2025-01-02 10:00:00", 
                           freq="1min")
            
            if data.empty:
                print(f"❌ {code}: 无数据")
            elif data.isnull().all().all():
                print(f"❌ {code}: 全NaN数据")
            else:
                print(f"✅ {code}: 数据正常")
                print(f"   数据形状: {data.shape}")
                print(f"   数据范围: {data.min().min():.2f} - {data.max().max():.2f}")
                print(f"   时间范围: {data.index.get_level_values('datetime').min()} 到 {data.index.get_level_values('datetime').max()}")
        except Exception as e:
            print(f"❌ {code}: 错误 - {e}")
    
    print("✅ 数据完整性检查完成\n")

# ====== 14:40预测并保存CSV ======
def predict_and_save_csv():
    """在14:40预测并保存CSV"""
    from qlib.data import D
    from qlib.workflow import R
    
    print("\n" + "="*70)
    print("🔮 14:40 预测模块启动")
    print("="*70)
    
    # 1. 获取14:40这一分钟的数据
    current_time = datetime.now()
    pred_time = current_time.strftime("%Y-%m-%d 14:40:00")
    
    print(f"⏰ 预测时间: {pred_time}")
    print("📊 读取14:40分钟数据...")
    
    # 2. 读取股票池
    stock_pool_df = pd.read_csv(Path("/home/intern0/qlib/examples/highfreq/tail_end_candidates.csv"))
    stock_codes = stock_pool_df["code"].tolist()
    print(f"📋 股票池: {len(stock_codes)} 支")
    
    # 3. 构造增量特征（只用14:40这一分钟）
    features = []
    codes_with_features = []
    
    for code in stock_codes:
        try:
            # 获取14:40这一分钟的数据
            minute_data = D.features(
                [code], 
                ["$open", "$high", "$low", "$close", "$volume"],
                start_time=pred_time,
                end_time=pred_time,
                freq="1min"
            )
            
            if minute_data.empty:
                continue
            
            # 获取前一日收盘数据
            prev_date = (datetime.strptime(pred_time.split()[0], "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            prev_day_data = D.features(
                [code],
                ["$open", "$high", "$low", "$close", "$volume"],
                start_time=f"{prev_date} 14:40:00",
                end_time=f"{prev_date} 14:40:00",
                freq="1min"
            )
            
            # 构造15个特征
            if not minute_data.empty:
                m = minute_data.iloc[0]
                if not prev_day_data.empty:
                    p = prev_day_data.iloc[0]
                else:
                    # 用当日数据近似
                    p = m
                
                # 计算VWAP
                vwap_current = (m['$open'] + 2*m['$high'] + 2*m['$low'] + m['$close']) / 6
                vwap_prev = (p['$open'] + 2*p['$high'] + 2*p['$low'] + p['$close']) / 6
                
                feature_vector = [
                    m['$open'], m['$high'], m['$low'], m['$close'], vwap_current,
                    p['$open'], p['$high'], p['$low'], p['$close'], vwap_prev,
                    m['$volume'], p['$volume'], 1.0, 0.0, 0.0
                ]
                
                features.append(feature_vector)
                codes_with_features.append(code)
        
        except Exception as e:
            print(f"⚠️ {code} 特征构造失败: {e}")
            continue
    
    print(f"✅ 特征构造完成: {len(features)} 支股票")
    
    # 4. 模型预测
    print("🔮 开始模型预测...")
    model = recorder.load_object("model")
    X = np.array(features)
    predictions = model.predict(X).flatten()
    
    print(f"📊 预测完成: {len(predictions)} 个分数")
    
    # 5. 生成CSV
    print("💾 生成CSV文件...")
    df = pd.DataFrame({
        'code': codes_with_features,
        'score': predictions
    })
    
    # 按分数排序
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    
    # 添加排名
    df['rank'] = range(1, len(df) + 1)
    
    # 添加固定字段
    df['shares'] = 200
    df['action'] = 'buy'
    df['timestamp'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # 只保存TOP 50
    df_top50 = df.head(50)
    
    # 添加价格字段（从最新数据获取）
    prices = []
    for code in df_top50['code']:
        try:
            price_data = D.features(
                [code],
                ["$close"],
                start_time=pred_time,
                end_time=pred_time,
                freq="1min"
            )
            if not price_data.empty:
                prices.append(float(price_data.iloc[0, 0]))
            else:
                prices.append(0.0)
        except:
            prices.append(0.0)
    
    df_top50['price'] = prices
    
    # 保存CSV
    csv_path = Path("daily_predictions.csv")
    df_top50.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ CSV已保存: {csv_path}")
    print(f"📊 Top 10 信号:")
    print(df_top50[['rank', 'code', 'score', 'price']].head(10).to_string(index=False))
    print("="*70 + "\n")

# ====== 定时预测任务 ======
def schedule_prediction_tasks():
    """安排定时预测任务"""
    import schedule
    import time
    
    print("\n" + "="*70)
    print("⏰ 定时预测任务调度")
    print("="*70)
    print("🔮 预测信号: 每天 14:40")
    print("💾 CSV保存: 预测完成后自动保存")
    print("="*70 + "\n")
    
    # 14:40执行预测
    schedule.every().day.at("14:40").do(predict_and_save_csv)
    
    # 检查当前时间
    current_hour = datetime.now().hour
    current_minute = datetime.now().minute
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"💡 当前时间: {current_time}")
    
    # 如果接近14:40，立即执行预测
    if current_hour == 14 and 38 <= current_minute <= 42:
        print("⏰ 检测到当前时间接近14:40，立即执行预测...")
        predict_and_save_csv()
    else:
        print("⏰ 等待定时任务触发...")
    
    # 主循环 - 持续运行
    print("\n🚀 程序开始持续运行，等待定时任务...")
    print("💡 按 Ctrl+C 停止程序")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
    except KeyboardInterrupt:
        print("\n👋 程序已停止")
        return

# 运行定时任务
print("\n" + "="*70)
print("🎯 程序运行模式选择")
print("="*70)
print("1. 只运行训练/回测，不启动定时任务")
print("2. 运行训练/回测 + 启动定时预测任务（持续运行）")
print("3. 启动实时数据更新器（独立运行）")
print("="*70)

# 可以通过环境变量控制，或者手动修改
RUN_PREDICTION_SCHEDULER = os.environ.get("RUN_PREDICTION_SCHEDULER", "False").lower() == "False"

# 先执行数据完整性检查
check_data_integrity()

if RUN_PREDICTION_SCHEDULER:
    print("🚀 启动定时预测任务模式...")
    schedule_prediction_tasks()
else:
    print("✅ 训练/回测完成，程序结束")
    print("💡 如需启动定时预测任务，请设置 RUN_PREDICTION_SCHEDULER = True")
    print("💡 如需启动实时数据更新，请运行: python realtime_data_updater.py")
