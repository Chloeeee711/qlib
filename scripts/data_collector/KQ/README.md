# 天勤数据收集器

## 简介

天勤数据收集器用于从天勤量化平台下载中国股票的历史数据，并转换为Qlib标准格式。

## 功能特性

- ✅ 支持从CSV股票池文件读取股票列表
- ✅ 支持1分钟、5分钟、日线数据下载
- ✅ 自动数据格式转换（天勤格式 → Qlib格式）
- ✅ 支持2024-2025年历史数据
- ✅ 多线程并发下载
- ✅ 自动重试机制
- ✅ 数据验证和清洗

## 安装依赖

```bash
pip install tqsdk pandas numpy loguru fire
```

## 使用方法

### 1. 基本使用

```bash
# 下载默认股票池数据
python scripts/data_collector/KQ/KQdownloader.py download_data \
    --source_dir ~/kq_data \
    --start 2024-01-01 \
    --end 2025-12-31 \
    --interval 1min \
    --delay 1
```

### 2. 使用CSV股票池

```bash
# 从CSV文件读取股票池
python scripts/data_collector/KQ/KQdownloaderor.py download_data \
    --source_dir ~/kq_data \
    --start 2024-01-01 \
    --end 2025-12-31 \
    --interval 1min \
    --csv_stock_pool sorted_high_preclose_ratio_2025.csv \
    --delay 1
```

### 3. 使用天勤账户

```bash
# 使用天勤账户（可选）
python scripts/data_collector/KQ/KQdownloader.py download_data \
    --source_dir ~/kq_data \
    --start 2024-01-01 \
    --end 2025-12-31 \
    --interval 1min \
    --username your_username \
    --password your_password \
    --csv_stock_pool sorted_high_preclose_ratio_2025.csv
```

### 4. 完整流程（下载+标准化）

```bash
# 运行完整流程
python scripts/data_collector/KQ/KQdownloader.py run \
    --source_dir ~/kq_data \
    --target_dir ~/.qlib/qlib_data/cn_data_1min \
    --start 2024-01-01 \
    --end 2025-12-31 \
    --interval 1min \
    --csv_stock_pool sorted_high_preclose_ratio_2025.csv \
    --max_workers 4 \
    --delay 1
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `source_dir` | 原始数据保存目录 | - |
| `target_dir` | Qlib格式数据目录 | - |
| `start` | 开始日期 | 2024-01-01 |
| `end` | 结束日期 | 2025-12-31 |
| `interval` | 数据频率 | 1min |
| `max_workers` | 最大工作线程数 | 4 |
| `delay` | 请求间隔（秒） | 1 |
| `csv_stock_pool` | CSV股票池文件路径 | - |
| `username` | 天勤用户名 | - |
| `password` | 天勤密码 | - |

## 数据格式

### 输入格式（天勤）
- 股票代码：000001.SZ, 600000.SH
- 时间格式：datetime
- 价格字段：open, high, low, close, volume

### 输出格式（Qlib）
- 二进制格式：.bin文件
- 目录结构：
  ```
  qlib_data/
  ├── calendars/     # 交易日历
  ├── features/      # 特征数据
  └── instruments/   # 股票列表
  ```

## 注意事项

1. **天勤SDK限制**：免费账户有数据访问限制
2. **网络延迟**：建议设置适当的delay参数
3. **数据质量**：自动过滤成交量为0的数据
4. **错误处理**：支持自动重试和错误记录

## 故障排除

### 1. 天勤SDK未安装
```bash
pip install tqsdk
```

### 2. 网络连接问题
- 检查网络连接
- 增加delay参数
- 使用代理（如需要）

### 3. 数据为空
- 检查股票代码格式
- 确认时间范围
- 验证天勤账户权限

## 示例

```python
from scripts.data_collector.KQ.KQcollector import KQRun

# 创建收集器
collector = KQRun(
    source_dir="~/kq_data",
    target_dir="~/.qlib/qlib_data/cn_data_1min",
    csv_stock_pool="sorted_high_preclose_ratio_2025.csv",
    interval="1min",
    max_workers=4,
    delay=1
)

# 运行完整流程
collector.run(start="2024-01-01", end="2025-12-31")
```

























