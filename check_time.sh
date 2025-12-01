#!/bin/bash
# 查看时间和时区信息

echo "════════════════════════════════════════"
echo "🕐 时间和时区信息"
echo "════════════════════════════════════════"
echo ""

echo "【方法 1】基本时间信息"
date
echo ""

echo "【方法 2】格式化时间"
date +"%Y-%m-%d %H:%M:%S %Z"
echo ""

echo "【方法 3】本地时间 vs UTC 时间"
echo "  本地时间: $(date)"
echo "  UTC 时间: $(date -u)"
echo ""

echo "【方法 4】时区环境变量"
echo "  TZ = ${TZ:-未设置}"
echo ""

echo "【方法 5】完整中文格式"
date "+📅 日期: %Y年%m月%d日 | ⏰ 时间: %H:%M:%S | 🌍 时区: %Z | 📆 星期: %A"
echo ""

echo "【方法 6】Python 时间信息"
python3 << 'PYEOF'
import os
import time
from datetime import datetime

tz = os.environ.get('TZ', '未设置')
print(f"  时区环境变量: {tz}")

try:
    time.tzset()
except:
    pass

now = datetime.now()
utc_now = datetime.utcnow()
print(f"  当前时间: {now}")
print(f"  UTC 时间: {utc_now}")

# 计算时差
if tz != '未设置':
    try:
        import pytz
        beijing = pytz.timezone('Asia/Shanghai')
        beijing_time = datetime.now(beijing)
        print(f"  北京时间: {beijing_time}")
    except:
        print(f"  (需要 pytz 库显示精确的北京时间)")
PYEOF

echo ""
echo "════════════════════════════════════════"

