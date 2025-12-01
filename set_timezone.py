#!/usr/bin/env python3
"""
设置时区为北京时间（UTC+8）
在 Python 脚本开头导入此模块即可：
    import set_timezone
"""
import os
import time

# 设置时区环境变量
os.environ['TZ'] = 'Asia/Shanghai'

try:
    time.tzset()  # Unix/Linux 系统
except AttributeError:
    # Windows 系统不支持 tzset
    pass

# 验证设置
from datetime import datetime
import pytz

# 创建北京时区对象
beijing_tz = pytz.timezone('Asia/Shanghai')

def get_beijing_time():
    """获取北京时间"""
    return datetime.now(beijing_tz)

if __name__ == '__main__':
    print(f"时区环境变量: {os.environ.get('TZ', '未设置')}")
    print(f"当前时间: {datetime.now()}")
    print(f"北京时间: {get_beijing_time()}")

