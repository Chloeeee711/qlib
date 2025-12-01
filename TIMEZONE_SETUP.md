# 时区设置说明（用户级，无需 sudo）

## ✅ 已完成的设置

### 1. Shell 环境时区设置
- ✅ `~/.bashrc` - 已添加 `export TZ="Asia/Shanghai"`
- ✅ `~/.profile` - 已添加 `export TZ="Asia/Shanghai"`

**生效方式：**
- 新打开的终端会自动使用北京时间
- 当前终端需要执行：`source ~/.bashrc` 或重新打开终端

### 2. Python 时区设置脚本
- ✅ `/home/intern0/qlib/set_timezone.py` - Python 时区初始化脚本

**使用方法：**
```python
# 在 Python 脚本开头导入
import set_timezone

from datetime import datetime
print(datetime.now())  # 会使用北京时间
```

或者直接在脚本中设置：
```python
import os
import time

os.environ['TZ'] = 'Asia/Shanghai'
try:
    time.tzset()  # Unix/Linux
except AttributeError:
    pass  # Windows 不支持

from datetime import datetime
print(datetime.now())
```

## 🔍 验证时区设置

### Shell 验证
```bash
echo $TZ
date
```

### Python 验证
```python
import os
import time
os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()

from datetime import datetime
print(datetime.now())
print(f"时区: {os.environ.get('TZ')}")
```

## ⚠️ 注意事项

1. **系统时间本身可能不准确**：如果系统时间本身就不对，时区设置只能改变显示方式，不能修正系统时间。需要联系管理员同步系统时间。

2. **Python 脚本中需要显式设置**：虽然 shell 环境变量已设置，但某些 Python 程序（特别是通过 cron 或 systemd 启动的）可能不会继承这些环境变量，建议在脚本中显式设置。

3. **持久化**：这些设置会在每次登录时自动加载，是永久性的（只要不删除这些配置文件）。

## 📝 如果需要修改

编辑配置文件：
```bash
# 编辑 .bashrc
nano ~/.bashrc

# 编辑 .profile
nano ~/.profile
```

修改后重新加载：
```bash
source ~/.bashrc
source ~/.profile
```

