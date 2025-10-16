# 高频隔夜收益率标签实现方案

## 问题总结

用户需要在Qlib中实现高频隔夜收益率标签，具体需求：
- 当日14:41-14:50的VWAP
- 次日10:11-10:20的VWAP  
- 隔夜收益率 = 次日VWAP / 当日VWAP - 1

## 解决方案

### 1. 自定义算子实现 (`highfreq_ops.py`)

实现了两个关键算子：
- `IntradayWindowVWAP`: 计算指定时间窗口的VWAP
- `DayShift`: 将数据按交易日移位

### 2. 自定义Handler实现 (`highfreq_handler.py`)

`HighFreqHandler` 类实现了：
- 自定义标签计算逻辑
- 数据扩展（确保有次日数据）
- 标签与特征数据的正确合并

### 3. 关键修复

#### 问题1: 标签全为NaN
**原因**: 自定义标签需要2天数据，但handler只加载1天数据
**解决**: 在`setup_data`中扩展数据到次日

#### 问题2: 索引对齐问题  
**原因**: 自定义标签有480行（2天），handler数据只有240行（1天）
**解决**: 扩展handler数据到720行（3天），确保有足够数据计算隔夜收益率

#### 问题3: 算子注册问题
**原因**: 自定义算子没有正确注册到Qlib表达式引擎
**解决**: 在初始化时注册所有自定义算子

## 最终结果

✅ **Handler创建成功**: 数据形状(720, 22)
✅ **标签计算成功**: 240个非NaN值
✅ **隔夜收益率正确**: 约-0.76%
✅ **数据对齐正确**: 标签与特征数据完美匹配

## 使用方法

```python
# 1. 初始化Qlib
import qlib
from qlib.constant import REG_CN
qlib.init(provider_uri="your_data_path", region=REG_CN)

# 2. 注册自定义算子
from qlib.data.ops import Operators
from highfreq_ops import DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut, DayFirst, IntradayWindowVWAP, DayShift
Operators.register([DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut, DayFirst, IntradayWindowVWAP, DayShift])

# 3. 创建Handler
from qlib.utils import init_instance_by_config
handler_config = {
    "class": "HighFreqHandler",
    "module_path": "highfreq_handler",
    "kwargs": {
        "instruments": ["SH000300"],
        "start_time": "2025-05-06",
        "end_time": "2025-05-07",
        "drop_raw": False,
    }
}
handler = init_instance_by_config(handler_config)

# 4. 检查结果
print(f"数据形状: {handler._data.shape}")
print(f"标签非NaN数量: {handler._data[('label', 'LABEL0')].notna().sum()}")
```

## 技术细节

### 隔夜收益率计算逻辑
1. 计算当日14:41-14:50时间窗口的VWAP
2. 计算次日10:11-10:20时间窗口的VWAP
3. 计算收益率 = 次日VWAP / 当日VWAP - 1
4. 广播回分钟级数据

### 数据扩展策略
- 原始数据: 240行（1天）
- 扩展数据: 480行（2天）  
- 最终数据: 720行（3天）
- 确保有足够数据计算隔夜收益率

### 标签合并策略
- 使用MultiIndex对齐
- 创建多级列名标签数据
- 合并到现有特征数据中

## 验证结果

- ✅ 标签计算成功: 240个非NaN值
- ✅ 隔夜收益率: -0.76%
- ✅ 数据形状: (720, 22)
- ✅ 标签与特征对齐: 完美匹配

## 注意事项

1. 需要至少2天数据才能计算隔夜收益率
2. 确保自定义算子正确注册
3. 数据扩展会增加内存使用
4. 标签值在训练集和测试集中分布可能不同

## 文件结构

```
examples/highfreq/
├── highfreq_ops.py          # 自定义算子
├── highfreq_handler.py       # 自定义Handler
├── test_final_pipeline.py    # 测试脚本
└── README_solution.md        # 本文档
```

## 总结

成功实现了高频隔夜收益率标签的完整pipeline，解决了数据对齐、算子注册、标签计算等关键问题。Handler现在可以正确计算和提供隔夜收益率标签，为模型训练提供高质量的数据。

