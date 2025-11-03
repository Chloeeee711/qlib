# 高频交易系统

## 系统架构

本系统分为三个独立模块：

1. **训练/回测模块** (`highfreq_workflow.py`)
   - 数据预处理和转换
   - 模型训练
   - 回测分析
   - 可选定时预测任务

2. **实时数据更新器** (`realtime_data_updater.py`)
   - 在交易时间内实时更新数据
   - 不覆盖历史数据，按时间顺序追加
   - 独立运行，不影响其他模块

3. **独立预测调度器** (`prediction_scheduler.py`)
   - 定时预测并保存CSV
   - 独立运行，不依赖训练模块

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 使用启动器（推荐）
```bash
python start_system.py
```

### 3. 手动启动各模块

#### 训练/回测 + 定时预测
```bash
python highfreq_workflow.py
```

#### 只运行训练/回测（无定时任务）
```bash
set RUN_PREDICTION_SCHEDULER=False
python highfreq_workflow.py
```

#### 启动实时数据更新器

**方案1：TQSDK实时更新器（推荐）**
```bash
python tqsdk_realtime_updater.py
```
- ✅ 稳定可靠，使用tqsdk直接订阅数据
- ✅ 不会覆盖历史数据
- ✅ 每1分钟自动更新
- ⚠️ 需要安装tqsdk：`pip install tqsdk`

**方案2：KQdownloader更新器**
```bash
python realtime_data_updater.py
```
- ⚠️ 可能有dump_bin.py兼容性问题
- ⚠️ 建议谨慎使用

#### 启动独立预测调度器
```bash
python prediction_scheduler.py
```

#### 测试实时数据更新器
```bash
python test_realtime_updater.py
```

## 系统特点

### ✅ 解决的问题
1. **数据污染问题**：实时更新不再覆盖历史数据，使用真正的增量更新
2. **信号异常问题**：增加了详细的调试信息
3. **模块耦合问题**：各模块独立运行，互不影响
4. **频繁更新问题**：改为每5分钟更新一次，避免过于频繁
5. **数据安全问题**：增加安全检查机制，防止覆盖有效数据

### 🔧 调试信息
- 数据完整性检查
- 训练数据质量检查
- 模型预测验证
- 信号生成统计

### 📊 输出文件
- `daily_predictions.csv`：每日预测结果
- `filtered_market.pkl`：筛选后的股票池
- 回测报告和图表

## 运行模式

### 模式1：完整系统
1. 先运行 `python highfreq_workflow.py` 进行训练和回测
2. 再运行 `python realtime_data_updater.py` 进行实时数据更新
3. 最后运行 `python prediction_scheduler.py` 进行定时预测

### 模式2：独立运行
- 每个模块都可以独立运行
- 实时数据更新器在交易时间内自动更新
- 预测调度器每天14:40自动预测

## 注意事项

1. **数据路径**：确保 `qlib_data_recent` 目录存在且包含正确数据
2. **交易时间**：实时更新只在交易时间内进行（9:30-11:30, 13:00-15:00）
3. **预测时间**：定时预测在每天14:40执行
4. **网络连接**：确保能正常访问天勤数据源

## 故障排除

### 问题1：所有预测值都相同
- 检查数据是否包含过多NaN值
- 检查模型是否正确训练
- 查看调试信息中的警告

### 问题2：数据更新失败
- 检查网络连接
- 检查天勤账号信息
- 查看错误日志

### 问题3：预测失败
- 确保模型已训练
- 检查数据时间范围
- 查看详细错误信息

### 问题4：实时更新覆盖数据
- 运行 `python test_realtime_updater.py` 检查数据完整性
- 确保主数据目录存在且有有效数据
- 检查临时目录是否正确创建

### 问题5：文件变为0KB
- 停止实时更新器
- 重新下载历史数据
- 使用测试脚本验证数据完整性