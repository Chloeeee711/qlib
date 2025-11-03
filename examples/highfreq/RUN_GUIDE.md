# 系统运行指南

## 系统架构总览

```
历史数据路径（训练用）             实时数据路径（预测用）
┌─────────────────────┐          ┌─────────────────────┐
│ kq_raw_data/        │          │ kq_raw_data_recent/ │
│ qlib_data/          │          │ qlib_data_recent/   │
└─────────────────────┘          └─────────────────────┘
        ↓                                ↓
   训练模型                          实时预测
```

## 运行流程

### 阶段1：模型训练（一次性，只需运行一次）

**前提条件**：确保历史数据已下载到 `kq_raw_data/` 和 `qlib_data/`

**运行命令**：
```bash
# 设置环境变量，禁用预测调度（只训练不回测）
set RUN_PREDICTION_SCHEDULER=False
python highfreq_workflow.py
```

**或者修改代码**：
```python
# 在 highfreq_workflow.py 最后，将 RUN_PREDICTION_SCHEDULER 设为 False
RUN_PREDICTION_SCHEDULER = False
```

**输出**：
- 训练好的模型（保存在mlflow记录器中）
- 回测报告

**注意**：训练完成后，**不要**使用 `highfreq_workflow.py` 来做日常预测，而是使用 `prediction_scheduler.py`

---

### 阶段2：日常运行（需要长期运行的服务）

需要同时运行3个后台进程：

#### 1️⃣ 实时数据更新器（必需）
```bash
python tqsdk_realtime_updater.py
```
**功能**：
- 交易时间内每分钟下载最新数据
- 自动追加到 `qlib_data_recent/`，不覆盖历史数据
- 需要确保交易时间内一直运行

**启动时间**：建议在交易日 **09:25** 之前启动

---

#### 2️⃣ 预测调度器（必需）
```bash
python prediction_scheduler.py
```
**功能**：
- 每天 **14:40** 自动执行预测
- 使用实时数据路径 `qlib_data_recent/`
- 加载已训练好的模型
- 生成 `daily_predictions.csv`

**启动时间**：建议在交易日 **14:30** 之前启动（或一直运行）

**测试模式**（收盘后测试）：
```bash
python prediction_scheduler.py --test
```

---

#### 3️⃣ 邮件发送器（必需）
```bash
python email_sender.py
```
**功能**：
- 每天 **14:45** 发送买入邮件
- 每天 **10:46** 发送卖出邮件（带收益计算）
- 自动读取 `daily_predictions.csv` 和 `positions.json`

**启动时间**：建议一直运行（或至少在 14:40 和 10:46 之前）

---

## 运行方案对比

### ❌ 不推荐：使用 `highfreq_workflow.py` 做日常预测
**原因**：
- `highfreq_workflow.py` 使用历史数据路径，不适合实时预测
- 包含大量训练和回测代码，启动较慢
- 架构设计上，历史数据路径和实时数据路径是分离的

### ✅ 推荐：使用 `prediction_scheduler.py` 做日常预测
**原因**：
- 专门为实时预测设计
- 使用正确的实时数据路径
- 代码简洁，启动快
- 支持测试模式

---

## 完整运行示例

### Windows PowerShell（3个独立终端窗口）

**终端1 - 实时数据更新器**：
```powershell
cd C:\Users\ASUS\qlib\examples\highfreq
python tqsdk_realtime_updater.py
```

**终端2 - 预测调度器**：
```powershell
cd C:\Users\ASUS\qlib\examples\highfreq
python prediction_scheduler.py
```

**终端3 - 邮件发送器**：
```powershell
cd C:\Users\ASUS\qlib\examples\highfreq
python email_sender.py
```

### 或者使用后台运行（Windows）
```powershell
# 启动后台进程
Start-Process python -ArgumentList "tqsdk_realtime_updater.py" -WindowStyle Minimized
Start-Process python -ArgumentList "prediction_scheduler.py" -WindowStyle Minimized
Start-Process python -ArgumentList "email_sender.py" -WindowStyle Minimized
```

---

## 时间线

### 交易日时间表

```
09:25 启动 tqsdk_realtime_updater.py（开始下载实时数据）
       ↓
10:46 email_sender.py 发送卖出邮件（前一天买入的股票）
       ↓
14:40 prediction_scheduler.py 执行预测，生成 CSV
       ↓
14:45 email_sender.py 发送买入邮件，保存持仓信息
       ↓
15:00 收盘后，所有服务继续运行（等待下一个交易日）
```

---

## 常见问题

### Q: 能否只运行一个脚本？
A: 不行。3个服务职责不同，需要同时运行：
- `tqsdk_realtime_updater.py` - 数据更新
- `prediction_scheduler.py` - 预测
- `email_sender.py` - 邮件

### Q: 什么时候运行 `highfreq_workflow.py`？
A: 只在首次训练或重新训练模型时运行一次。日常预测不需要它。

### Q: 如何测试系统？
A: 收盘后测试：
```bash
python prediction_scheduler.py --test    # 生成模拟预测
python email_sender.py test            # 测试买入邮件
python email_sender.py test_sell        # 测试卖出邮件（需要positions.json）
```

### Q: 如果某个服务崩溃了怎么办？
A: 建议使用进程管理工具（如 `supervisor` 或 Windows 任务计划程序）自动重启服务。

---

## 总结

✅ **你的理解基本正确**，但有一个重要更正：

- ✅ `tqsdk_realtime_updater.py` - 一直运行 ✓
- ✅ `email_sender.py` - 一直运行 ✓
- ⚠️ **使用 `prediction_scheduler.py` 而不是 `highfreq_workflow.py` 做日常预测**
- ✅ 前提：已有训练好的模型 ✓

**`highfreq_workflow.py` 的用途**：
- 仅用于：历史数据下载、模型训练、回测
- 不用于：日常实时预测（那是 `prediction_scheduler.py` 的职责）

