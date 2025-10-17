#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import qlib
from qlib.constant import REG_CN
from qlib.data import D
import pandas as pd
import numpy as np

# 设置数据路径
KQ_DATA_DIR = r"C:\Users\ASUS\qlib_data"

# 初始化 Qlib
print("🔧 初始化 Qlib...")
qlib.init(provider_uri=KQ_DATA_DIR, region=REG_CN)
print("✅ Qlib 初始化完成")

# 测试自定义标签计算逻辑
print("🔧 测试自定义标签计算逻辑...")

def test_custom_label_calculation():
    """测试自定义标签计算逻辑"""
    inst = "SH000300"
    start_time = "2025-05-06"
    end_time = "2025-05-07"
    
    print(f"📊 测试股票: {inst}")
    print(f"📊 时间范围: {start_time} 到 {end_time}")
    
    # 拉取分钟数据
    print("📊 拉取分钟数据...")
    try:
        df = D.features([inst], ["$open", "$high", "$low", "$close", "$volume"],
                        start_time=start_time, end_time=end_time, freq="1min")
        print(f"✅ 数据拉取成功，形状: {df.shape}")
        print(f"📊 数据索引类型: {type(df.index)}")
        print(f"📊 数据列名: {df.columns.tolist()}")
    except Exception as e:
        print(f"❌ 数据拉取失败: {e}")
        return
    
    # MultiIndex 安全切片
    if isinstance(df.index, pd.MultiIndex):
        lvl = "instrument" if "instrument" in df.index.names else df.index.names[0]
        inst_levels = df.index.get_level_values(lvl).unique()
        print(f"📊 可用标的: {inst_levels.tolist()}")
        if inst not in inst_levels:
            print(f"❌ 标的 {inst} 不在数据中")
            return
        df = df.xs(inst, level=lvl, drop_level=True)
        print(f"✅ 切片成功，形状: {df.shape}")
    
    if df.empty:
        print("❌ 切片后数据为空")
        return
    
    print(f"📊 数据时间范围: {df.index.min()} 到 {df.index.max()}")
    
    # 时间窗口掩码
    dt = pd.to_datetime(df.index)
    t_int = dt.hour * 100 + dt.minute
    mask_today = (t_int >= int(14.41 * 100)) & (t_int <= int(14.50 * 100))
    mask_next  = (t_int >= int(10.11 * 100)) & (t_int <= int(10.20 * 100))
    
    print(f"📊 14:41-14:50 时间段数据点数量: {mask_today.sum()}")
    print(f"📊 10:11-10:20 时间段数据点数量: {mask_next.sum()}")
    
    if mask_today.sum() == 0:
        print("❌ 14:41-14:50 时间段没有数据")
        return
    
    if mask_next.sum() == 0:
        print("❌ 10:11-10:20 时间段没有数据")
        return
    
    # 典型价格与成交量
    tp  = (df["$open"] + df["$high"] + df["$low"] + df["$close"]) / 4.0
    vol = df["$volume"]
    
    print(f"📊 典型价格范围: {tp.min():.2f} 到 {tp.max():.2f}")
    print(f"📊 成交量范围: {vol.min():.0f} 到 {vol.max():.0f}")
    
    # 按"日期"聚合成日级 VWAP
    day_key = pd.Index(dt.date)
    print(f"📊 交易日数量: {len(day_key.unique())}")
    print(f"📊 交易日: {day_key.unique()}")
    
    def window_vwap(mask):
        num = (tp.where(mask, 0.0) * vol.where(mask, 0.0)).groupby(day_key).sum()
        den = vol.where(mask, 0.0).groupby(day_key).sum()
        wv  = (num / den.replace(0.0, np.nan)).astype("float64")
        wv = wv.fillna(tp.where(mask).groupby(day_key).mean())
        wv = wv.fillna(df["$close"].groupby(day_key).last())
        return wv

    print("📊 计算当日VWAP...")
    today_day = window_vwap(mask_today)
    print(f"📊 当日VWAP: {today_day}")
    
    print("📊 计算次日VWAP...")
    next_day  = window_vwap(mask_next).shift(1)
    print(f"📊 次日VWAP: {next_day}")
    
    # 广播回分钟
    today_map = today_day.to_dict()
    next_map  = next_day.to_dict()
    days = pd.Index(dt.date)
    today_series = pd.Series([today_map.get(d, np.nan) for d in days], index=df.index, dtype="float64")
    next_series  = pd.Series([next_map.get(d, np.nan)  for d in days], index=df.index, dtype="float64")
    
    print(f"📊 当日VWAP系列非NaN数量: {today_series.notna().sum()}")
    print(f"📊 次日VWAP系列非NaN数量: {next_series.notna().sum()}")
    
    lbl = next_series / today_series - 1
    print(f"📊 标签非NaN数量: {lbl.notna().sum()}")
    
    if lbl.notna().sum() > 0:
        print("✅ 标签计算成功！")
        print(f"📊 标签统计: {lbl.describe()}")
        print(f"📊 标签示例值:")
        print(lbl.dropna().head())
    else:
        print("❌ 标签全为NaN")
        print(f"📊 当日VWAP系列: {today_series.head()}")
        print(f"📊 次日VWAP系列: {next_series.head()}")

if __name__ == "__main__":
    test_custom_label_calculation()
    print("\n🎉 调试完成！")




