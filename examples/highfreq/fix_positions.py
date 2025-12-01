#!/usr/bin/env python3
"""
修复positions.json：从CSV读取股票列表，获取11月28日14:40的价格，重新生成持仓
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# 添加路径
PROJECT_ROOT = "/home/intern0/qlib"
WORKDIR = "/home/intern0/qlib/examples/highfreq"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, WORKDIR)

from qlib.data import D

def get_price_at_time(code, target_date, target_time_str):
    """获取指定时间点的价格"""
    try:
        # 使用QLib获取价格
        ts_str = f"{target_date} {target_time_str}"
        close_df = D.features(
            [code],
            ["$close"],
            start_time=ts_str,
            end_time=ts_str,
            freq="1min",
        )
        if close_df is not None and not close_df.empty:
            col = close_df.columns[0]
            reset_df = close_df.reset_index()
            reset_df = reset_df.dropna(subset=[col])
            if not reset_df.empty:
                price = float(reset_df.iloc[-1][col])
                if price > 0:
                    return price
        
        # 如果QLib没有，尝试从CSV读取
        features_dir = Path("/home/intern0/qlib_data_recent") / "features"
        csv_path = features_dir / code / 'data.csv'
        if csv_path.exists():
            df_stock = pd.read_csv(csv_path)
            df_stock['datetime'] = pd.to_datetime(df_stock['datetime'], errors='coerce')
            df_stock = df_stock.dropna(subset=['datetime'])
            
            target_dt = pd.to_datetime(f"{target_date} {target_time_str}")
            exact_data = df_stock[df_stock['datetime'] == target_dt]
            if not exact_data.empty:
                price = float(exact_data.iloc[0]['close'])
                if price > 0:
                    return price
            
            # 如果精确时间没有，找同一天<=14:40的最后一条
            same_day = df_stock[df_stock['datetime'].dt.date == pd.to_datetime(target_date).date()]
            if not same_day.empty:
                target_minutes = 14 * 60 + 40
                before_target = same_day[same_day['datetime'].dt.hour * 60 + same_day['datetime'].dt.minute <= target_minutes]
                if not before_target.empty:
                    price = float(before_target.iloc[-1]['close'])
                    if price > 0:
                        return price
    except Exception as e:
        print(f"[WARN] {code} 获取价格失败: {e}")
    return None

def fix_positions():
    """修复positions.json"""
    workdir = Path(WORKDIR)
    csv_path = workdir / "daily_predictions.csv"
    positions_file = workdir / "positions.json"
    
    if not csv_path.exists():
        print(f"❌ 未找到买入预测CSV文件: {csv_path}")
        return False
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    if df.empty:
        print("❌ CSV文件为空")
        return False
    
    # 只处理买入信号
    buy_df = df[df['action'] == 'buy'].copy() if 'action' in df.columns else df.copy()
    if buy_df.empty:
        print("❌ 没有买入信号")
        return False
    
    print(f"📊 从CSV读取到 {len(buy_df)} 个买入信号")
    print(f"📅 使用买入日期: 2025-11-28 (最近的交易日)")
    
    # 初始化QLib（如果需要）
    try:
        import qlib
        from qlib.constant import REG_CN
        from highfreq_ops import DayLast, DayFirst, FFillNan, BFillNan, Date, Select, IsNull, DayShift, Cut, IntradayWindowVWAP, If, Gt, Lt
        SPEC_CONF = {"custom_ops": [DayLast, DayFirst, FFillNan, BFillNan, Date, Select, IsNull, DayShift, Cut, IntradayWindowVWAP, If, Gt, Lt], "expression_cache": None}
        CUSTOM_QLIB_CONFIG = {
            "provider_uri": "/home/intern0/qlib_data_recent",
            "dataset_cache": None,
            "expression_cache": "DiskExpressionCache",
            "region": REG_CN,
            **SPEC_CONF
        }
        qlib.init(**CUSTOM_QLIB_CONFIG, joblib_backend="threading")
        print("✅ QLib初始化成功")
    except Exception as e:
        print(f"⚠️ QLib初始化失败，将使用CSV文件: {e}")
    
    # 重新生成持仓
    positions = {}
    buy_date = "2025-11-28"
    target_time = "14:40:00"
    
    print(f"\n📊 开始获取 {buy_date} {target_time} 的价格...")
    success_count = 0
    fail_count = 0
    
    for idx, row in buy_df.iterrows():
        code = row['code']
        score = float(row.get('score', 0.0))
        rank = int(row.get('rank', idx + 1))
        shares = int(row.get('shares', 200))
        
        # 获取价格
        price = get_price_at_time(code, buy_date, target_time)
        
        if price is None or price <= 0:
            print(f"⚠️ {code} 无法获取价格，跳过")
            fail_count += 1
            continue
        
        positions[code] = {
            'buy_date': buy_date,
            'buy_price': price,
            'shares': shares,
            'code': code,
            'rank': rank,
            'score': score
        }
        success_count += 1
        if success_count % 10 == 0:
            print(f"   ✅ 已处理 {success_count} 个股票...")
    
    if len(positions) == 0:
        print("❌ 没有成功获取任何价格，无法生成持仓")
        return False
    
    # 保存positions.json
    with open(positions_file, 'w', encoding='utf-8') as f:
        json.dump(positions, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 持仓修复完成！")
    print(f"   📊 成功: {success_count} 个股票")
    print(f"   ⚠️ 失败: {fail_count} 个股票")
    print(f"   💾 已保存到: {positions_file}")
    print(f"\n📋 前5个持仓:")
    for i, (code, pos) in enumerate(list(positions.items())[:5]):
        print(f"   {i+1}. {code}: 买入价格 {pos['buy_price']:.2f}, 买入日期 {pos['buy_date']}")
    
    return True

if __name__ == "__main__":
    print("="*70)
    print("修复positions.json - 从CSV恢复11月28日的持仓")
    print("="*70)
    success = fix_positions()
    if success:
        print("\n✅ 修复成功！现在可以发送12-01的卖出邮件了")
    else:
        print("\n❌ 修复失败，请检查错误信息")
    sys.exit(0 if success else 1)

