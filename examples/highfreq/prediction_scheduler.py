
# ============================================
# 独立预测和CSV保存脚本
# 功能：定时预测并保存CSV文件
# ============================================

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import schedule
from datetime import datetime, timedelta
import pickle

# 添加路径
sys.path.append(r"C:\Users\ASUS\qlib\examples\highfreq")

def is_trading_day(date_obj=None):
    """判断是否为交易日（简单实现：排除周末）"""
    if date_obj is None:
        date_obj = datetime.now()
    weekday = date_obj.weekday()  # 0=Monday, 6=Sunday
    return weekday < 5  # 周一到周五

class PredictionScheduler:
    def __init__(self, test_mode=False):
        # 配置 - 使用实时数据路径
        self.KQ_DATA_DIR = r"C:\Users\ASUS\qlib_data_recent"
        self.STOCK_POOL_CSV = r"C:\Users\ASUS\qlib\examples\highfreq\sorted_high_preclose_ratio_2025.csv"
        # 测试模式标志
        self.test_mode = test_mode
        print(f"[INIT] 预测调度器初始化完成")
        print(f"[INFO] 使用实时数据路径: {self.KQ_DATA_DIR}")
        print(f"[INFO] 测试模式: {test_mode}")
        # 非测试模式只加载模型，不加载Dataset
        if not test_mode:
            self._init_qlib()
            self._load_model_only()

    def _init_qlib(self):
        """初始化qlib（仅用于Recorder加载模型，不读取特征数据）"""
        import qlib
        from qlib.constant import REG_CN
        # 自定义操作符配置
        from highfreq_ops import DayLast, DayFirst, FFillNan, BFillNan, Date, Select, IsNull, DayShift, Cut, IntradayWindowVWAP, If, Gt, Lt
        SPEC_CONF = {"custom_ops": [DayLast, DayFirst, FFillNan, BFillNan, Date, Select, IsNull, DayShift, Cut, IntradayWindowVWAP, If, Gt, Lt], "expression_cache": None}
        CUSTOM_QLIB_CONFIG = {
            "provider_uri": self.KQ_DATA_DIR,
            "dataset_cache": None,
            "expression_cache": "DiskExpressionCache",
            "region": REG_CN,
            **SPEC_CONF
        }
        try:
            qlib.init(**CUSTOM_QLIB_CONFIG, joblib_backend="threading")
            print(f"✅ Qlib 已初始化（用于加载模型），数据路径: {self.KQ_DATA_DIR}")
        except Exception as e:
            print(f"⚠️ 自定义路径初始化失败: {e}")
            qlib.init(region=REG_CN)
            print(f"✅ 使用默认配置初始化成功")

    def _load_model_only(self):
        """只从Recorder加载已训练模型，不配置Dataset，避免全量数据加载"""
        from qlib.workflow import R
        try:
            recorder = R.get_recorder()
            self.model = recorder.load_object("model")
            print(f"✅ 模型加载成功: {type(self.model).__name__}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def _load_normalization_params(self):
        """从训练 recorder 读取标准化参数。若未保存，直接报错，不重算、不兜底。"""
        try:
            from qlib.workflow import R
            rec = R.get_recorder()
            # 常见保存名尝试
            for key in ("norm_params", "highfreq_norm", "processor_state", "highfreq_norm_params"):
                try:
                    obj = rec.load_object(key)
                    if isinstance(obj, dict) and "price" in obj and "volume" in obj:
                        print(f"✅ 从recorder加载标准化参数: {key}")
                        return obj
                except Exception:
                    continue
            print("❌ 未在recorder中找到标准化参数(norm_params)。")
            print("   必须使用训练时保存的参数，当前不支持重算或兜底。")
            return None
        except Exception as e:
            print(f"❌ 读取recorder标准化参数失败: {e}")
            return None
    
    def _normalize_features(self, features, norm_params):
        """对特征应用标准化（与 HighFreqNorm 处理器一致）"""
        # 特征顺序：open, high, low, close, vwap_current, prev_open, prev_high, prev_low, prev_close, prev_vwap, volume, prev_volume, 1.0, 0.0, 0.0
        # 价格类：索引 0-9 (前5个是当前，后5个是前一日)
        # 成交量类：索引 10-11
        # 固定值：索引 12-14 (不需要标准化)
        
        X = np.array(features, dtype=np.float32)
        
        # 价格类标准化： (val - med) / mad，然后clip到[-8, 8]
        price_med = norm_params["price"]["med"]
        price_mad = norm_params["price"]["mad"]
        for i in range(10):  # 索引0-9是价格类
            X[:, i] = (X[:, i] - price_med) / price_mad
        X[:, :10] = np.clip(X[:, :10], -8.0, 8.0)
        
        # 成交量类标准化：log1p后 (val - med) / mad，然后clip到[-8, 8]
        volume_med = norm_params["volume"]["med"]
        volume_mad = norm_params["volume"]["mad"]
        for i in range(10, 12):  # 索引10-11是成交量
            X[:, i] = np.log1p(X[:, i])
            X[:, i] = (X[:, i] - volume_med) / volume_mad
        X[:, 10:12] = np.clip(X[:, 10:12], -8.0, 8.0)
        
        return X
    
    def _get_stock_codes(self):
        """获取股票代码"""
        try:
            # 尝试加载筛选后的股票代码
            with open("filtered_market.pkl", "rb") as f:
                stock_codes = pickle.load(f)
            print(f"✅ 加载筛选后股票代码: {len(stock_codes)} 支")
            return stock_codes
        except:
            # 备用方案：从CSV读取
            stock_pool_df = pd.read_csv(self.STOCK_POOL_CSV)
            stock_codes = stock_pool_df["code"].tolist()
            print(f"✅ 从CSV加载股票代码: {len(stock_codes)} 支")
            return stock_codes
    
    def predict_and_save_csv(self):
        import glob
        print(f"\n🔮 开始预测: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            # 1. 获取当前时间
            current_time = datetime.now()
            # 统一使用 "YYYY-MM-DD HH:MM"（无秒）做比较，避免格式不一致
            pred_dt = current_time.strftime("%Y-%m-%d 14:40")

            # 回溯，取最近有数据的前一天（保留原逻辑）
            def get_prev_trading_dt(df, curr_dt_min):
                base_date = datetime.strptime(curr_dt_min[:10], "%Y-%m-%d")
                for d in range(1, 8):  # 最多回退7天
                    test_dt = (base_date - timedelta(days=d)).strftime("%Y-%m-%d 14:40")
                    if (df['dt_min'] == test_dt).any():
                        return test_dt
                return None

            # 在当日内，若14:40未到/尚未写入，使用当日<=14:40的最后一条
            def find_row_at_or_before(df, target_dt_min):
                # 精确匹配分钟
                row = df[df['dt_min'] == target_dt_min]
                if not row.empty:
                    return row.iloc[0]
                # 同日内向前寻找小于等于该分钟的最后一条
                date_part = target_dt_min[:10]
                time_part = target_dt_min[11:]
                same_day = df[df['dt_min'].str.startswith(date_part)]
                if same_day.empty:
                    return None
                same_day = same_day.copy()
                same_day = same_day[same_day['dt_min'].str.slice(11) <= time_part]
                if same_day.empty:
                    return None
                return same_day.iloc[-1]

            # 最长等待窗口（应对实时落盘延迟）
            max_wait_sec = 120
            wait_step_sec = 10

            features = []
            codes_with_features = []
            close_map = {}
            skipped_no_data = 0  # 统计因无数据被跳过的股票数
            # 诊断统计（不改变逻辑，只打印）
            diag_total = 0
            diag_m_eq_p = 0
            diag_use_lastrow_m = 0
            diag_use_lastrow_p = 0
            diag_prev_dt_none = 0
            diag_same_day_used = 0
            diag_m_time_samples = []  # 收集少量时间样本便于观察
            diag_p_time_samples = []
            print("📊 开始遍历实时features目录收集特征...")

            features_dir = Path(r"C:\\Users\\ASUS\\qlib_data_recent\\features")
            all_stock_folders = [p for p in features_dir.iterdir() if p.is_dir() and (p.name.startswith('SH') or p.name.startswith('SZ'))]
            print(f"发现股票文件夹数: {len(all_stock_folders)}")

            # 等待阶段：若总体有效股票过少，则先等待文件写入（但不会无限等）
            def count_ready_stocks(sample_n=50):
                ready = 0
                sample = min(len(all_stock_folders), sample_n)
                for folder in all_stock_folders[:sample]:
                    csv_path = folder / 'data.csv'
                    if not csv_path.exists():
                        continue
                    try:
                        df = pd.read_csv(csv_path)
                        if find_row_at_or_before(df, pred_dt) is not None:
                            ready += 1
                    except Exception:
                        pass
                return ready

            # 额外调试：打印前3个文件的mtime与尾部时间
            for folder in all_stock_folders[:3]:
                csv_path = folder / 'data.csv'
                if csv_path.exists():
                    try:
                        mtime = datetime.fromtimestamp(csv_path.stat().st_mtime)
                        df_dbg = pd.read_csv(csv_path)
                        tail_dt = df_dbg['datetime'].tail(3).tolist() if 'datetime' in df_dbg.columns else []
                        print(f"[DBG] {folder.name} mtime={mtime} tail={tail_dt}")
                    except Exception as e:
                        print(f"[DBG] {folder.name} 读取失败: {e}")

            waited = 0
            ready = count_ready_stocks()
            if ready == 0:
                print("[INFO] 就绪样本为0，跳过等待，直接尝试读取并预测（使用回退策略）。")
            else:
                while waited < max_wait_sec and ready < 10:
                    print(f"⏳ 14:40数据未完全就绪，已就绪样本: {ready}，等待 {wait_step_sec}s...")
                    time.sleep(wait_step_sec)
                    waited += wait_step_sec
                    ready = count_ready_stocks()

            for stock_folder in all_stock_folders:
                code = stock_folder.name
                csv_path = stock_folder / 'data.csv'
                if not csv_path.exists():
                    continue
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    continue

                # 统一时间列为 datetime，并派生分钟粒度字符串列 dt_min（YYYY-MM-DD HH:MM）
                try:
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                        df = df.dropna(subset=['datetime'])
                        if df.empty:
                            skipped_no_data += 1
                            continue
                        df['dt_min'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
                    else:
                        skipped_no_data += 1
                        continue
                except Exception:
                    skipped_no_data += 1
                    continue

                m = find_row_at_or_before(df, pred_dt)
                _used_lastrow_m = False
                _used_lastrow_p = False
                if m is None:
                    # 当日没有，则回退到最近交易日14:40（仅真实回退，不再使用最后一行兜底）
                    prev_dt = get_prev_trading_dt(df, pred_dt)
                    if prev_dt is None:
                        diag_prev_dt_none += 1
                        skipped_no_data += 1
                        continue
                    row_prev_exact = df[df['dt_min'] == prev_dt]
                    if row_prev_exact.empty:
                        skipped_no_data += 1
                        continue
                    # 使用前一交易日14:40作为 m，再向前一交易日取 p
                    m = row_prev_exact.iloc[0]
                    prev2_dt = get_prev_trading_dt(df, prev_dt)
                    if prev2_dt is None:
                        skipped_no_data += 1
                        continue
                    row_prev2_exact = df[df['dt_min'] == prev2_dt]
                    if row_prev2_exact.empty:
                        skipped_no_data += 1
                        continue
                    p = row_prev2_exact.iloc[0]
                else:
                    prev_dt = get_prev_trading_dt(df, pred_dt)
                    if prev_dt is None:
                        diag_prev_dt_none += 1
                        skipped_no_data += 1
                        continue
                    else:
                        row_prev_exact = df[df['dt_min'] == prev_dt]
                        if not row_prev_exact.empty:
                            p = row_prev_exact.iloc[0]
                        else:
                            skipped_no_data += 1
                            continue

                # 确保 m 与 p 时间不同（避免同质化）
                try:
                    if str(m['dt_min']) == str(p['dt_min']):
                        skipped_no_data += 1
                        continue
                except Exception:
                    pass

                # ===== 训练一致的前置特征工程（价格比值归一：价 / 前一交易日的日末收盘） =====
                try:
                    # 获取 m 所在日期与其前一交易日日期
                    m_date = str(m['dt_min'])[:10]
                    # 找 m 的前一交易日（相对 m_date）
                    def _prev_trading_date(df_local, base_date):
                        base_dt = datetime.strptime(base_date, '%Y-%m-%d')
                        for d in range(1, 8):
                            dstr = (base_dt - timedelta(days=d)).strftime('%Y-%m-%d')
                            if (df_local['dt_min'].str.startswith(dstr)).any():
                                return dstr
                        return None
                    prev_date_for_m = _prev_trading_date(df, m_date)
                    if prev_date_for_m is None:
                        skipped_no_data += 1
                        continue
                    day_m_prev = df[df['dt_min'].str.startswith(prev_date_for_m)]
                    if day_m_prev.empty:
                        skipped_no_data += 1
                        continue
                    ref_close_m = float(day_m_prev.iloc[-1]['close'])
                    if not np.isfinite(ref_close_m) or ref_close_m <= 0:
                        skipped_no_data += 1
                        continue

                    # 获取 p 所在日期与其前一交易日日期（即相当于 m 的前前一日）
                    p_date = str(p['dt_min'])[:10]
                    prev_date_for_p = _prev_trading_date(df, p_date)
                    if prev_date_for_p is None:
                        skipped_no_data += 1
                        continue
                    day_p_prev = df[df['dt_min'].str.startswith(prev_date_for_p)]
                    if day_p_prev.empty:
                        skipped_no_data += 1
                        continue
                    ref_close_p = float(day_p_prev.iloc[-1]['close'])
                    if not np.isfinite(ref_close_p) or ref_close_p <= 0:
                        skipped_no_data += 1
                        continue

                    # 计算 vwap
                    vwap_current_raw = (m['open'] + 2*m['high'] + 2*m['low'] + m['close']) / 6
                    vwap_prev_raw = (p['open'] + 2*p['high'] + 2*p['low'] + p['close']) / 6

                    # 比值归一（与 handler 中 Cut(.../Ref(Ref(DayLast($close),240),-1)) 的含义对齐）
                    m_open_r  = float(m['open'])  / ref_close_m
                    m_high_r  = float(m['high'])  / ref_close_m
                    m_low_r   = float(m['low'])   / ref_close_m
                    m_close_r = float(m['close']) / ref_close_m
                    m_vwap_r  = float(vwap_current_raw) / ref_close_m

                    p_open_r  = float(p['open'])  / ref_close_p
                    p_high_r  = float(p['high'])  / ref_close_p
                    p_low_r   = float(p['low'])   / ref_close_p
                    p_close_r = float(p['close']) / ref_close_p
                    p_vwap_r  = float(vwap_prev_raw) / ref_close_p

                except Exception:
                    skipped_no_data += 1
                    continue

                feature_vector = [
                    m_open_r, m_high_r, m_low_r, m_close_r, m_vwap_r,
                    p_open_r, p_high_r, p_low_r, p_close_r, p_vwap_r,
                    float(m['volume']), float(p['volume']), 1.0, 0.0, 0.0
                ]
                features.append(feature_vector)
                codes_with_features.append(code)
                close_map[code] = m['close']

                # 诊断统计更新
                diag_total += 1
                # 记录时间样本（仅取前若干）
                try:
                    _m_dt = str(m['datetime']) if 'datetime' in m else None
                    _p_dt = str(p['datetime']) if 'datetime' in p else None
                    if _m_dt:
                        if len(diag_m_time_samples) < 5:
                            diag_m_time_samples.append(_m_dt)
                    if _p_dt:
                        if len(diag_p_time_samples) < 5:
                            diag_p_time_samples.append(_p_dt)
                    # 同日判断（字符串前10位为日期）
                    if _m_dt and _p_dt and _m_dt[:10] == _p_dt[:10]:
                        diag_same_day_used += 1
                except Exception:
                    pass
                # m 与 p 是否同一行（关键指标）
                try:
                    if (m is p) or (
                        ('datetime' in m and 'datetime' in p and str(m['datetime']) == str(p['datetime'])) and
                        all((m.get(col, np.nan) == p.get(col, np.nan)) for col in ['open','high','low','close','volume'])
                    ):
                        diag_m_eq_p += 1
                except Exception:
                    pass
                if _used_lastrow_m:
                    diag_use_lastrow_m += 1
                if _used_lastrow_p:
                    diag_use_lastrow_p += 1

            print(f"✅ 特征收集完成: {len(features)} 支股票")
            # 打印诊断汇总
            if diag_total > 0:
                print("\n🧪 诊断汇总（不影响逻辑，仅用于定位同质化根因）")
                print(f"   样本总数: {diag_total}")
                print(f"   m 与 p 完全相同的样本数: {diag_m_eq_p} ({diag_m_eq_p/diag_total:.2%})")
                print(f"   使用最后一行作为 m 的样本数: {diag_use_lastrow_m} ({diag_use_lastrow_m/diag_total:.2%})")
                print(f"   使用最后一行作为 p 的样本数: {diag_use_lastrow_p} ({diag_use_lastrow_p/diag_total:.2%})")
                print(f"   prev_dt 未找到次数: {diag_prev_dt_none}")
                print(f"   m 与 p 属于同一交易日的样本数: {diag_same_day_used} ({diag_same_day_used/diag_total:.2%})")
                if diag_m_time_samples:
                    print(f"   m 时间样本: {diag_m_time_samples}")
                if diag_p_time_samples:
                    print(f"   p 时间样本: {diag_p_time_samples}")
            if len(features) == 0:
                print("❌ 没有有效的特征数据")
                return False

            # 3.5. 加载训练时的标准化参数（强一致性：未保存则中止）
            if not hasattr(self, 'norm_params') or self.norm_params is None:
                print("📊 加载训练时的标准化参数...")
                self.norm_params = self._load_normalization_params()
                if self.norm_params is None:
                    print("❌ 未找到训练时的标准化参数，已终止预测（禁止重算/兜底）。")
                    return False
            
            # 3.6. 应用特征标准化（与训练时一致）
            print("📊 应用特征标准化...")
            X = self._normalize_features(features, self.norm_params)
            print(f"   标准化后特征范围: [{X.min():.4f}, {X.max():.4f}]")
            print(f"   标准化后特征均值: {X.mean():.4f}, 标准差: {X.std():.4f}")

            # 4. 模型预测
            print("🔮 开始模型预测...")
            try:
                predictions = self.model.predict(X)
                # 若返回DataFrame/Series，统一为扁平ndarray
                if hasattr(predictions, 'values'):
                    predictions = predictions.values
                predictions = np.asarray(predictions).reshape(-1)
            except AttributeError:
                # 兼容QLib LGBModel要求Dataset的情况，直接调用底层模型
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'predict'):
                    predictions = self.model.model.predict(X)
                    predictions = np.asarray(predictions).reshape(-1)
                else:
                    # 最后兜底：尝试直接调用predict
                    predictions = np.asarray(self.model.predict(X)).reshape(-1)

            print(f"📊 预测完成: {len(predictions)} 个分数")
            print(f"   预测值范围: {predictions.min():.6f} 到 {predictions.max():.6f}")
            print(f"   预测值标准差: {predictions.std():.6f}")

            df = pd.DataFrame({
                'code': codes_with_features,
                'score': predictions
            })
            df = df.sort_values('score', ascending=False).reset_index(drop=True)
            df['rank'] = range(1, len(df) + 1)
            df['shares'] = 200
            df['action'] = 'buy'
            df['timestamp'] = current_time.strftime('%Y-%m-%d %H:%M:%S')
            df_top50 = df.head(50)
            prices = [close_map.get(c, 0.0) for c in df_top50['code']]
            df_top50['price'] = prices
            csv_path = Path("daily_predictions.csv")
            df_top50.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ CSV已保存: {csv_path}")
            print(f"📊 Top 10 信号:")
            print(df_top50[['rank', 'code', 'score', 'price']].head(10).to_string(index=False))
            return True
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_scheduler(self):
        """启动定时任务"""
        print("\n" + "="*70)
        print("[SCHEDULER] 定时预测任务调度")
        print("="*70)
        print("[INFO] 预测信号: 每天 14:40")
        print("[INFO] CSV保存: 预测完成后自动保存")
        print("="*70 + "\n")
        # 14:40执行预测
        schedule.every().day.at("14:40").do(self.predict_and_save_csv)
        # 检查当前时间
        current_hour = datetime.now().hour
        current_minute = datetime.now().minute
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"💡 当前时间: {current_time}")
        # 如果接近14:40，立即执行预测
        if current_hour == 14 and 38 <= current_minute <= 42:
            print("⏰ 检测到当前时间接近14:40，立即执行预测...")
            self.predict_and_save_csv()
        elif current_hour > 14 and current_hour <= 16:
            print("⏰ 已过14:40，立即执行当日预测一次...")
            self.predict_and_save_csv()
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

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='预测调度器')
    parser.add_argument('--once', action='store_true', help='立即执行一次预测并退出')
    args = parser.parse_args()
    
    try:
        scheduler = PredictionScheduler(test_mode=False) # 确保test_mode为False
        
        if args.once:
            print("[ONCE] 立即执行一次预测...")
            ok = scheduler.predict_and_save_csv()
            print(f"[ONCE] 预测完成: {ok}")
            return
        
        scheduler.start_scheduler()
            
    except Exception as e:
        print(f"[ERROR] 程序启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
