
# ============================================
# 独立预测和CSV保存脚本
# 功能：定时预测并保存CSV文件
# ============================================

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import time
import schedule
from datetime import datetime, timedelta
import pickle

# 添加路径（必须在导入qlib模块之前）
PROJECT_ROOT = "/home/intern0/qlib"
WORKDIR = "/home/intern0/qlib/examples/highfreq"

for path in (PROJECT_ROOT, WORKDIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# 在路径设置后导入qlib模块
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data import D
from highfreq_handler_alpha158fz7 import HighFreqHandler

def is_trading_day(date_obj=None):
    """判断是否为交易日（简单实现：排除周末）"""
    if date_obj is None:
        date_obj = datetime.now()
    weekday = date_obj.weekday()  # 0=Monday, 6=Sunday
    return weekday < 5  # 周一到周五

def get_previous_trading_day(date_obj):
    """获取最近的交易日（向前查找，排除周末）"""
    if date_obj is None:
        date_obj = datetime.now().date()
    elif isinstance(date_obj, str):
        date_obj = datetime.strptime(date_obj, '%Y-%m-%d').date()
    
    # 如果当前是交易日，直接返回
    if is_trading_day(date_obj):
        return date_obj
    
    # 向前查找最近的交易日
    current = date_obj
    for i in range(1, 8):  # 最多查找7天
        prev_date = current - timedelta(days=i)
        if is_trading_day(prev_date):
            return prev_date
    # 如果都没找到（理论上不会发生），返回原日期
    return date_obj

class PredictionScheduler:
    def __init__(self, test_mode=False):
        # 配置 - 使用实时数据路径（服务器路径）
        self.KQ_DATA_DIR = Path("/home/intern0/qlib_data_recent")
        self.STOCK_POOL_CSV = "/home/intern0/qlib/examples/highfreq/sorted_high_preclose_ratio_2025.csv"
        # 工作目录
        self.WORKDIR = WORKDIR
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
            "provider_uri": str(self.KQ_DATA_DIR),
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
        import glob
        from pathlib import Path
        
        try:
            # 尝试从最新的 recorder 加载模型
            # 首先尝试获取当前 recorder
            try:
                recorder = R.get_recorder()
                if recorder is not None:
                    self.model = recorder.load_object("model")
                    print(f"✅ 从当前recorder加载模型成功: {type(self.model).__name__}")
                    return
            except:
                pass
            
            # 如果当前recorder不存在，查找最新的experiment和run
            mlruns_dir = Path(self.WORKDIR) / "mlruns"
            if not mlruns_dir.exists():
                mlruns_dir = Path("mlruns")  # 尝试当前目录
            
            if mlruns_dir.exists():
                # 查找所有experiment目录
                exp_dirs = sorted([d for d in mlruns_dir.iterdir() if d.is_dir() and d.name.isdigit()], 
                                key=lambda x: x.stat().st_mtime, reverse=True)
                
                for exp_dir in exp_dirs[:5]:  # 只检查最近5个experiment
                    run_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], 
                                    key=lambda x: x.stat().st_mtime, reverse=True)
                    
                    for run_dir in run_dirs[:3]:  # 每个experiment检查最近3个run
                        try:
                            recorder_path = run_dir / "artifacts" / "recorder"
                            if not recorder_path.exists():
                                continue
                            
                            # 尝试加载模型
                            model_path = recorder_path / "model.pkl"
                            if model_path.exists():
                                import pickle
                                with open(model_path, 'rb') as f:
                                    self.model = pickle.load(f)
                                print(f"✅ 从最新recorder加载模型成功: {run_dir.name} ({type(self.model).__name__})")
                                return
                        except Exception as e:
                            continue
            
            # 如果都失败了，尝试使用默认方式
            recorder = R.get_recorder()
            self.model = recorder.load_object("model")
            print(f"✅ 从默认recorder加载模型成功: {type(self.model).__name__}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_normalization_params(self):
        """从训练 recorder 读取标准化参数。若未保存，直接报错，不重算、不兜底。"""
        try:
            from qlib.workflow import R
            from pathlib import Path
            import pickle
            
            # 首先尝试从当前recorder加载
            try:
                rec = R.get_recorder()
                if rec is not None:
                    for key in ("norm_params", "highfreq_norm", "processor_state", "highfreq_norm_params"):
                        try:
                            obj = rec.load_object(key)
                            if isinstance(obj, dict) and "price" in obj and "volume" in obj:
                                print(f"✅ 从当前recorder加载标准化参数: {key}")
                                return obj
                        except Exception:
                            continue
            except:
                pass
            
            # 如果当前recorder没有，查找最新的recorder
            mlruns_dir = Path(self.WORKDIR) / "mlruns"
            if not mlruns_dir.exists():
                mlruns_dir = Path("mlruns")
            
            if mlruns_dir.exists():
                exp_dirs = sorted([d for d in mlruns_dir.iterdir() if d.is_dir() and d.name.isdigit()], 
                                key=lambda x: x.stat().st_mtime, reverse=True)
                
                for exp_dir in exp_dirs[:5]:
                    run_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], 
                                    key=lambda x: x.stat().st_mtime, reverse=True)
                    
                    for run_dir in run_dirs[:3]:
                        try:
                            recorder_path = run_dir / "artifacts" / "recorder"
                            if not recorder_path.exists():
                                continue
                            
                            # 尝试加载标准化参数
                            for key in ("norm_params", "highfreq_norm", "processor_state", "highfreq_norm_params"):
                                param_path = recorder_path / f"{key}.pkl"
                                if param_path.exists():
                                    with open(param_path, 'rb') as f:
                                        obj = pickle.load(f)
                                    if isinstance(obj, dict) and "price" in obj and "volume" in obj:
                                        print(f"✅ 从最新recorder加载标准化参数: {key} (run: {run_dir.name})")
                                        return obj
                        except Exception as e:
                            continue
            
            print("❌ 未在recorder中找到标准化参数(norm_params)。")
            print("   必须使用训练时保存的参数，当前不支持重算或兜底。")
            return None
        except Exception as e:
            print(f"❌ 读取recorder标准化参数失败: {e}")
            return None
    
    def _normalize_features(self, feature_df: pd.DataFrame, norm_params):
        """对特征应用标准化（复用 HighFreqNorm 逻辑）"""
        price_med = norm_params["price"]["med"]
        price_mad = max(norm_params["price"]["mad"], 1e-6)
        volume_med = norm_params["volume"]["med"]
        volume_mad = max(norm_params["volume"]["mad"], 1e-6)
        
        values = feature_df.to_numpy(dtype=np.float32)
        columns = feature_df.columns
        for idx, col in enumerate(columns):
            col_name = col[1] if isinstance(col, tuple) and len(col) > 1 else str(col)
            col_name_lower = str(col_name).lower()
            if "volume" in col_name_lower:
                vals = np.log1p(values[:, idx])
                vals = (vals - volume_med) / volume_mad
            else:
                vals = (values[:, idx] - price_med) / price_mad
            values[:, idx] = np.clip(vals, -8.0, 8.0)
        return values
    
    def _compute_features_with_handler(self, target_date, stock_codes, target_time=None, history_days=2, batch_size=100):
        """
        使用handler计算特征（极速优化版：最小历史数据 + 分批处理）
        
        参数:
            target_date: 目标日期
            stock_codes: 股票代码列表
            target_time: 目标时间
            history_days: 历史数据天数（默认2天，只加载目标日期+前1天）
            batch_size: 分批处理大小（默认100支股票一批）
        """
        if not stock_codes:
            return None
        # 默认使用14:40，如果指定了target_time则使用指定时间
        if target_time is None:
            target_time = datetime.min.time().replace(hour=14, minute=40)
        elif isinstance(target_time, str):
            # 支持 "HH:MM" 格式
            hour, minute = map(int, target_time.split(':'))
            target_time = datetime.min.time().replace(hour=hour, minute=minute)
        target_ts = pd.Timestamp(datetime.combine(target_date, target_time))
        
        # 极速优化：只加载目标日期当天 + 前1-2天的数据（最小必要范围）
        start_time = (target_date - timedelta(days=history_days)).strftime("%Y-%m-%d 09:30:00")
        end_time = target_ts.strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"📊 开始加载特征数据（{len(stock_codes)}支股票，{history_days}天历史数据，分批处理，每批{batch_size}支）...")
        print(f"   时间范围: {start_time} 至 {end_time}")
        start_load_time = time.time()
        
        # 分批处理股票，避免一次性加载太多数据
        all_slices = []
        total_batches = (len(stock_codes) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(stock_codes))
            batch_codes = stock_codes[batch_start:batch_end]
            
            print(f"   ⏳ 处理批次 {batch_idx + 1}/{total_batches} ({len(batch_codes)}支股票)...")
            batch_start_time = time.time()
            
            try:
                handler = HighFreqHandler(
                    instruments=batch_codes,
                    start_time=start_time,
                    end_time=end_time,
                    infer_processors=[],
                    learn_processors=[],
                )
                handler.setup_data()
                
                dataset = DatasetH(handler=handler, instruments=batch_codes, segments={'pred': (start_time, end_time)})
                prepared = dataset.prepare("pred", col_set=["feature"], data_key=DataHandlerLP.DK_I)
                
                feature_df = prepared.get("feature")
                if feature_df is None or feature_df.empty:
                    print(f"      ⚠️ 批次 {batch_idx + 1} 特征数据为空，跳过")
                    continue
                
                # 提取目标时间点的特征
                try:
                    batch_slice = feature_df.xs(target_ts, level='datetime')
                except KeyError:
                    # 如果精确时间点不存在，尝试找最接近的时间点
                    available_times = feature_df.index.get_level_values('datetime').unique()
                    if len(available_times) == 0:
                        continue
                    available_times = sorted(available_times)
                    closest_time = None
                    for t in reversed(available_times):
                        if t <= target_ts:
                            closest_time = t
                            break
                    if closest_time is None:
                        closest_time = available_times[0]
                    batch_slice = feature_df.xs(closest_time, level='datetime')
                
                batch_slice = batch_slice.dropna(how='all')
                if not batch_slice.empty:
                    all_slices.append(batch_slice)
                
                batch_time = time.time() - batch_start_time
                print(f"      ✅ 批次 {batch_idx + 1} 完成（耗时 {batch_time:.1f}秒，有效股票 {len(batch_slice)} 支）")
                
            except Exception as e:
                print(f"      ❌ 批次 {batch_idx + 1} 失败: {e}")
                continue
        
        if not all_slices:
            print("   ❌ 所有批次都没有有效特征数据")
            return None
        
        # 合并所有批次的结果
        print("   ⏳ 正在合并批次结果...")
        final_slice = pd.concat(all_slices)
        total_time = time.time() - start_load_time
        print(f"   ✅ 特征提取完成（总耗时 {total_time:.1f}秒，有效股票 {len(final_slice)} 支）")
        return final_slice
    
    def _load_close_prices(self, codes, target_timestamp):
        close_map = {}
        if not codes:
            return close_map
        try:
            ts_str = target_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            close_df = D.features(
                codes,
                ["$close"],
                start_time=ts_str,
                end_time=ts_str,
                freq="1min",
            )
            if close_df is None or close_df.empty:
                return close_map
            col = close_df.columns[0]
            reset_df = close_df.reset_index()
            reset_df = reset_df.dropna(subset=[col])
            for code in reset_df['instrument'].unique():
                inst_rows = reset_df[reset_df['instrument'] == code]
                if inst_rows.empty:
                    continue
                price = float(inst_rows.iloc[-1][col])
                close_map[code] = price
        except Exception:
            return close_map
        return close_map
    
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
    
    def predict_and_save_csv(self, target_date=None, target_time=None):
        """生成买入预测CSV
        
        参数:
            target_date: 目标日期，格式 'YYYY-MM-DD'，默认为None（使用今天）
            target_time: 目标时间，格式 'HH:MM' 或 datetime.time 对象，默认为None（使用14:40）
        """
        import glob
        if target_date is None:
            target_date = datetime.now().date()
        else:
            if isinstance(target_date, str):
                target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        
        # 处理target_time参数
        if target_time is None:
            target_time = datetime.min.time().replace(hour=14, minute=40)
            time_str = "14:40"
        elif isinstance(target_time, str):
            hour, minute = map(int, target_time.split(':'))
            target_time = datetime.min.time().replace(hour=hour, minute=minute)
            time_str = target_time.strftime("%H:%M")
        else:
            time_str = target_time.strftime("%H:%M")
        
        # 检查是否为交易日，如果不是，自动调整到最近的交易日
        if not is_trading_day(target_date):
            original_date = target_date
            target_date = get_previous_trading_day(target_date)
            print(f"⚠️  {original_date} 不是交易日，自动调整为最近的交易日: {target_date}")
        
        print(f"\n🔮 开始预测 {target_date} {time_str}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            target_timestamp = datetime.combine(target_date, target_time)
            stock_codes = self._get_stock_codes()
            print(f"📊 目标股票数量: {len(stock_codes)}")
            feature_slice = self._compute_features_with_handler(target_date, stock_codes, target_time)
            if feature_slice is None or feature_slice.empty:
                print(f"❌ 未能获取{time_str}的180维特征，请检查bin数据是否最新。")
                return False
            codes_with_features = feature_slice.index.tolist()
            print(f"✅ 成功获取 {len(codes_with_features)} 支股票的特征")
            
            if not hasattr(self, 'norm_params') or self.norm_params is None:
                print("📊 加载训练时的标准化参数...")
                self.norm_params = self._load_normalization_params()
                if self.norm_params is None:
                    print("❌ 未找到训练时的标准化参数，已终止预测（禁止重算/兜底）。")
                    return False

            X = self._normalize_features(feature_slice, self.norm_params)
            close_map = self._load_close_prices(codes_with_features, target_timestamp)
            print(f"   标准化后特征范围: [{np.nanmin(X):.4f}, {np.nanmax(X):.4f}]")
            print(f"   标准化后特征均值: {np.nanmean(X):.4f}, 标准差: {np.nanstd(X):.4f}")
            print(f"   特征形状: {X.shape}")

            # 3.6. 加载训练时的标准化参数（强一致性：未保存则中止）
            if not hasattr(self, 'norm_params') or self.norm_params is None:
                print("📊 加载训练时的标准化参数...")
                self.norm_params = self._load_normalization_params()
                if self.norm_params is None:
                    print("❌ 未找到训练时的标准化参数，已终止预测（禁止重算/兜底）。")
                    return False
            
            # 3.7. 应用特征标准化（与训练时一致）
            if X.shape[1] == 15:
                print("📊 应用特征标准化（15特征模式）...")
                X = self._normalize_features(X, self.norm_params)
            else:
                print(f"⚠️ 特征数量异常: {X.shape[1]}，跳过标准化（可能不是15个基础特征）")
            
            print(f"   标准化后特征范围: [{X.min():.4f}, {X.max():.4f}]")
            print(f"   标准化后特征均值: {X.mean():.4f}, 标准差: {X.std():.4f}")
            print(f"   特征形状: {X.shape}")

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
            # 使用目标日期和时间的timestamp，而不是当前时间
            df['timestamp'] = target_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            df_top50 = df.head(50).copy()  # 避免SettingWithCopyWarning
            # 重新获取精确的14:40价格（确保价格准确）
            prices = []
            features_dir = Path(self.KQ_DATA_DIR) / "features"
            for code in df_top50['code']:
                price = 0.0
                # 优先从close_map获取（已经是14:40的价格）
                if code in close_map:
                    price = float(close_map[code])
                else:
                    # 如果close_map中没有，直接从数据获取精确的14:40价格
                    try:
                        folder_name = code
                        csv_path = features_dir / folder_name / 'data.csv'
                        if csv_path.exists():
                            df_stock = pd.read_csv(csv_path)
                            df_stock['datetime'] = pd.to_datetime(df_stock['datetime'], errors='coerce')
                            df_stock = df_stock.dropna(subset=['datetime'])
                            exact_time = target_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                            exact_data = df_stock[df_stock['datetime'] == pd.to_datetime(exact_time)]
                            if not exact_data.empty:
                                price = float(exact_data.iloc[0]['close'])
                            else:
                                # 如果精确时间没有，找同一天<=14:40的最后一条
                                same_day = df_stock[df_stock['datetime'].dt.date == target_date]
                                if not same_day.empty:
                                    # 使用target_time计算分钟数
                                    target_minutes = target_time.hour * 60 + target_time.minute
                                    before_target = same_day[same_day['datetime'].dt.hour * 60 + same_day['datetime'].dt.minute <= target_minutes]
                                    if not before_target.empty:
                                        price = float(before_target.iloc[-1]['close'])
                    except Exception as e:
                        print(f"[WARN] {code} 获取价格失败: {e}")
                prices.append(price)
            df_top50['price'] = prices
            # 保存到工作目录
            csv_path = Path(self.WORKDIR) / "daily_predictions.csv"
            df_top50.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ CSV已保存: {csv_path}")
            print(f"📊 Top 10 信号:")
            print(df_top50[['rank', 'code', 'score', 'price']].head(10).to_string(index=False))
            
            # 保存持仓到positions.json（买入时自动保存）- 使用工作目录
            positions_file = Path(self.WORKDIR) / "positions.json"
            try:
                buy_date = target_date.strftime('%Y-%m-%d')
                positions = {}
                for _, row in df_top50.iterrows():
                    code = row['code']
                    positions[code] = {
                        'buy_date': buy_date,
                        'buy_price': float(row['price']),
                        'shares': int(row['shares']),
                        'code': code,
                        'rank': int(row['rank']),
                        'score': float(row['score'])
                    }
                
                with open(positions_file, 'w', encoding='utf-8') as f:
                    json.dump(positions, f, indent=2, ensure_ascii=False)
                
                print(f"✅ 持仓信息已保存: {positions_file} ({len(positions)} 个股票)")
            except Exception as e:
                print(f"[WARN] 保存持仓失败: {e}")
                import traceback
                traceback.print_exc()
                # 不因为保存持仓失败而中断整个流程
            
            return True
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_and_save_sell_csv(self, target_date=None):
        """在10:46预测并保存卖出CSV（包含所有持仓股票）
        
        参数:
            target_date: 目标日期，格式 'YYYY-MM-DD'，默认为None（使用今天）
        """
        import json
        import glob
        if target_date is None:
            target_date = datetime.now().date()
        else:
            if isinstance(target_date, str):
                target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        
        # 检查是否为交易日，如果不是，自动调整到最近的交易日
        if not is_trading_day(target_date):
            original_date = target_date
            target_date = get_previous_trading_day(target_date)
            print(f"⚠️  {original_date} 不是交易日，自动调整为最近的交易日: {target_date}")
        
        print(f"\n🔮 开始 {target_date} 10:46卖出预测: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            # 1. 读取持仓信息
            positions_file = Path("positions.json")
            positions = {}
            
            # 尝试从positions.json读取（使用工作目录）
            positions_file = Path(self.WORKDIR) / "positions.json"
            if positions_file.exists():
                file_size = positions_file.stat().st_size
                if file_size > 0:
                    try:
                        with open(positions_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content:
                                positions = json.loads(content)
                    except json.JSONDecodeError as e:
                        print(f"[WARN] JSON解析失败: {positions_file}")
                        print(f"[WARN] 错误位置: 第{e.lineno}行，第{e.colno}列")
                        print(f"[WARN] 将尝试从买入预测CSV自动生成持仓...")
                        positions = {}
            
            # 如果持仓为空，尝试从买入预测CSV自动生成（兜底方案）
            if not positions or len(positions) == 0:
                print("[WARN] 持仓为空！正常情况下，买入预测应该已经保存持仓到positions.json")
                print("[INFO] 尝试从买入预测CSV自动生成持仓（兜底方案）...")
                
                # 查找最新的买入预测CSV文件（使用工作目录）
                csv_files = []
                csv_pattern = Path(self.WORKDIR) / "daily_predictions*.csv"
                for csv_file in Path(self.WORKDIR).glob("daily_predictions*.csv"):
                    if csv_file.name != "daily_predictions_sell.csv":  # 排除卖出文件
                        csv_files.append(csv_file)
                
                if not csv_files:
                    print("[WARN] 未找到买入预测CSV文件，无法生成卖出预测")
                    return False
                
                # 使用最新的CSV文件
                latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
                print(f"[INFO] 使用买入预测文件: {latest_csv.name}")
                
                try:
                    df = pd.read_csv(latest_csv)
                    if df.empty:
                        print("[WARN] 买入预测CSV文件为空，无法生成卖出预测")
                        return False
                    
                    # 只处理买入信号
                    buy_df = df[df['action'] == 'buy'].copy() if 'action' in df.columns else df.copy()
                    
                    if buy_df.empty:
                        print("[WARN] 买入预测CSV中没有买入信号，无法生成卖出预测")
                        return False
                    
                    # 从CSV生成持仓，从timestamp提取买入日期
                    positions = {}
                    for _, row in buy_df.iterrows():
                        code = row['code']
                        
                        # 从timestamp提取买入日期
                        buy_date = target_date.strftime('%Y-%m-%d')  # 默认使用目标日期
                        if 'timestamp' in row and pd.notna(row['timestamp']):
                            try:
                                ts = pd.to_datetime(row['timestamp'])
                                buy_date = ts.strftime('%Y-%m-%d')
                            except:
                                pass
                        
                        positions[code] = {
                            'buy_date': buy_date,
                            'buy_price': float(row.get('price', 0.0)),
                            'shares': int(row.get('shares', 200)),
                            'code': code,
                            'rank': int(row.get('rank', 0)),
                            'score': float(row.get('score', 0.0))
                        }
                    
                    # 保存到positions.json
                    with open(positions_file, 'w', encoding='utf-8') as f:
                        json.dump(positions, f, indent=2, ensure_ascii=False)
                    
                    print(f"[OK] 从CSV自动生成持仓: {len(positions)} 个股票")
                    
                except Exception as e:
                    print(f"[ERROR] 从CSV生成持仓失败: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            
            if not isinstance(positions, dict):
                print(f"[WARN] 持仓文件格式错误: 期望字典类型，实际为 {type(positions)}")
                return False
            
            if len(positions) == 0:
                print("[WARN] 持仓为空，无法生成卖出预测")
                return False
            
            print(f"[INFO] 读取持仓信息: {len(positions)} 个股票")
            
            # 2. 提取持仓股票代码列表
            stock_codes = list(positions.keys())
            print(f"📊 持仓股票数量: {len(stock_codes)}")
            
            # 3. 使用_compute_features_with_handler计算完整的180个特征（10:46时刻）
            target_time = datetime.min.time().replace(hour=10, minute=46)
            target_timestamp = datetime.combine(target_date, target_time)
            
            print("📊 开始计算持仓股票的完整特征（使用HighFreqHandler计算180维特征：15基础+158 Alpha158+7 FZ7）...")
            feature_slice = self._compute_features_with_handler(target_date, stock_codes, target_time)
            
            if feature_slice is None or feature_slice.empty:
                print(f"❌ 未能获取10:46的180维特征，请检查bin数据是否最新。")
                return False
            
            codes_with_features = feature_slice.index.tolist()
            # 显示特征数量信息
            num_features = feature_slice.shape[1] if hasattr(feature_slice, 'shape') else len(feature_slice.columns)
            print(f"✅ 成功获取 {len(codes_with_features)} 支股票的特征")
            print(f"   📊 特征数量: {num_features} 维（预期180维：15基础+158 Alpha158+7 FZ7）")
            if hasattr(feature_slice, 'columns'):
                feature_cols = list(feature_slice.columns)
                if len(feature_cols) > 0:
                    print(f"   📊 前5个特征: {feature_cols[:5]}")
                    if len(feature_cols) > 10:
                        print(f"   📊 后5个特征: {feature_cols[-5:]}")
            
            # 4. 加载标准化参数
            if not hasattr(self, 'norm_params') or self.norm_params is None:
                print("📊 加载训练时的标准化参数...")
                self.norm_params = self._load_normalization_params()
                if self.norm_params is None:
                    print("❌ 未找到训练时的标准化参数，已终止预测")
                    return False
            
            # 5. 应用特征标准化
            print("📊 应用特征标准化...")
            X = self._normalize_features(feature_slice, self.norm_params)
            print(f"   标准化后特征范围: [{np.nanmin(X):.4f}, {np.nanmax(X):.4f}]")
            print(f"   标准化后特征均值: {np.nanmean(X):.4f}, 标准差: {np.nanstd(X):.4f}")
            print(f"   特征形状: {X.shape}")
            
            # 6. 加载收盘价（用于后续处理）
            close_map = self._load_close_prices(codes_with_features, target_timestamp)
            
            # 7. 模型预测
            print("🔮 开始模型预测...")
            try:
                predictions = self.model.predict(X)
                if hasattr(predictions, 'values'):
                    predictions = predictions.values
                predictions = np.asarray(predictions).reshape(-1)
            except AttributeError:
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'predict'):
                    predictions = self.model.model.predict(X)
                    predictions = np.asarray(predictions).reshape(-1)
                else:
                    predictions = np.asarray(self.model.predict(X)).reshape(-1)
            
            print(f"📊 预测完成: {len(predictions)} 个分数")
            
            # 8. 生成CSV
            df = pd.DataFrame({
                'code': codes_with_features,
                'score': predictions
            })
            # 使用目标日期的10:46作为timestamp，而不是当前时间
            df['timestamp'] = target_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            # 保存CSV（包含所有持仓股票，不限制TOP 50）- 使用工作目录
            csv_path = Path(self.WORKDIR) / "daily_predictions_sell.csv"
            
            # 读取现有文件（如果存在），保留非目标日期的数据
            if csv_path.exists():
                df_existing = pd.read_csv(csv_path)
                if 'timestamp' in df_existing.columns:
                    df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
                    # 移除目标日期的数据（避免重复）
                    df_existing = df_existing[df_existing['timestamp'].dt.date != target_date]
                    if not df_existing.empty:
                        df_combined = pd.concat([df_existing, df], ignore_index=True)
                    else:
                        df_combined = df
                else:
                    df_combined = df
            else:
                df_combined = df
            
            df_combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ 卖出预测CSV已保存: {csv_path}")
            print(f"📊 预测股票数量: {len(df)}")
            print(f"📊 样本信号值:")
            print(df[['code', 'score']].head(10).to_string(index=False))
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
        print("[INFO] 买入预测信号: 每天 14:40")
        print("[INFO] 卖出预测信号: 每天 10:46")
        print("[INFO] CSV保存: 预测完成后自动保存")
        print("="*70 + "\n")
        # 14:40执行买入预测
        schedule.every().day.at("14:40").do(self.predict_and_save_csv)
        # 10:46执行卖出预测
        schedule.every().day.at("10:46").do(self.predict_and_save_sell_csv)
        # 检查当前时间（使用北京时间）
        current_time = datetime.now()
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"💡 当前时间: {current_time}")
        # 如果接近14:40，立即执行买入预测
        if current_hour == 14 and 38 <= current_minute <= 42:
            print("⏰ 检测到当前时间接近14:40，立即执行买入预测...")
            self.predict_and_save_csv()
        elif current_hour > 14 and current_hour <= 16:
            print("⏰ 已过14:40，立即执行当日买入预测一次...")
            self.predict_and_save_csv()
        # 如果接近10:46，立即执行卖出预测
        elif current_hour == 10 and 44 <= current_minute <= 48:
            print("⏰ 检测到当前时间接近10:46，立即执行卖出预测...")
            self.predict_and_save_sell_csv()
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
    parser.add_argument('--once', action='store_true', help='立即执行一次买入预测（默认14:40，可通过--time指定）并退出')
    parser.add_argument('--sell-once', action='store_true', help='立即执行一次卖出预测（10:46）并退出')
    parser.add_argument('--date', type=str, help='指定日期，格式: YYYY-MM-DD (例如: 2025-11-21)，用于生成指定日期的预测')
    parser.add_argument('--time', type=str, help='指定预测时间，格式: HH:MM (例如: 11:30)，用于测试任意时间点的预测')
    args = parser.parse_args()
    
    try:
        scheduler = PredictionScheduler(test_mode=False) # 确保test_mode为False
        
        # 如果没有指定日期，默认使用昨天（北京时间）
        target_date = args.date
        if target_date is None:
            yesterday = datetime.now() - timedelta(days=1)
            target_date = yesterday.strftime('%Y-%m-%d')
            print(f"[INFO] 未指定日期，使用昨天: {target_date}")
        else:
            print(f"[INFO] 使用指定日期: {target_date}")
        
        if args.sell_once:
            print(f"[SELL-ONCE] 立即执行一次 {target_date} 的卖出预测（10:46）...")
            ok = scheduler.predict_and_save_sell_csv(target_date=target_date)
            if ok:
                print("[OK] 卖出预测完成")
            else:
                print("[FAIL] 卖出预测失败")
            return
        
        if args.once:
            time_str = args.time if args.time else "14:40"
            print(f"[ONCE] 立即执行一次 {target_date} {time_str} 的买入预测...")
            ok = scheduler.predict_and_save_csv(target_date=target_date, target_time=args.time)
            print(f"[ONCE] 预测完成: {ok}")
            return
        
        scheduler.start_scheduler()
            
    except Exception as e:
        print(f"[ERROR] 程序启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
