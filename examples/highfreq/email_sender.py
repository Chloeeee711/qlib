# ============================================
# 邮件发送脚本
# 功能：读取CSV信号，定时发送买入和卖出邮件
# ============================================

import pandas as pd
import schedule
import time
from datetime import datetime, timedelta, date
from pathlib import Path
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免在服务器上显示
import matplotlib.pyplot as plt
import io
import base64

# 工作目录
WORKDIR = "/home/intern0/qlib/examples/highfreq"

class EmailSender:
    _qlib_initialized = False  # 类级别标志，避免重复初始化
    
    def __init__(self):
        # 邮件配置
        self.EMAIL_ADDRESS = 'chloechen@xcquant.com'
        self.EMAIL_PASSWORD = 'SX8DzzlyqbvwOk5V'
        
        # 收件人列表（主要收件人）
        self.RECIPIENT_LIST = ['chloechen@xcquant.com']
        
        # 抄送列表
        self.CC_LIST = ['chloechen@xcquant.com']
        
        # 文件路径（使用工作目录）
        self.WORKDIR = WORKDIR
        self.CSV_FILE = Path(WORKDIR) / "daily_predictions.csv"
        self.POSITIONS_FILE = Path(WORKDIR) / "positions.json"  # 持仓记录文件
        self.PNL_HISTORY_FILE = Path(WORKDIR) / "daily_pnl_history.csv"  # 历史盈亏记录文件
        # 实时数据路径
        self.KQ_DATA_DIR = "/home/intern0/qlib_data_recent"
        
        # 模型和标准化参数（延迟加载）
        self.model = None
        self.norm_params = None
        self._model_loaded = False
        
        print(f"[INIT] 邮件发送器初始化完成")
        print(f"[INFO] CSV文件: {self.CSV_FILE}")
        print(f"[INFO] 持仓文件: {self.POSITIONS_FILE}")
    
    def _ensure_qlib_initialized(self):
        """确保qlib只初始化一次"""
        if EmailSender._qlib_initialized:
            return True
        try:
            import qlib
            from qlib.constant import REG_CN
            # 尝试多个数据路径
            data_paths = [
                "/home/intern0/qlib_data_ipynb",
                "/home/intern0/qlib_data_recent",
                "/home/intern0/qlib_data",
            ]
            for data_path in data_paths:
                try:
                    qlib.init(provider_uri=data_path, region=REG_CN)
                    EmailSender._qlib_initialized = True
                    print(f"[INFO] Qlib已初始化（email_sender），使用数据路径: {data_path}")
                    return True
                except Exception as e:
                    print(f"[WARN] 尝试路径 {data_path} 失败: {e}")
                    continue
            print("[WARN] 所有数据路径都初始化失败")
            return False
        except Exception as e:
            print(f"[WARN] Qlib初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def read_signal_csv(self, target_date=None):
        """读取信号CSV文件（带真实性校验：当日且非空）
        
        参数:
            target_date: 目标日期，默认为None（使用今天）
        """
        try:
            if not self.CSV_FILE.exists():
                print(f"[WARN] CSV文件不存在: {self.CSV_FILE}")
                return None
            
            if target_date is None:
                target_date = datetime.now().date()
            elif isinstance(target_date, str):
                target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
            
            # 校验文件修改时间为目标日期（放宽检查，允许1天误差）
            mtime = datetime.fromtimestamp(self.CSV_FILE.stat().st_mtime)
            if abs((mtime.date() - target_date).days) > 1:
                print(f"[WARN] CSV非当日生成，mtime={mtime}")
                return None
            df = pd.read_csv(self.CSV_FILE)
            if df is None or len(df) == 0:
                print(f"[WARN] CSV为空: {self.CSV_FILE}")
                return None
            # 如果有timestamp列，校验为目标日期
            if 'timestamp' in df.columns:
                try:
                    ts0 = pd.to_datetime(df['timestamp'].iloc[0])
                    if ts0.date() != target_date:
                        print(f"[WARN] CSV时间戳非目标日期 {target_date}: {ts0.date()}")
                        # 放宽检查：允许使用目标日期的数据
                        df_filtered = df[pd.to_datetime(df['timestamp']).dt.date == target_date]
                        if len(df_filtered) > 0:
                            df = df_filtered
                            print(f"[INFO] 已过滤为 {target_date} 的数据: {len(df)} 条")
                        else:
                            print(f"[WARN] 未找到 {target_date} 的数据")
                            return None
                except Exception:
                    pass
            print(f"[OK] 读取信号成功: {len(df)} 条 (目标日期={target_date}, mtime={mtime})")
            return df
        except Exception as e:
            print(f"[ERROR] 读取CSV失败: {e}")
            return None
    
    def save_positions(self, df, buy_date=None):
        """保存持仓信息（买入时调用）
        
        参数:
            df: 买入信号DataFrame
            buy_date: 买入日期，格式 'YYYY-MM-DD'，默认为None（使用今天）
        """
        try:
            if buy_date is None:
                buy_date = datetime.now().strftime('%Y-%m-%d')
            elif isinstance(buy_date, (datetime, date)):
                buy_date = buy_date.strftime('%Y-%m-%d')
            elif isinstance(buy_date, str):
                pass  # 已经是字符串格式
            else:
                buy_date = str(buy_date)
            
            positions = {}
            for _, row in df.iterrows():
                code = row['code']
                positions[code] = {
                    'buy_date': buy_date,
                    'buy_price': float(row['price']),
                    'shares': int(row['shares']),
                    'code': code,
                    'rank': int(row['rank']),
                    'score': float(row['score'])
                }
            
            # 保存为JSON
            with open(self.POSITIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(positions, f, indent=2, ensure_ascii=False)
            
            print(f"[OK] 持仓信息已保存: {len(positions)} 个")
            return True
            
        except Exception as e:
            print(f"[ERROR] 保存持仓失败: {e}")
            return False
    
    def load_positions(self):
        """读取持仓信息"""
        try:
            if not self.POSITIONS_FILE.exists():
                print(f"[WARN] 持仓文件不存在: {self.POSITIONS_FILE}")
                return {}
            
            with open(self.POSITIONS_FILE, 'r', encoding='utf-8') as f:
                positions = json.load(f)
            
            print(f"[OK] 读取持仓信息: {len(positions)} 个")
            return positions
            
        except Exception as e:
            print(f"[ERROR] 读取持仓失败: {e}")
            return {}
    
    def _load_model_and_params(self):
        """延迟加载模型和标准化参数"""
        if self._model_loaded:
            return True
        
        try:
            from qlib.workflow import R
            from pathlib import Path
            import pickle
            
            # 首先尝试从当前recorder加载
            try:
                recorder = R.get_recorder()
                if recorder is not None:
                    self.model = recorder.load_object("model")
                    self.norm_params = recorder.load_object("norm_params")
                    print(f"✅ 从当前recorder加载模型和标准化参数成功")
                    self._model_loaded = True
                    return True
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
                            norm_params_path = recorder_path / "norm_params.pkl"
                            
                            if model_path.exists():
                                with open(model_path, 'rb') as f:
                                    self.model = pickle.load(f)
                                
                                # 尝试加载标准化参数
                                if norm_params_path.exists():
                                    with open(norm_params_path, 'rb') as f:
                                        self.norm_params = pickle.load(f)
                                else:
                                    # 尝试其他可能的名称
                                    for key in ("highfreq_norm", "processor_state", "highfreq_norm_params"):
                                        alt_path = recorder_path / f"{key}.pkl"
                                        if alt_path.exists():
                                            with open(alt_path, 'rb') as f:
                                                self.norm_params = pickle.load(f)
                                            break
                                
                                print(f"✅ 从最新recorder加载模型成功: {run_dir.name}")
                                if self.norm_params:
                                    print(f"✅ 标准化参数已加载")
                                self._model_loaded = True
                                return True
                        except Exception as e:
                            continue
            
            print(f"[WARN] 无法从recorder加载模型，可能需要先训练模型")
            return False
        except Exception as e:
            print(f"[ERROR] 加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _predict_realtime_signal(self, code, pred_time_str=None):
        """
        实时预测单个股票的信号值
        pred_time_str: 预测时间，格式 "YYYY-MM-DD HH:MM:SS"，如果为None则使用当前时间
        """
        try:
            # 确保qlib已初始化
            if not EmailSender._qlib_initialized and not self._ensure_qlib_initialized():
                print("[WARN] Qlib初始化失败，无法实时预测")
                return None

            try:
                from qlib.data import D
            except ImportError as e:
                print(f"[ERROR] 导入qlib失败: {e}")
                return None
            
            # 如果没有加载模型，先加载
            if not self._load_model_and_params():
                print(f"[WARN] {code} 模型未加载，无法实时预测")
                return None
            
            # 确定预测时间
            if pred_time_str is None:
                current_time = datetime.now()
                # 如果当前时间在10:46左右，使用10:46的时间
                if current_time.hour == 10 and current_time.minute >= 46:
                    pred_time_str = current_time.strftime("%Y-%m-%d 10:46:00")
                else:
                    pred_time_str = current_time.strftime("%Y-%m-%d %H:%M:00")
            else:
                pred_time_str = pred_time_str  # 使用指定时间
            
            # 1. 获取当前时刻的数据
            try:
                minute_data = D.features(
                    [code],
                    ["$open", "$high", "$low", "$close", "$volume"],
                    start_time=pred_time_str,
                    end_time=pred_time_str,
                    freq="1min"
                )
                if minute_data.empty:
                    print(f"[WARN] {code} 在 {pred_time_str} 无数据")
                    return None
                m = minute_data.iloc[0]
            except Exception as e:
                print(f"[WARN] {code} 获取当前数据失败: {e}")
                return None
            
            # 2. 获取前一日同一时刻的数据
            try:
                pred_date = datetime.strptime(pred_time_str, "%Y-%m-%d %H:%M:%S")
                prev_date = pred_date - timedelta(days=1)
                prev_time_str = prev_date.strftime("%Y-%m-%d %H:%M:%S")
                
                prev_day_data = D.features(
                    [code],
                    ["$open", "$high", "$low", "$close", "$volume"],
                    start_time=prev_time_str,
                    end_time=prev_time_str,
                    freq="1min"
                )
                if prev_day_data.empty:
                    # 如果前一日没有数据，使用当前数据近似
                    p = m
                else:
                    p = prev_day_data.iloc[0]
            except Exception as e:
                print(f"[WARN] {code} 获取前一日数据失败: {e}，使用当前数据近似")
                p = m
            
            # 3. 获取参考收盘价（用于归一化）
            try:
                ref_date_str = pred_date.strftime("%Y-%m-%d")
                prev_trading_date = (pred_date - timedelta(days=1)).strftime("%Y-%m-%d")
                # 获取前一交易日的收盘价
                ref_close_data = D.features(
                    [code],
                    ["$close"],
                    start_time=f"{prev_trading_date} 14:40:00",
                    end_time=f"{prev_trading_date} 14:40:00",
                    freq="1min"
                )
                if ref_close_data.empty:
                    ref_close = float(m['$close'])  # 使用当前收盘价
                else:
                    ref_close = float(ref_close_data.iloc[0, 0])
            except Exception:
                ref_close = float(m['$close'])
            
            # 4. 构造特征（与训练时一致）
            vwap_current = (m['$open'] + 2*m['$high'] + 2*m['$low'] + m['$close']) / 6
            vwap_prev = (p['$open'] + 2*p['$high'] + 2*p['$low'] + p['$close']) / 6
            
            # 比值归一化（与训练时一致）
            m_open_r = float(m['$open']) / ref_close
            m_high_r = float(m['$high']) / ref_close
            m_low_r = float(m['$low']) / ref_close
            m_close_r = float(m['$close']) / ref_close
            m_vwap_r = vwap_current / ref_close
            
            p_open_r = float(p['$open']) / ref_close
            p_high_r = float(p['$high']) / ref_close
            p_low_r = float(p['$low']) / ref_close
            p_close_r = float(p['$close']) / ref_close
            p_vwap_r = vwap_prev / ref_close
            
            feature_vector = [
                m_open_r, m_high_r, m_low_r, m_close_r, m_vwap_r,
                p_open_r, p_high_r, p_low_r, p_close_r, p_vwap_r,
                float(m['$volume']), float(p['$volume']), 1.0, 0.0, 0.0
            ]
            
            # 5. 标准化特征
            X = np.array([feature_vector], dtype=np.float32)
            X = self._normalize_features_realtime(X, self.norm_params)
            
            # 6. 模型预测
            try:
                prediction = self.model.predict(X)
                if hasattr(prediction, 'values'):
                    prediction = prediction.values
                prediction = np.asarray(prediction).reshape(-1)[0]
                return float(prediction)
            except Exception as e:
                print(f"[WARN] {code} 预测失败: {e}")
                return None
                
        except Exception as e:
            print(f"[ERROR] {code} 实时预测异常: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _normalize_features_realtime(self, X, norm_params):
        """对特征应用标准化（与训练时一致）"""
        # 价格类标准化：索引 0-9
        price_med = norm_params["price"]["med"]
        price_mad = norm_params["price"]["mad"]
        for i in range(10):
            X[:, i] = (X[:, i] - price_med) / price_mad
        X[:, :10] = np.clip(X[:, :10], -8.0, 8.0)
        
        # 成交量类标准化：索引 10-11
        volume_med = norm_params["volume"]["med"]
        volume_mad = norm_params["volume"]["mad"]
        for i in range(10, 12):
            X[:, i] = np.log1p(X[:, i])
            X[:, i] = (X[:, i] - volume_med) / volume_mad
        X[:, 10:12] = np.clip(X[:, 10:12], -8.0, 8.0)
        
        return X
    
    def get_current_price(self, code, target_time=None):
        """
        直接从C:/Users/ASUS/qlib_data_recent/features/<code>/data.csv读取对应时间close价。
        入参code可能是 sh.603072 / sz.000001 / SH600519 / SZ000001 等，需统一为 SHxxxxxx / SZxxxxxx。
        """
        try:
            # 统一代码为文件夹格式 SHxxxxxx / SZxxxxxx
            raw = str(code).strip()
            up = raw.upper()
            folder_code = up
            if up.startswith('SH.') and len(up) > 3:
                folder_code = 'SH' + up.split('.', 1)[1]
            elif up.startswith('SZ.') and len(up) > 3:
                folder_code = 'SZ' + up.split('.', 1)[1]
            elif up.startswith('SH') and len(up) > 2:
                folder_code = 'SH' + up[2:]
            elif up.startswith('SZ') and len(up) > 2:
                folder_code = 'SZ' + up[2:]
            # 其余情况直接用大写

            folder = Path(self.KQ_DATA_DIR) / "features" / folder_code
            csv_file = folder / "data.csv"
            if not csv_file.exists():
                print(f"[WARN] {folder_code} data.csv不存在 ({csv_file})")
                return None
            df = pd.read_csv(csv_file)
            # 统一解析 datetime，兼容 'YYYY/MM/DD HH:MM' 与 'YYYY-MM-DD HH:MM:SS' 等格式
            if 'datetime' not in df.columns or df.empty:
                return None
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.dropna(subset=['datetime'])
            
            if df.empty:
                return None
            # 构造分钟粒度列
            df['dt_min'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M')

            # 如果指定了target_time，使用精确时间匹配
            if target_time is not None:
                # target_time可能是字符串格式 "YYYY-MM-DD HH:MM:SS"
                if isinstance(target_time, str):
                    target_dt = pd.to_datetime(target_time)
                else:
                    target_dt = pd.to_datetime(target_time)
                
                # 精确匹配
                target_dt_str = target_dt.strftime('%Y-%m-%d %H:%M')
                row = df[df['dt_min'] == target_dt_str]
                if not row.empty:
                    return float(row.iloc[0]['close'])
                
                # 如果精确匹配失败，找最接近的时间点（10分钟内）
                time_diff = (df['datetime'] - target_dt).abs()
                closest_idx = time_diff.idxmin()
                if time_diff[closest_idx] <= pd.Timedelta(minutes=10):
                    return float(df.loc[closest_idx, 'close'])
                
                print(f"[WARN] {folder_code} 未找到 {target_time} 附近的价格")
                return None

            # 如果没有指定target_time，使用旧的10:46逻辑
            tgt_date = datetime.now().date()

            # 查找规则：
            # 1) 当天精确 10:46
            # 2) 当天 10:40-10:50 区间最后一条
            # 3) 向前最多 7 个交易日的 1) 或 2)

            def find_price_for_date(date_obj):
                date_str = date_obj.strftime('%Y-%m-%d')
                exact_key = f"{date_str} 10:46"
                row = df[df['dt_min'] == exact_key]
                if not row.empty:
                    return float(row.iloc[0]['close'])
                # 区间回退
                day_rows = df[df['dt_min'].str.startswith(date_str)]
                if not day_rows.empty:
                    window = day_rows[(day_rows['dt_min'].str.slice(11) >= '10:40') & (day_rows['dt_min'].str.slice(11) <= '10:50')]
                    if not window.empty:
                        return float(window.iloc[-1]['close'])
                return None

            # 当天
            price = find_price_for_date(tgt_date)
            if price is not None:
                return price

            # 向前回溯最多7天（仅工作日）
            for d in range(1, 8):
                check_date = tgt_date - timedelta(days=d)
                if check_date.weekday() >= 5:
                    continue
                price = find_price_for_date(check_date)
                if price is not None:
                    return price

            print(f"[WARN] {folder_code} 最近7个交易日未找到10:46或邻近窗口价格")
            return None
        except Exception as e:
            print(f"[ERROR] 获取{code} 价格失败: {e}")
            return None
    
    def generate_buy_email_html(self, df):
        """生成买入邮件HTML"""
        datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        count = len(df)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset='utf-8'>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                h2 {{ color: #1e88e5; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #1e88e5; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .rank {{ font-weight: bold; color: #1976d2; }}
                .code {{ font-weight: bold; color: #d32f2f; }}
                .price {{ color: #388e3c; }}
                .score {{ color: #f57c00; }}
            </style>
        </head>
        <body>
            <h2>📈 买入信号 - {datetime_str}</h2>
            <p>共 {count} 个买入信号</p>
            <table>
                <tr>
                    <th>排名</th>
                    <th>代码</th>
                    <th>信号分数</th>
                    <th>目标价格</th>
                    <th>建议股数</th>
                </tr>
        """
        
        for _, row in df.iterrows():
            html += f"""
                <tr>
                    <td class="rank">{row['rank']}</td>
                    <td class="code">{row['code']}</td>
                    <td class="score">{row['score']:.4f}</td>
                    <td class="price">{row['price']:.2f}</td>
                    <td>{row['shares']}</td>
                </tr>
            """
        
        html += """
            </table>
            <p style="color: #666; font-size: 12px;">本邮件由量化交易系统自动发送</p>
        </body>
        </html>
        """
        
        return html
    
    def save_daily_pnl(self, daily_pnl, total_principal, total_cost, total_stamp_tax, win_count, loss_count):
        """保存每日盈亏记录（追加模式，不覆盖）
        
        注意：只保存2025-11-28及之后的数据（最新模型训练结果）
        """
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            today_dt = pd.to_datetime(today)
            start_date = pd.to_datetime('2025-11-28')
            
            # 如果今天早于2025-11-28，不保存
            if today_dt < start_date:
                print(f"[WARN] 日期 {today} 早于2025-11-28，不保存历史记录（只记录最新模型结果）")
                return 0
            
            # 读取历史记录
            if self.PNL_HISTORY_FILE.exists():
                history_df = pd.read_csv(self.PNL_HISTORY_FILE)
                # 检查今天是否已有记录
                if len(history_df) > 0 and 'date' in history_df.columns:
                    history_df['date'] = pd.to_datetime(history_df['date'])
                    # 只保留2025-11-28及之后的数据
                    history_df = history_df[history_df['date'] >= start_date]
                    history_df = history_df.sort_values('date')
                    history_df['date'] = history_df['date'].dt.strftime('%Y-%m-%d')
                    
                    if today in history_df['date'].values:
                        # 更新今天的记录
                        history_df.loc[history_df['date'] == today, 'daily_pnl'] = daily_pnl
                        history_df.loc[history_df['date'] == today, 'total_cost'] = total_cost
                        history_df.loc[history_df['date'] == today, 'total_capital'] = total_principal
                        history_df.loc[history_df['date'] == today, 'profit_count'] = win_count
                        history_df.loc[history_df['date'] == today, 'loss_count'] = loss_count
                        # 重新计算累计盈亏
                        history_df = history_df.sort_values('date')
                        history_df['total_pnl'] = history_df['daily_pnl'].cumsum()
                    else:
                        # 追加新记录
                        last_total_pnl = history_df['total_pnl'].iloc[-1] if 'total_pnl' in history_df.columns and len(history_df) > 0 else 0
                        new_row = {
                            'date': today,
                            'daily_pnl': daily_pnl,
                            'total_pnl': last_total_pnl + daily_pnl,
                            'total_cost': total_cost,
                            'total_capital': total_principal,
                            'profit_count': win_count,
                            'loss_count': loss_count
                        }
                        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
                else:
                    # 文件格式不对，重新创建
                    new_row = {
                        'date': today,
                        'daily_pnl': daily_pnl,
                        'total_pnl': daily_pnl,
                        'total_cost': total_cost,
                        'total_capital': total_principal,
                        'profit_count': win_count,
                        'loss_count': loss_count
                    }
                    history_df = pd.DataFrame([new_row])
            else:
                # 首次创建
                new_row = {
                    'date': today,
                    'daily_pnl': daily_pnl,
                    'total_pnl': daily_pnl,
                    'total_cost': total_cost,
                    'total_capital': total_principal,
                    'profit_count': win_count,
                    'loss_count': loss_count
                }
                history_df = pd.DataFrame([new_row])
            
            # 保存
            history_df.to_csv(self.PNL_HISTORY_FILE, index=False, encoding='utf-8-sig')
            print(f"[OK] 每日盈亏记录已保存: {today}, 当日盈亏: {daily_pnl:.2f} 元")
            return history_df['total_pnl'].iloc[-1]  # 返回累计盈亏
            
        except Exception as e:
            print(f"[ERROR] 保存每日盈亏记录失败: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def get_historical_total_pnl(self):
        """获取历史累计盈亏"""
        try:
            if not self.PNL_HISTORY_FILE.exists():
                return 0
            
            history_df = pd.read_csv(self.PNL_HISTORY_FILE)
            if len(history_df) == 0 or 'total_pnl' not in history_df.columns:
                return 0
            
            # 返回最新的累计盈亏
            return float(history_df['total_pnl'].iloc[-1])
            
        except Exception as e:
            print(f"[WARN] 读取历史盈亏失败: {e}")
            return 0

    def _load_benchmark_from_history(self, df, initial_capital):
        """如果历史盈亏文件包含benchmark字段，则直接使用"""
        try:
            if 'benchmark_cumulative' in df.columns:
                series = pd.Series(df['benchmark_cumulative'].astype(float).values, index=df['date'])
                print("[INFO] 使用历史文件中的benchmark累计收益")
                return series
            if 'benchmark_return' in df.columns:
                returns = pd.Series(df['benchmark_return'].astype(float).fillna(0).values, index=df['date'])
                bench_series = ((1 + returns).cumprod() - 1) * initial_capital
                print("[INFO] 使用历史文件中的benchmark收益率")
                return bench_series
        except Exception as e:
            print(f"[WARN] 从历史盈亏文件读取benchmark失败: {e}")
        return None

    def _load_benchmark_from_recorder(self, df, initial_capital):
        """尝试从最近的recorder报告中提取等权隔夜收益benchmark"""
        try:
            mlruns_dir = Path(self.WORKDIR) / "mlruns"
            if not mlruns_dir.exists():
                mlruns_dir = Path("mlruns")
            if not mlruns_dir.exists():
                print("[WARN] mlruns目录不存在，无法从recorder加载benchmark")
                return None

            import pickle

            exp_dirs = sorted(
                [d for d in mlruns_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

            for exp_dir in exp_dirs[:5]:
                run_dirs = sorted(
                    [d for d in exp_dir.iterdir() if d.is_dir()],
                    key=lambda x: x.stat().st_mtime,
                    reverse=True,
                )
                for run_dir in run_dirs[:3]:
                    try:
                        report_path = run_dir / "artifacts" / "recorder" / "portfolio_analysis" / "report_normal_1min.pkl"
                        if not report_path.exists():
                            continue
                        with open(report_path, "rb") as f:
                            report = pickle.load(f)
                        if "bench" not in report.columns:
                            continue
                        
                        # 从分钟级benchmark数据中提取日频数据
                        bench_returns = report["bench"]
                        
                        # 将分钟级数据转换为日频：每天只取10:46时刻的数据（等权隔夜收益benchmark的特点）
                        bench_df = pd.DataFrame({
                            "datetime": bench_returns.index,
                            "bench_return": bench_returns.values
                        })
                        bench_df["datetime"] = pd.to_datetime(bench_df["datetime"])
                        bench_df["date"] = bench_df["datetime"].dt.normalize()
                        bench_df["hour"] = bench_df["datetime"].dt.hour
                        bench_df["minute"] = bench_df["datetime"].dt.minute
                        
                        # 只取10:46时刻的数据（等权隔夜收益benchmark在10:46记入）
                        bench_1046 = bench_df[(bench_df["hour"] == 10) & (bench_df["minute"] == 46)].copy()
                        if bench_1046.empty:
                            # 如果没有10:46数据，使用每天的平均值
                            daily_bench = bench_df.groupby("date")["bench_return"].mean()
                        else:
                            daily_bench = bench_1046.groupby("date")["bench_return"].first()
                        
                        # 过滤掉全为0的日期（非交易日）
                        daily_bench = daily_bench[daily_bench != 0]
                        if daily_bench.empty:
                            continue
                        
                        # 计算累计收益
                        bench_series = ((1 + daily_bench).cumprod() - 1) * initial_capital
                        bench_series.index = pd.to_datetime(bench_series.index)
                        
                        # 对齐到目标日期
                        target_index = pd.to_datetime(df["date"])
                        aligned = bench_series.reindex(target_index, method="ffill").fillna(method="bfill")
                        if aligned.isna().all():
                            continue
                        
                        print(f"[INFO] ✅ 从recorder {run_dir.name} 载入等权隔夜收益benchmark数据（{len(bench_series)}天）")
                        return pd.Series(aligned.values, index=df["date"])
                    except Exception as inner_e:
                        print(f"[WARN] 解析{run_dir} benchmark失败: {inner_e}")
                        continue
            print("[WARN] 未在recorder中找到可用的benchmark数据")
        except Exception as e:
            print(f"[WARN] 从recorder加载benchmark失败: {e}")
        return None

    def _calculate_equal_weight_overnight_benchmark(self, df, initial_capital):
        """计算等权隔夜收益benchmark（从CSV文件直接读取）"""
        try:
            import numpy as np
            from datetime import timedelta
            from pathlib import Path
            
            # 数据路径
            data_dir = Path("/home/intern0/qlib_data_recent/features")
            if not data_dir.exists():
                print(f"[WARN] 数据目录不存在: {data_dir}")
                return None
            
            # 获取股票池（从positions.json或CSV文件）
            stock_codes = []
            
            # 方法1：从positions.json获取
            positions = self.load_positions()
            if positions:
                stock_codes = list(positions.keys())
                print(f"[INFO] 从positions.json获取股票池: {len(stock_codes)} 支")
            
            # 方法2：如果positions为空，尝试从CSV文件获取
            if not stock_codes and self.CSV_FILE.exists():
                try:
                    df_csv = pd.read_csv(self.CSV_FILE)
                    if 'code' in df_csv.columns:
                        stock_codes = df_csv['code'].unique().tolist()
                        print(f"[INFO] 从CSV文件获取股票池: {len(stock_codes)} 支")
                except:
                    pass
            
            if not stock_codes:
                print("[WARN] 无法获取股票池，无法计算等权隔夜收益benchmark")
                return None
            
            # 限制股票数量（避免计算太慢）
            if len(stock_codes) > 500:
                stock_codes = stock_codes[:500]
                print(f"[INFO] 限制股票池为前500支")
            
            # 获取日期范围（只处理交易日）
            min_date = df["date"].min()
            max_date = df["date"].max()
            
            # 只处理交易日（排除周末）
            def is_trading_day(date_obj):
                """判断是否为交易日（排除周末）"""
                weekday = date_obj.weekday()  # 0=Monday, 6=Sunday
                return weekday < 5  # 周一到周五
            
            def get_previous_trading_day(date_obj):
                """获取前一个交易日（排除周末）"""
                prev_date = date_obj - timedelta(days=1)
                while not is_trading_day(prev_date):
                    prev_date = prev_date - timedelta(days=1)
                return prev_date
            
            # 只处理交易日
            all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
            target_dates = [d for d in all_dates if is_trading_day(d)]
            
            print(f"[INFO] 计算等权隔夜收益benchmark: {min_date} ~ {max_date}, {len(stock_codes)} 支股票")
            print(f"[INFO] 交易日数量: {len(target_dates)}")
            
            # 存储每日的等权隔夜收益
            daily_bench_returns = {}
            
            # 对每个目标日期，计算等权隔夜收益
            for target_date in target_dates:
                target_date_str = target_date.strftime("%Y-%m-%d")
                # 获取前一个交易日（排除周末）
                prev_date = get_previous_trading_day(target_date)
                prev_date_str = prev_date.strftime("%Y-%m-%d")
                
                print(f"[DEBUG] {target_date_str} 的前一个交易日: {prev_date_str}")
                
                stock_returns = []
                
                # 对每只股票计算隔夜收益
                for code in stock_codes:
                    try:
                        csv_path = data_dir / code / "data.csv"
                        if not csv_path.exists():
                            continue
                        
                        # 读取CSV文件
                        df_stock = pd.read_csv(csv_path)
                        df_stock['datetime'] = pd.to_datetime(df_stock['datetime'])
                        df_stock['date'] = df_stock['datetime'].dt.date
                        
                        # 获取前一天和当天的数据
                        prev_data = df_stock[df_stock['date'] == prev_date.date()]
                        curr_data = df_stock[df_stock['date'] == target_date.date()]
                        
                        if prev_data.empty or curr_data.empty:
                            continue
                        
                        # 计算前一天14:41-14:50的VWAP
                        prev_afternoon = prev_data[
                            (prev_data['datetime'].dt.hour == 14) & 
                            (prev_data['datetime'].dt.minute >= 41) & 
                            (prev_data['datetime'].dt.minute < 50)
                        ]
                        if prev_afternoon.empty:
                            continue
                        
                        tp_prev = (prev_afternoon['open'] + prev_afternoon['high'] + 
                                  prev_afternoon['low'] + prev_afternoon['close']) / 4.0
                        vol_prev = prev_afternoon['volume']
                        afternoon_vwap = (tp_prev * vol_prev).sum() / vol_prev.sum() if vol_prev.sum() > 0 else None
                        
                        if afternoon_vwap is None or afternoon_vwap <= 0:
                            continue
                        
                        # 计算当天10:11-10:20的VWAP
                        curr_morning = curr_data[
                            (curr_data['datetime'].dt.hour == 10) & 
                            (curr_data['datetime'].dt.minute >= 11) & 
                            (curr_data['datetime'].dt.minute < 20)
                        ]
                        if curr_morning.empty:
                            continue
                        
                        tp_curr = (curr_morning['open'] + curr_morning['high'] + 
                                   curr_morning['low'] + curr_morning['close']) / 4.0
                        vol_curr = curr_morning['volume']
                        morning_vwap = (tp_curr * vol_curr).sum() / vol_curr.sum() if vol_curr.sum() > 0 else None
                        
                        if morning_vwap is None or morning_vwap <= 0:
                            continue
                        
                        # 计算隔夜收益
                        overnight_return = (morning_vwap / afternoon_vwap) - 1.0
                        stock_returns.append(overnight_return)
                        
                    except Exception as e:
                        continue
                
                # 计算等权平均（只对有效数据）
                if stock_returns:
                    equal_weight_return = np.mean(stock_returns)
                    daily_bench_returns[target_date_str] = equal_weight_return
                    print(f"[INFO] {target_date_str}: {len(stock_returns)} 支股票，等权隔夜收益: {equal_weight_return:.6f}")
            
            if not daily_bench_returns:
                print("[WARN] 没有成功计算任何日期的等权隔夜收益")
                return None
            
            # 转换为Series并计算累计收益
            bench_returns_series = pd.Series(daily_bench_returns)
            bench_returns_series.index = pd.to_datetime(bench_returns_series.index)
            
            # 对异常值进行裁剪（1%和99%分位数）
            if len(bench_returns_series) > 0:
                lower_bound = bench_returns_series.quantile(0.01)
                upper_bound = bench_returns_series.quantile(0.99)
                bench_returns_series = bench_returns_series.clip(lower=lower_bound, upper=upper_bound)
            
            # 计算累计收益
            bench_series = ((1 + bench_returns_series).cumprod() - 1) * initial_capital
            
            # 对齐到目标日期
            target_index = pd.to_datetime(df["date"])
            aligned = bench_series.reindex(target_index, method="ffill").bfill()
            if aligned.isna().all():
                return None
            
            print(f"[INFO] ✅ 成功计算等权隔夜收益benchmark（{len(bench_series)}天）")
            return pd.Series(aligned.values, index=df["date"])
            
        except Exception as e:
            print(f"[WARN] 计算等权隔夜收益benchmark失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_benchmark_curve(self, df, initial_capital):
        """综合尝试多个来源获取等权隔夜收益benchmark累计收益"""
        for loader in (
            self._load_benchmark_from_history,
            self._load_benchmark_from_recorder,
            self._calculate_equal_weight_overnight_benchmark,
        ):
            series = loader(df, initial_capital)
            if series is not None and not series.empty:
                return series
        return None
    
    def generate_pnl_chart_base64(self, max_days=30):
        """
        生成历史盈亏图表并转换为 base64 编码（用于嵌入邮件）
        
        Args:
            max_days: 最多显示最近多少天的数据
        
        Returns:
            base64 编码的图片字符串，如果失败返回 None
        """
        try:
            if not self.PNL_HISTORY_FILE.exists():
                print("[WARN] 历史盈亏文件不存在，无法生成图表")
                return None
            
            # 读取历史数据
            df = pd.read_csv(self.PNL_HISTORY_FILE)
            if len(df) == 0:
                print("[WARN] 历史盈亏数据为空，无法生成图表")
                return None
            
            # 处理日期
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 只保留2025-11-28及之后的数据（最新模型训练结果）
            start_date = pd.to_datetime('2025-11-28')
            df = df[df['date'] >= start_date]
            
            if len(df) == 0:
                print("[WARN] 2025-11-28之后没有数据，无法生成图表")
                return None
            
            # 只取最近 max_days 天的数据
            if len(df) > max_days:
                df = df.tail(max_days)
            
            # 设置中文字体（尝试多种方式）
            import matplotlib.font_manager as fm
            # 尝试找到可用的中文字体
            chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial Unicode MS']
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            font_found = False
            for font_name in chinese_fonts:
                if font_name in available_fonts:
                    plt.rcParams['font.sans-serif'] = [font_name]
                    font_found = True
                    print(f"[INFO] 使用中文字体: {font_name}")
                    break
            
            if not font_found:
                # 如果没有找到中文字体，尝试使用DejaVu Sans（至少能显示英文）
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
                print("[WARN] 未找到中文字体，使用DejaVu Sans（可能无法显示中文）")
            
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建图表（2x2布局，适合邮件显示）
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('交易盈亏历史曲线 (自2025-11-28起)', fontsize=16, fontweight='bold')
            
            # 1. 累计盈亏曲线（添加benchmark对比）
            ax1 = axes[0, 0]
            initial_capital = df['total_capital'].iloc[0] if 'total_capital' in df.columns and len(df) > 0 else 1000000
            if initial_capital <= 0:
                initial_capital = 1000000
            benchmark_series = self._load_benchmark_curve(df, initial_capital)
            
            # 调试信息：检查benchmark是否加载成功
            if benchmark_series is None or len(benchmark_series) == 0:
                print("[WARN] Benchmark未加载成功，尝试使用零基准线")
                # 如果benchmark加载失败，使用零基准线（初始本金）
                benchmark_series = pd.Series([0.0] * len(df), index=df['date'])
            else:
                print(f"[INFO] Benchmark加载成功，数据点: {len(benchmark_series)}")

            # 绘制累计盈亏
            line1 = ax1.plot(df['date'], df['total_pnl'], marker='o', linewidth=2, markersize=5, 
                     color='#2E86AB', label='Strategy PnL', zorder=3)
            ax1.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
            
            # 绘制benchmark（确保显示）
            lines = line1
            # 确保benchmark数据与df的日期对齐
            try:
                if benchmark_series.index.equals(df['date']):
                    # 索引完全匹配
                    benchmark_aligned = benchmark_series
                else:
                    # 尝试重新索引对齐
                    benchmark_aligned = benchmark_series.reindex(df['date'], method='ffill').bfill()
                    if benchmark_aligned.isna().all() or len(benchmark_aligned) == 0:
                        # 如果对齐失败，使用零基准线
                        benchmark_aligned = pd.Series([0.0] * len(df), index=df['date'])
                        print("[WARN] Benchmark对齐失败，使用零基准线")
            except Exception as e:
                print(f"[WARN] Benchmark对齐过程出错: {e}，使用零基准线")
                benchmark_aligned = pd.Series([0.0] * len(df), index=df['date'])
            
            line2 = ax1.plot(
                df['date'],
                benchmark_aligned.values,
                linewidth=2,
                color='orange',
                label='Benchmark (等权隔夜收益)',
                linestyle='--',
                alpha=0.8,
                zorder=2,
            )
            lines = line1 + line2
            
            # 填充区域（只在策略PnL下方）
            ax1.fill_between(df['date'], 0, df['total_pnl'], 
                             where=(df['total_pnl'] >= 0), alpha=0.3, color='green', label='Profit Zone')
            ax1.fill_between(df['date'], 0, df['total_pnl'], 
                             where=(df['total_pnl'] < 0), alpha=0.3, color='red', label='Loss Zone')
            
            ax1.set_xlabel('Date', fontsize=10)
            ax1.set_ylabel('Cumulative PnL (Yuan)', fontsize=10)
            ax1.set_title('Historical PnL Curve (Last 30 Days)', fontsize=12, fontweight='bold')
            
            # 更新图例
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, fontsize=8, loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45, labelsize=8)
            ax1.tick_params(axis='y', labelsize=8)
            
            # 标注最新值
            if len(df) > 0:
                latest_date = df['date'].iloc[-1]
                latest_pnl = df['total_pnl'].iloc[-1]
                ax1.annotate(f'{latest_pnl:.0f}', (latest_date, latest_pnl), 
                            textcoords="offset points", xytext=(0,10), ha='center', 
                            fontsize=9, fontweight='bold', 
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            # 2. 每日盈亏柱状图（只显示交易日）
            ax2 = axes[0, 1]
            # 过滤非交易日（周末）
            def is_trading_day(date_obj):
                """判断是否为交易日（排除周末）"""
                weekday = date_obj.weekday()  # 0=Monday, 6=Sunday
                return weekday < 5  # 周一到周五
            
            df_trading = df[df['date'].apply(is_trading_day)].copy()
            if len(df_trading) == 0:
                df_trading = df  # 如果没有交易日数据，使用全部数据
            
            colors = ['#4CAF50' if pnl >= 0 else '#F44336' for pnl in df_trading['daily_pnl']]
            bars = ax2.bar(df_trading['date'], df_trading['daily_pnl'], color=colors, alpha=0.7, 
                          edgecolor='black', linewidth=0.5)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.set_xlabel('Date', fontsize=10)
            ax2.set_ylabel('Daily PnL (Yuan)', fontsize=10)
            ax2.set_title('Daily PnL Bar Chart', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.tick_params(axis='x', rotation=45, labelsize=8)
            ax2.tick_params(axis='y', labelsize=8)
            
            # 标注最新值
            if len(bars) > 0:
                latest_bar = bars[-1]
                latest_daily_pnl = df_trading['daily_pnl'].iloc[-1]
                height = latest_bar.get_height()
                ax2.text(latest_bar.get_x() + latest_bar.get_width()/2., height,
                        f'{latest_daily_pnl:.0f}',
                        ha='center', va='bottom' if height >= 0 else 'top', 
                        fontsize=9, fontweight='bold')
            
            # 3. 盈亏比例饼图（最近一次）
            ax3 = axes[1, 0]
            if len(df) > 0:
                latest = df.iloc[-1]
                total_trades = latest['profit_count'] + latest['loss_count']
                if total_trades > 0:
                    profit_pct = latest['profit_count'] / total_trades * 100
                    loss_pct = latest['loss_count'] / total_trades * 100
                    
                    sizes = [latest['profit_count'], latest['loss_count']]
                    labels = [f'Profit {latest["profit_count"]} ({profit_pct:.1f}%)', 
                             f'Loss {latest["loss_count"]} ({loss_pct:.1f}%)']
                    colors_pie = ['#4CAF50', '#F44336']
                    explode = (0.05, 0.05)
                    
                    ax3.pie(sizes, explode=explode, labels=labels, colors=colors_pie, 
                           autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 9})
                    ax3.set_title(f'Performance Distribution ({latest["date"].strftime("%Y-%m-%d")})', 
                                 fontsize=12, fontweight='bold')
                else:
                    ax3.text(0.5, 0.5, 'No Trading Data', ha='center', va='center', fontsize=12)
                    ax3.set_title('Performance Distribution', fontsize=12, fontweight='bold')
            
            # 4. 累计盈亏趋势 + 交易成本
            ax4 = axes[1, 1]
            ax4_twin = ax4.twinx()
            
            # 累计盈亏
            line1 = ax4.plot(df['date'], df['total_pnl'], marker='o', linewidth=2, 
                            markersize=5, color='#2E86AB', label='Cumulative PnL')
            ax4.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
            ax4.set_xlabel('Date', fontsize=10)
            ax4.set_ylabel('Cumulative PnL (Yuan)', fontsize=10, color='#2E86AB')
            ax4.tick_params(axis='y', labelcolor='#2E86AB', labelsize=8)
            ax4.tick_params(axis='x', rotation=45, labelsize=8)
            
            # 累计交易成本
            if 'total_cost' in df.columns and df['total_cost'].sum() > 0:
                cumulative_cost = df['total_cost'].cumsum()
                line2 = ax4_twin.plot(df['date'], cumulative_cost, marker='s', linewidth=2, 
                                     markersize=4, color='#FF6B35', label='Cumulative Cost', linestyle='--')
                ax4_twin.set_ylabel('Cumulative Cost (Yuan)', fontsize=10, color='#FF6B35')
                ax4_twin.tick_params(axis='y', labelcolor='#FF6B35', labelsize=8)
                lines = line1 + line2
            else:
                lines = line1
            
            ax4.set_title('Cumulative PnL vs Trading Cost', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # 合并图例
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper left', fontsize=8)
            
            plt.tight_layout()
            
            # 将图表转换为 base64 编码
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            buffer.close()
            plt.close(fig)  # 关闭图表释放内存
            
            print(f"[OK] 历史盈亏图表已生成（最近 {len(df)} 天数据）")
            return image_base64
            
        except Exception as e:
            print(f"[ERROR] 生成历史盈亏图表失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_sell_email_html(self, df_with_profit, total_principal_all=None):
        """生成卖出邮件HTML（包含收益信息）
        
        Args:
            df_with_profit: 成功交易的股票收益数据
            total_principal_all: 所有持仓的总本金（包括无法获取卖出价格的股票）
        """
        # 计算汇总统计
        daily_profit = df_with_profit['profit'].sum()  # 当日盈亏（已扣除交易成本）
        total_principal_traded = df_with_profit['principal'].sum()  # 成功交易股票的本金
        # 使用所有持仓的本金（如果提供），否则使用成功交易的本金
        total_principal = total_principal_all if total_principal_all is not None else total_principal_traded
        total_cost = df_with_profit['total_cost'].sum() if 'total_cost' in df_with_profit.columns else 0  # 总交易成本
        total_stamp_tax = df_with_profit['stamp_tax'].sum() if 'stamp_tax' in df_with_profit.columns else 0  # 总印花税
        # 使用加权平均收益率：总盈亏 / 总本金（更准确）
        daily_profit_rate = (daily_profit / total_principal) if total_principal > 0 else 0
        win_count = len(df_with_profit[df_with_profit['profit'] > 0])
        loss_count = len(df_with_profit[df_with_profit['profit'] < 0])
        
        # 保存当日盈亏记录（使用所有持仓的本金）
        historical_total_pnl = self.save_daily_pnl(
            daily_profit, total_principal, total_cost, total_stamp_tax, win_count, loss_count
        )
        
        # 分析信号值与盈亏的关系
        df_win = df_with_profit[df_with_profit['profit'] > 0]  # 赚钱的品种
        df_loss = df_with_profit[df_with_profit['profit'] < 0]  # 亏钱的品种
        
        # 计算信号统计
        signal_analysis = ""
        if len(df_win) > 0 and len(df_loss) > 0:
            # 赚钱品种的信号统计
            win_avg_buy_signal = df_win['buy_signal'].mean() if 'buy_signal' in df_win.columns else 0
            win_avg_sell_signal = df_win['sell_signal'].mean() if 'sell_signal' in df_win.columns else 0
            win_signal_change = win_avg_sell_signal - win_avg_buy_signal
            
            # 亏钱品种的信号统计
            loss_avg_buy_signal = df_loss['buy_signal'].mean() if 'buy_signal' in df_loss.columns else 0
            loss_avg_sell_signal = df_loss['sell_signal'].mean() if 'sell_signal' in df_loss.columns else 0
            loss_signal_change = loss_avg_sell_signal - loss_avg_buy_signal
            
            signal_analysis = f"""
                <div class="summary-item" style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
                    <h4 style="color: #1976d2; margin-bottom: 10px;">📊 信号强弱分析</h4>
                    <div style="margin-left: 10px;">
                        <p style="color: #e53935; font-weight: bold;">💰 赚钱品种 ({len(df_win)}个):</p>
                        <ul style="margin: 5px 0; padding-left: 20px;">
                            <li>平均买入信号: <strong>{win_avg_buy_signal:.4f}</strong></li>
                            <li>平均卖出信号: <strong>{win_avg_sell_signal:.4f}</strong></li>
                            <li>信号变化: <strong style="color: {'#388e3c' if win_signal_change >= 0 else '#e53935'}">{win_signal_change:+.4f}</strong></li>
                        </ul>
                        <p style="color: #388e3c; font-weight: bold; margin-top: 10px;">📉 亏钱品种 ({len(df_loss)}个):</p>
                        <ul style="margin: 5px 0; padding-left: 20px;">
                            <li>平均买入信号: <strong>{loss_avg_buy_signal:.4f}</strong></li>
                            <li>平均卖出信号: <strong>{loss_avg_sell_signal:.4f}</strong></li>
                            <li>信号变化: <strong style="color: {'#388e3c' if loss_signal_change >= 0 else '#e53935'}">{loss_signal_change:+.4f}</strong></li>
                        </ul>
                    </div>
                </div>
            """
        
        datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        count = len(df_with_profit)
        daily_color = '#e53935' if daily_profit >= 0 else '#388e3c'  # 当日盈亏颜色
        daily_rate_color = '#e53935' if daily_profit_rate >= 0 else '#388e3c'  # 当日收益率颜色
        historical_color = '#e53935' if historical_total_pnl >= 0 else '#388e3c'  # 历史累计盈亏颜色
        
        # 生成历史盈亏图表（base64编码）
        chart_image_base64 = self.generate_pnl_chart_base64(max_days=30)
        chart_html = ""
        if chart_image_base64:
            chart_html = f"""
            <div style="margin: 20px 0; padding: 15px; background-color: #ffffff; border: 1px solid #ddd; border-radius: 5px;">
                <h3 style="color: #1976d2; margin-bottom: 15px;">📈 历史盈亏曲线（最近30天）</h3>
                <img src="data:image/png;base64,{chart_image_base64}" alt="历史盈亏曲线" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 3px;" />
                <p style="color: #666; font-size: 11px; margin-top: 10px; text-align: center;">图表包含：累计盈亏曲线、每日盈亏柱状图、盈亏分布饼图、累计盈亏vs交易成本</p>
            </div>
            """
        else:
            chart_html = """
            <div style="margin: 20px 0; padding: 15px; background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 5px;">
                <p style="color: #856404; font-size: 12px;">⚠️ 历史盈亏图表生成失败，可能因为数据不足或文件不存在</p>
            </div>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset='utf-8'>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                h2 {{ color: #e53935; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .summary-item {{ margin: 5px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #e53935; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .rank {{ font-weight: bold; color: #1976d2; }}
                .code {{ font-weight: bold; color: #d32f2f; }}
                .price {{ color: #388e3c; }}
                .profit-positive {{ color: #e53935; font-weight: bold; }}  /* 赚钱用红色 */
                .profit-negative {{ color: #388e3c; font-weight: bold; }}  /* 亏钱用绿色 */
                .profit-rate-positive {{ color: #e53935; }}  /* 赚钱用红色 */
                .profit-rate-negative {{ color: #388e3c; }}  /* 亏钱用绿色 */
            </style>
        </head>
        <body>
            <h2>📉 卖出信号 - {datetime_str}</h2>
            <div class="summary">
                <h3>📊 当日收益汇总（已扣除交易成本）</h3>
                <div class="summary-item">总本金: <span style="color: #1976d2; font-weight: bold;">{total_principal:.2f} 元</span></div>
                <div class="summary-item">总交易成本: <span style="color: #f57c00; font-weight: bold;">{total_cost:.2f} 元</span> (其中印花税: <span style="color: #e53935; font-weight: bold;">{total_stamp_tax:.2f} 元</span>)</div>
                <div class="summary-item" style="border-top: 2px solid #ddd; padding-top: 10px; margin-top: 10px;">
                    <strong>当日盈亏:</strong> <span style="color: {daily_color}; font-weight: bold; font-size: 16px;">{daily_profit:+.2f} 元</span>
                </div>
                <div class="summary-item">
                    <strong>当日收益率:</strong> <span style="color: {daily_rate_color}; font-weight: bold;">{daily_profit_rate*100:+.2f}%</span>
                </div>
                <div class="summary-item" style="border-top: 2px solid #ddd; padding-top: 10px; margin-top: 10px; background-color: #fff3cd; padding: 10px; border-radius: 5px;">
                    <strong style="font-size: 14px;">📈 历史累计盈亏（自2025-11-28起，最新模型结果）:</strong> 
                    <span style="color: {historical_color}; font-weight: bold; font-size: 18px;">{historical_total_pnl:+.2f} 元</span>
                </div>
                <div class="summary-item">盈利数量: <span style="color: #e53935; font-weight: bold;">{win_count}</span> | 亏损数量: <span style="color: #388e3c; font-weight: bold;">{loss_count}</span></div>
                {signal_analysis}
            </div>
            {chart_html}
            <p>共 {count} 个卖出信号</p>
            <table>
                <tr>
                    <th>排名</th>
                    <th>代码</th>
                    <th>买入价格</th>
                    <th>卖出价格</th>
                    <th>股数</th>
                    <th>买入信号</th>
                    <th>卖出信号</th>
                    <th>本金</th>
                    <th>交易成本</th>
                    <th>收益率</th>
                    <th>盈亏金额</th>
                </tr>
        """
        
        for _, row in df_with_profit.iterrows():
            profit_class = 'profit-positive' if row['profit'] >= 0 else 'profit-negative'
            rate_class = 'profit-rate-positive' if row['profit_rate'] >= 0 else 'profit-rate-negative'
            
            cost_display = row.get('total_cost', 0)
            buy_signal = row.get('buy_signal', 0.0)
            sell_signal = row.get('sell_signal', 0.0)
            # 信号值变化：绿色表示上涨，红色表示下跌
            signal_change = sell_signal - buy_signal
            signal_color = '#388e3c' if signal_change >= 0 else '#e53935'
            
            html += f"""
                <tr>
                    <td class="rank">{row['rank']}</td>
                    <td class="code">{row['code']}</td>
                    <td class="price">{row['buy_price']:.2f}</td>
                    <td class="price">{row['sell_price']:.2f}</td>
                    <td>{int(row['shares'])}</td>
                    <td style="color: #1976d2; font-weight: bold;">{buy_signal:.4f}</td>
                    <td style="color: {signal_color}; font-weight: bold;">{sell_signal:.4f}</td>
                    <td class="price">{row['principal']:.2f}</td>
                    <td style="color: #f57c00; font-size: 11px;">{cost_display:.2f}</td>
                    <td class="{rate_class}">{row['profit_rate']*100:+.2f}%</td>
                    <td class="{profit_class}">{row['profit']:+.2f}</td>
                </tr>
            """
        
        html += """
            </table>
            <p style="color: #666; font-size: 12px;">本邮件由量化交易系统自动发送</p>
        </body>
        </html>
        """
        
        return html
    
    def send_email(self, subject, html_content, signal_type='buy'):
        """发送邮件"""
        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.EMAIL_ADDRESS
            msg['To'] = ', '.join(self.RECIPIENT_LIST)  # 主要收件人
            msg['Cc'] = ', '.join(self.CC_LIST) if self.CC_LIST else ''  # 抄送列表
            msg['Subject'] = Header(subject, 'utf-8')
            
            # 添加HTML内容
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
            # 构建所有收件人列表（包含收件人+抄送）
            all_recipients = self.RECIPIENT_LIST + self.CC_LIST
            # 去重
            all_recipients = list(dict.fromkeys(all_recipients))
            
            # 发送邮件（使用SSL，465端口）
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL('smtp.feishu.cn', 465, context=context) as server:
                server.login(self.EMAIL_ADDRESS, self.EMAIL_PASSWORD)
                server.sendmail(self.EMAIL_ADDRESS, all_recipients, msg.as_string())
            
            print(f"[OK] {signal_type}邮件发送成功: {subject}")
            print(f"[INFO] 收件人: {len(self.RECIPIENT_LIST)}人, 抄送: {len(self.CC_LIST)}人")
            return True
            
        except Exception as e:
            print(f"[ERROR] 邮件发送失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def send_buy_email(self, target_date=None):
        """发送买入邮件（15:40，延后一小时）
        
        参数:
            target_date: 目标日期，格式 'YYYY-MM-DD'，默认为None（使用今天）
        """
        if target_date is None:
            target_date = datetime.now().date()
        else:
            if isinstance(target_date, str):
                target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        
        print(f"[INFO] 发送 {target_date} 的买入邮件...")
        
        df = self.read_signal_csv(target_date=target_date)
        if df is None:
            print("[ABORT] 未检测到当日真实预测CSV，取消发送买入邮件。")
            return
        if len(df) == 0:
            print("[WARN] CSV文件为空，取消发送")
            return
        buy_signals = df.sort_values('rank').head(50)
        html = self.generate_buy_email_html(buy_signals)
        subject = f"买入信号 - {target_date.strftime('%Y-%m-%d')}"
        success = self.send_email(subject, html, 'buy')
        if success:
            self.save_positions(buy_signals, buy_date=target_date)
    
    def send_sell_email(self, target_date=None):
        """发送卖出邮件（11:46发送，仍使用10:46价格逻辑）
        
        参数:
            target_date: 目标日期，格式 'YYYY-MM-DD'，默认为None（使用今天）
        """
        if target_date is None:
            target_date = datetime.now().date()
        else:
            if isinstance(target_date, str):
                target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        
        print(f"[INFO] 发送 {target_date} 的卖出邮件...")
        
        # 读取持仓信息
        positions = self.load_positions()
        if len(positions) == 0:
            print("[WARN] 没有持仓信息，无法计算收益")
            return
        
        # 读取10:46的预测文件（卖出信号）
        print(f"[INFO] 📖 读取10:46卖出预测文件...")
        sell_csv_file = Path(self.WORKDIR) / "daily_predictions_sell.csv"
        current_signals = {}
        signal_data_date = None
        
        try:
            if sell_csv_file.exists():
                df_signals = pd.read_csv(sell_csv_file)
                # 检查CSV文件的时间戳
                if 'timestamp' in df_signals.columns:
                    df_signals['timestamp'] = pd.to_datetime(df_signals['timestamp'])
                    
                    # 优先使用目标日期的数据
                    target_data = df_signals[df_signals['timestamp'].dt.date == target_date]
                    if len(target_data) > 0:
                        df_signals = target_data
                        signal_data_date = target_date
                        print(f"[INFO] ✅ 使用目标日期 {target_date} 的10:46预测数据")
                    else:
                        # 使用最新的一条记录
                        latest_time = df_signals['timestamp'].max()
                        latest_date = latest_time.date()
                        df_signals = df_signals[df_signals['timestamp'] == latest_time]
                        signal_data_date = latest_date
                        if latest_date != target_date:
                            print(f"[WARN] ⚠️ 未找到 {target_date} 的10:46数据，使用历史数据: {latest_date}")
                        else:
                            print(f"[INFO] 使用最新10:46数据: {latest_date}")
                
                for _, row in df_signals.iterrows():
                    code = row['code']
                    current_signals[code] = float(row['score'])
                print(f"[INFO] 读取卖出信号值: {len(current_signals)} 个")
                
                # 调试信息：显示部分信号值
                if len(current_signals) > 0:
                    sample_codes = list(current_signals.keys())[:3]
                    sample_values = {c: current_signals[c] for c in sample_codes}
                    print(f"[DEBUG] 样本信号值: {sample_values}")
            else:
                print(f"[WARN] 卖出预测文件不存在: {sell_csv_file}")
        except Exception as e:
            print(f"[WARN] 读取卖出预测文件失败: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"[INFO] 开始计算收益，持仓数量: {len(positions)}")
        
        # 先计算所有持仓的总本金（包括无法获取卖出价格的股票）
        total_principal_all = 0
        for code, position in positions.items():
            buy_price = position['buy_price']
            shares = position['shares']
            buy_amount = buy_price * shares
            buy_commission = max(buy_amount * 0.0003, 5.0)
            buy_cost = buy_amount + buy_commission
            total_principal_all += buy_cost
        
        print(f"[INFO] 总持仓本金（所有股票）: {total_principal_all:.2f} 元")
        
        # 准备收益数据
        profit_data = []
        
        for code, position in positions.items():
            buy_price = position['buy_price']
            shares = position['shares']
            buy_signal = position.get('score', 0.0)  # 买入时的信号值
            sell_signal = current_signals.get(code, None)  # 卖出时的信号值（从10:46预测文件读取）
            
            # 如果卖出信号不存在（股票不在预测文件中），使用买入信号作为默认值
            if sell_signal is None:
                sell_signal = buy_signal  # 使用买入信号作为默认值
                print(f"[WARN] {code} 未在10:46预测文件中找到，使用买入信号值: {buy_signal:.6f}")
            else:
                # 显示信号变化（只打印前3个）
                if len(profit_data) < 3:
                    signal_change = sell_signal - buy_signal
                    print(f"[INFO] {code} 买入信号: {buy_signal:.6f}, 卖出信号: {sell_signal:.6f}, 变化: {signal_change:+.6f}")
            
            # 获取目标日期10:46的价格（卖出时间点）
            sell_time_str = f"{target_date} 10:46:00"
            sell_price = self.get_current_price(code, target_time=sell_time_str)
            if sell_price is None:
                print(f"[WARN] {code} 无法获取 {sell_time_str} 价格，跳过")
                continue
            
            # A股交易成本配置
            BUY_COMMISSION_RATE = 0.0003  # 买入佣金：万分之3
            SELL_COMMISSION_RATE = 0.0003  # 卖出佣金：万分之3
            STAMP_TAX_RATE = 0.0005  # 印花税：万分之5（只在卖出时收取，这是大头！）
            MIN_COMMISSION = 5.0  # 最低佣金：5元
            
            # 计算成交金额
            buy_amount = buy_price * shares  # 买入金额
            sell_amount = sell_price * shares  # 卖出金额
            
            # ====== 本金计算说明 ======
            # 本金 = 买入金额 + 买入佣金
            # 买入佣金 = max(买入金额 * 0.0003, 5.0)  # 万分之3，最低5元
            # 例如：买入价10元，200股，买入金额=2000元，佣金=max(2000*0.0003, 5)=max(0.6, 5)=5元
            # 本金 = 2000 + 5 = 2005元
            # ======
            
            # 计算买入成本（买入价 + 买入佣金）
            buy_commission = max(buy_amount * BUY_COMMISSION_RATE, MIN_COMMISSION)
            buy_cost = buy_amount + buy_commission  # 这就是本金（实际投入的资金）
            
            # 计算卖出收入（卖出价 - 卖出佣金 - 印花税）
            sell_commission = max(sell_amount * SELL_COMMISSION_RATE, MIN_COMMISSION)
            stamp_tax = sell_amount * STAMP_TAX_RATE  # 印花税（卖出时收取，大头！）
            sell_income = sell_amount - sell_commission - stamp_tax
            
            # 实际盈亏（扣除所有交易成本）
            # 盈亏 = 卖出收入 - 买入成本（本金）
            profit = sell_income - buy_cost
            profit_rate = (profit / buy_cost * 100) if buy_cost > 0 else 0  # 实际收益率 = 盈亏 / 本金
            principal = buy_cost  # 实际投入本金（含买入佣金）
            
            # 交易成本明细（用于显示）
            total_cost = buy_commission + sell_commission + stamp_tax
            
            profit_data.append({
                'code': code,
                'rank': position.get('rank', 0),
                'buy_price': buy_price,
                'sell_price': sell_price,
                'shares': shares,
                'principal': principal,  # 实际投入本金（含买入佣金）
                'profit': profit,  # 实际盈亏（已扣除所有交易成本）
                'profit_rate': profit_rate / 100,  # 转换为小数（用于HTML显示）
                'buy_date': position.get('buy_date', ''),
                'total_cost': total_cost,  # 总交易成本
                'stamp_tax': stamp_tax,  # 印花税
                'buy_commission': buy_commission,
                'sell_commission': sell_commission,
                'buy_signal': buy_signal,  # 买入时的信号值
                'sell_signal': sell_signal if sell_signal is not None else 0.0  # 卖出时的信号值
            })
        
        if len(profit_data) == 0:
            print("[WARN] 没有可用的收益数据")
            return
        
        # 转换为DataFrame
        df_with_profit = pd.DataFrame(profit_data)
        df_with_profit = df_with_profit.sort_values('rank')
        
        # 生成HTML（传入所有持仓的本金）
        html = self.generate_sell_email_html(df_with_profit, total_principal_all)
        
        # 发送邮件
        subject = f"卖出信号 - {datetime.now().strftime('%Y-%m-%d')}"
        self.send_email(subject, html, 'sell')
        
        # 打印收益汇总
        daily_profit = df_with_profit['profit'].sum()
        total_principal_traded = df_with_profit['principal'].sum()
        # 使用所有持仓的本金（如果计算了）
        total_principal = total_principal_all if 'total_principal_all' in locals() else total_principal_traded
        # 使用加权平均收益率：总盈亏 / 总本金
        avg_rate = (daily_profit / total_principal * 100) if total_principal > 0 else 0
        
        # 获取历史累计盈亏
        historical_total_pnl = self.get_historical_total_pnl()
        
        # 判断盈亏状态（赚钱红色🟢改为🔴，亏钱绿色🔴改为🟢）
        profit_status = "盈利" if daily_profit >= 0 else "亏损"
        profit_color = "🔴" if daily_profit >= 0 else "🟢"  # 赚钱红色，亏钱绿色
        
        # 信号值分析
        df_win = df_with_profit[df_with_profit['profit'] > 0]
        df_loss = df_with_profit[df_with_profit['profit'] < 0]
        
        total_cost = df_with_profit['total_cost'].sum() if 'total_cost' in df_with_profit.columns else 0
        total_stamp_tax = df_with_profit['stamp_tax'].sum() if 'stamp_tax' in df_with_profit.columns else 0
        
        print(f"[SUMMARY] {profit_color} 当日盈亏: {daily_profit:+.2f} 元 ({profit_status}), 平均收益率: {avg_rate:+.2f}%")
        print(f"[SUMMARY] 📈 历史累计盈亏: {historical_total_pnl:+.2f} 元")
        print(f"[SUMMARY] 总本金: {total_principal:.2f} 元")
        if total_cost > 0:
            stamp_tax_ratio = total_stamp_tax / total_cost * 100
            print(f"[SUMMARY] 总交易成本: {total_cost:.2f} 元 (其中印花税: {total_stamp_tax:.2f} 元，占比: {stamp_tax_ratio:.1f}%)")
        else:
            print(f"[SUMMARY] 总交易成本: {total_cost:.2f} 元")
        
        # 打印信号值分析
        if len(df_win) > 0 and len(df_loss) > 0:
            win_avg_buy_signal = df_win['buy_signal'].mean() if 'buy_signal' in df_win.columns else 0
            win_avg_sell_signal = df_win['sell_signal'].mean() if 'sell_signal' in df_win.columns else 0
            win_signal_change = win_avg_sell_signal - win_avg_buy_signal
            
            loss_avg_buy_signal = df_loss['buy_signal'].mean() if 'buy_signal' in df_loss.columns else 0
            loss_avg_sell_signal = df_loss['sell_signal'].mean() if 'sell_signal' in df_loss.columns else 0
            loss_signal_change = loss_avg_sell_signal - loss_avg_buy_signal
            
            print(f"\n[信号分析] 🔴 赚钱品种 ({len(df_win)}个):")
            print(f"  平均买入信号: {win_avg_buy_signal:.4f}, 平均卖出信号: {win_avg_sell_signal:.4f}, 信号变化: {win_signal_change:+.4f}")
            print(f"[信号分析] 🟢 亏钱品种 ({len(df_loss)}个):")
            print(f"  平均买入信号: {loss_avg_buy_signal:.4f}, 平均卖出信号: {loss_avg_sell_signal:.4f}, 信号变化: {loss_signal_change:+.4f}")
    
    def send_4days_backtest_email(self):
        """
        用前4天的14:40信号和转天10:46价格计算收益情况，发送邮件，发4次
        每次发送一天的收益报告
        """
        print(f"\n{'='*70}")
        print("发送前4天回测收益邮件")
        print(f"{'='*70}\n")
        
        # 获取前4个交易日（排除周末）
        today = datetime.now().date()
        test_dates = []
        i = 1
        while len(test_dates) < 4 and i < 30:  # 最多往前找30天
            test_date = today - timedelta(days=i)
            # 排除周末（周六=5, 周日=6）
            if test_date.weekday() < 5:  # 周一到周五
                test_dates.append(test_date)
            i += 1
        test_dates = sorted(test_dates)  # 从早到晚排序
        
        print(f"将计算以下日期的收益: {[str(d) for d in test_dates]}\n")
        
        success_count = 0
        for test_date in test_dates:
            print(f"\n{'='*70}")
            print(f"处理日期: {test_date}")
            print(f"{'='*70}")
            
            # 读取该日期的14:40预测文件
            csv_filename = f"daily_predictions_{test_date.strftime('%Y%m%d')}.csv"
            csv_file = Path(self.WORKDIR) / csv_filename
            
            if not csv_file.exists():
                print(f"[WARN] 预测文件不存在: {csv_filename}，跳过")
                continue
            
            try:
                # 读取预测文件作为持仓
                df_predictions = pd.read_csv(csv_file)
                if df_predictions.empty:
                    print(f"[WARN] 预测文件为空: {csv_filename}，跳过")
                    continue
                
                # 创建持仓信息（模拟买入，与positions.json保持一致：每只股票200股）
                positions = {}
                for idx, row in df_predictions.iterrows():
                    code = row['code']
                    score = float(row.get('score', 0.0))
                    price = float(row.get('price', 0.0))
                    rank = int(row.get('rank', idx + 1))
                    
                    if price <= 0:
                        continue
                    
                    # 与positions.json保持一致：每只股票固定200股
                    shares = 200
                    
                    positions[code] = {
                        'buy_price': price,
                        'shares': shares,
                        'score': score,
                        'rank': rank,
                        'buy_date': str(test_date)
                    }
                
                if len(positions) == 0:
                    print(f"[WARN] 没有有效持仓，跳过")
                    continue
                
                print(f"[INFO] 持仓数量: {len(positions)}")
                
                # 获取转天10:46的价格
                next_date = test_date + timedelta(days=1)
                sell_time_str = f"{next_date} 10:46:00"
                print(f"[INFO] 获取 {sell_time_str} 价格...")
                
                # 准备收益数据
                profit_data = []
                
                # 先计算总本金（所有持仓，不管是否能获取卖出价格）
                total_principal_all = 0
                for code, position in positions.items():
                    buy_price = position['buy_price']
                    shares = position['shares']
                    buy_amount = buy_price * shares
                    buy_commission = max(buy_amount * 0.0003, 5.0)
                    buy_cost = buy_amount + buy_commission
                    total_principal_all += buy_cost
                
                print(f"[INFO] 总持仓本金（所有股票）: {total_principal_all:.2f} 元")
                
                for code, position in positions.items():
                    buy_price = position['buy_price']
                    shares = position['shares']
                    buy_signal = position.get('score', 0.0)
                    
                    # 获取转天10:46的价格
                    sell_price = self.get_current_price(code, sell_time_str)
                    if sell_price is None:
                        print(f"[WARN] {code} 无法获取 {sell_time_str} 价格，跳过（但仍计入总本金）")
                        # 即使没有卖出价格，也计入本金统计
                        buy_amount = buy_price * shares
                        buy_commission = max(buy_amount * 0.0003, 5.0)
                        buy_cost = buy_amount + buy_commission
                        # 不计算收益，但记录本金
                        continue
                    
                    # A股交易成本配置
                    BUY_COMMISSION_RATE = 0.0003  # 买入佣金：万分之3
                    SELL_COMMISSION_RATE = 0.0003  # 卖出佣金：万分之3
                    STAMP_TAX_RATE = 0.0005  # 印花税：万分之5（只在卖出时收取）
                    MIN_COMMISSION = 5.0  # 最低佣金：5元
                    
                    # 计算成交金额
                    buy_amount = buy_price * shares
                    sell_amount = sell_price * shares
                    
                    # 计算买入成本
                    buy_commission = max(buy_amount * BUY_COMMISSION_RATE, MIN_COMMISSION)
                    buy_cost = buy_amount + buy_commission
                    
                    # 计算卖出收入
                    sell_commission = max(sell_amount * SELL_COMMISSION_RATE, MIN_COMMISSION)
                    stamp_tax = sell_amount * STAMP_TAX_RATE
                    sell_income = sell_amount - sell_commission - stamp_tax
                    
                    # 实际盈亏
                    profit = sell_income - buy_cost
                    profit_rate = (profit / buy_cost * 100) if buy_cost > 0 else 0
                    
                    # 交易成本明细
                    total_cost = buy_commission + sell_commission + stamp_tax
                    
                    profit_data.append({
                        'code': code,
                        'rank': position.get('rank', 0),
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'shares': shares,
                        'principal': buy_cost,
                        'profit': profit,
                        'profit_rate': profit_rate / 100,
                        'buy_date': str(test_date),
                        'total_cost': total_cost,
                        'stamp_tax': stamp_tax,
                        'buy_commission': buy_commission,
                        'sell_commission': sell_commission,
                        'buy_signal': buy_signal,
                        'sell_signal': buy_signal  # 这里没有卖出信号，使用买入信号
                    })
                
                if len(profit_data) == 0:
                    print(f"[WARN] 没有可用的收益数据")
                    continue
                
                # 转换为DataFrame
                df_with_profit = pd.DataFrame(profit_data)
                df_with_profit = df_with_profit.sort_values('rank')
                
                # 生成HTML
                html = self.generate_sell_email_html(df_with_profit)
                
                # 发送邮件
                subject = f"回测收益报告 - {test_date} (买入: {test_date} 14:40, 卖出: {next_date} 10:46)"
                self.send_email(subject, html, 'sell')
                
                # 打印收益汇总
                total_profit = df_with_profit['profit'].sum()
                total_principal = df_with_profit['principal'].sum()
                avg_rate = (total_profit / total_principal * 100) if total_principal > 0 else 0
                
                profit_status = "盈利" if total_profit >= 0 else "亏损"
                profit_color = "🔴" if total_profit >= 0 else "🟢"
                
                # 计算实际参与交易的股票数量
                valid_trades = len(df_with_profit)
                total_holdings = len(positions)
                
                print(f"\n[SUMMARY] {profit_color} {test_date} 总盈亏: {total_profit:+.2f} 元 ({profit_status}), 平均收益率: {avg_rate:+.2f}%")
                print(f"[SUMMARY] 参与交易股票: {valid_trades}/{total_holdings} 只")
                print(f"[SUMMARY] 交易本金: {total_principal:.2f} 元")
                print(f"[SUMMARY] 总持仓本金: {total_principal_all:.2f} 元（包含无法获取卖出价格的股票）")
                print(f"[SUMMARY] 邮件已发送")
                
                success_count += 1
                
            except Exception as e:
                print(f"[ERROR] 处理 {test_date} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*70}")
        print(f"完成: 成功发送 {success_count}/{len(test_dates)} 封邮件")
        print(f"{'='*70}\n")
        return success_count == len(test_dates)
    
    def start_scheduler(self):
        """启动定时任务"""
        print("\n" + "="*70)
        print("[SCHEDULER] 邮件发送定时任务")
        print("="*70)
        print("[INFO] 买入邮件: 每天 15:40")
        print("[INFO] 卖出邮件: 每天 11:46（使用10:46价格数据）")
        print("="*70 + "\n")
        
        # 设置定时任务
        schedule.every().day.at("15:40").do(self.send_buy_email)
        schedule.every().day.at("11:46").do(self.send_sell_email)
        
        # 检查当前时间（使用北京时间）
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[TIME] 当前时间: {current_time}")
        print("[INFO] 等待定时任务触发...")
        
        # 主循环
        print("\n[START] 程序开始持续运行，等待定时任务...")
        print("[INFO] 按 Ctrl+C 停止程序")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
        except KeyboardInterrupt:
            print("\n[STOP] 程序已停止")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='邮件发送器')
    parser.add_argument('--buy-now', action='store_true', help='立即发送买入邮件')
    parser.add_argument('--sell-now', action='store_true', help='立即发送卖出邮件（10:46价格逻辑）')
    parser.add_argument('--backtest-4days', action='store_true', help='发送前4天回测收益邮件（4封）')
    parser.add_argument('--date', type=str, help='指定日期，格式: YYYY-MM-DD (例如: 2025-11-21)，用于发送指定日期的邮件')
    args = parser.parse_args()
    
    sender = EmailSender()
    
    # 如果没有指定日期，默认使用昨天（最近的交易日，使用北京时间）
    target_date = args.date
    if target_date is None:
        # 计算昨天
        yesterday = datetime.now() - timedelta(days=1)
        target_date = yesterday.strftime('%Y-%m-%d')
        print(f"[INFO] 未指定日期，使用昨天: {target_date}")
    else:
        print(f"[INFO] 使用指定日期: {target_date}")
    
    if args.buy_now:
        print(f"[NOW] 立即发送 {target_date} 的买入邮件...")
        sender.send_buy_email(target_date=target_date)
        return
    if args.sell_now:
        print(f"[NOW] 立即发送 {target_date} 的卖出邮件...")
        sender.send_sell_email(target_date=target_date)
        return
    
    if args.backtest_4days:
        print("[4DAYS] 开始发送前4天回测收益邮件...")
        ok = sender.send_4days_backtest_email()
        print(f"[4DAYS] 完成: {ok}")
        return
    
    # 未指定立即发送时，进入定时任务
    sender.start_scheduler()

if __name__ == "__main__":
    main()
