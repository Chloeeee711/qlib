"""
从CSV数据直接计算完整的180个特征（15基础 + 158 Alpha158 + 7 FZ7）
使用qlib的Alpha158Handler获取特征配置，然后从CSV数据计算
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# 添加路径以便导入qlib模块
PROJECT_ROOT = "/home/intern0/qlib"
WORKDIR = "/home/intern0/qlib/examples/highfreq"
for path in (PROJECT_ROOT, WORKDIR):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from scipy.stats import kurtosis
except ImportError:
    # 如果没有scipy，使用numpy实现kurtosis
    def kurtosis(a, fisher=False):
        """简化的kurtosis实现"""
        a = np.array(a)
        if len(a) < 4:
            return 0.0
        mean = np.mean(a)
        std = np.std(a)
        if std == 0:
            return 0.0
        normalized = (a - mean) / std
        kurt = np.mean(normalized ** 4)
        if fisher:
            return kurt - 3.0
        else:
            return kurt

# 导入qlib相关模块
try:
    import qlib
    from qlib.data import D
    from qlib.constant import REG_CN
    from qlib.contrib.data.handler import Alpha158 as Alpha158Handler
    from qlib.data.ops import Operators
    from qlib.data.dataset.loader import QlibDataLoader
    
    # 导入自定义操作符
    try:
        from highfreq_ops import DayLast, DayFirst, FFillNan, BFillNan, Date, Select, IsNull, DayShift, Cut, IntradayWindowVWAP, If, Gt, Lt
        Operators.register([DayLast, FFillNan, Date, Select, IsNull, Cut, DayFirst, IntradayWindowVWAP, DayShift, If, Gt, Lt])
    except Exception as e:
        print(f"[WARN] 导入自定义操作符失败: {e}")
except Exception as e:
    print(f"[WARN] 导入qlib模块失败: {e}")
    Alpha158Handler = None


class CSVFeatureCalculator:
    """从CSV数据直接计算180个特征"""
    
    def __init__(self, csv_data_dir: str):
        """
        csv_data_dir: CSV数据目录（/home/intern0/qlib_data_recent/features）
        """
        self.csv_data_dir = Path(csv_data_dir)
    
    def _read_csv_data(self, code: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """从CSV文件读取股票数据"""
        csv_path = self.csv_data_dir / code / "data.csv"
        if not csv_path.exists():
            return None
        
        try:
            df = pd.read_csv(csv_path)
            if 'datetime' not in df.columns:
                return None
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # 确保有必要的列
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                return None
            
            return df
        except Exception as e:
            print(f"[WARN] 读取 {code} CSV数据失败: {e}")
            return None
    
    def _calculate_basic_features(self, df: pd.DataFrame, target_idx: int) -> np.ndarray:
        """
        计算15个基础特征
        特征顺序：当前open, high, low, close, vwap, 前一天open, high, low, close, vwap, 当前volume, 前一天volume, 1.0, 0.0, 0.0
        """
        if target_idx >= len(df) or target_idx < 0:
            return None
        
        # 当前时刻的数据
        m = df.iloc[target_idx]
        m_vwap = (m['open'] + 2*m['high'] + 2*m['low'] + m['close']) / 6
        
        # 找前一个交易日（至少240分钟前，即1天前）
        prev_idx = None
        for i in range(target_idx - 1, max(0, target_idx - 500), -1):  # 最多向前找500分钟
            time_diff = (df.iloc[target_idx]['datetime'] - df.iloc[i]['datetime']).total_seconds() / 60
            if time_diff >= 240:  # 至少240分钟（1个交易日）
                prev_idx = i
                break
        
        if prev_idx is None:
            return None
        
        p = df.iloc[prev_idx]
        p_vwap = (p['open'] + 2*p['high'] + 2*p['low'] + p['close']) / 6
        
        # 获取归一化参考价格（前一交易日的收盘价）
        # 找m所在日期的前一交易日收盘价
        m_date = m['datetime'].date()
        prev_date = None
        for i in range(target_idx - 1, max(0, target_idx - 500), -1):
            row_date = df.iloc[i]['datetime'].date()
            if row_date < m_date:
                prev_date = row_date
                prev_day_data = df[df['datetime'].dt.date == prev_date]
                if not prev_day_data.empty:
                    ref_close_m = prev_day_data.iloc[-1]['close']
                    break
        else:
            return None
        
        # 找p所在日期的前一交易日收盘价
        p_date = p['datetime'].date()
        prev_date_p = None
        for i in range(prev_idx - 1, max(0, prev_idx - 500), -1):
            row_date = df.iloc[i]['datetime'].date()
            if row_date < p_date:
                prev_date_p = row_date
                prev_day_data_p = df[df['datetime'].dt.date == prev_date_p]
                if not prev_day_data_p.empty:
                    ref_close_p = prev_day_data_p.iloc[-1]['close']
                    break
        else:
            return None
        
        if ref_close_m <= 0 or ref_close_p <= 0:
            return None
        
        # 归一化价格（除以前一交易日收盘价）
        m_open_r = m['open'] / ref_close_m
        m_high_r = m['high'] / ref_close_m
        m_low_r = m['low'] / ref_close_m
        m_close_r = m['close'] / ref_close_m
        m_vwap_r = m_vwap / ref_close_m
        
        p_open_r = p['open'] / ref_close_p
        p_high_r = p['high'] / ref_close_p
        p_low_r = p['low'] / ref_close_p
        p_close_r = p['close'] / ref_close_p
        p_vwap_r = p_vwap / ref_close_p
        
        # 15个基础特征
        basic_features = np.array([
            m_open_r, m_high_r, m_low_r, m_close_r, m_vwap_r,
            p_open_r, p_high_r, p_low_r, p_close_r, p_vwap_r,
            float(m['volume']), float(p['volume']), 1.0, 0.0, 0.0
        ], dtype=np.float32)
        
        return basic_features
    
    def _calculate_alpha158_features(self, df: pd.DataFrame, target_idx: int, 
                                     qlib_data_dir: str) -> np.ndarray:
        """
        使用qlib的Alpha158Handler计算158个Alpha158特征
        
        方法：将CSV数据转换为qlib可以读取的格式，然后使用handler计算特征
        """
        if Alpha158Handler is None:
            print("[WARN] Alpha158Handler不可用，使用零填充")
            return np.zeros(158, dtype=np.float32)
        
        try:
            # 获取股票代码（从df的路径或其他方式）
            # 这里需要知道股票代码，暂时从外部传入
            code = getattr(self, '_current_code', None)
            if code is None:
                return np.zeros(158, dtype=np.float32)
            
            # 初始化qlib（如果还没初始化）
            qlib_data_path = Path(qlib_data_dir)
            if not qlib.initialized():
                try:
                    qlib.init(provider_uri=str(qlib_data_path), region=REG_CN)
                except:
                    qlib.init(region=REG_CN)
            
            # 检查qlib数据是否存在（bin文件）
            feature_dir = qlib_data_path / "features" / code
            if not feature_dir.exists():
                # 如果bin文件不存在，尝试从CSV创建临时bin文件
                if not self._create_temp_bin_from_csv(code, df, qlib_data_path):
                    return np.zeros(158, dtype=np.float32)
            
            # 使用Alpha158Handler计算特征
            # 构建时间范围（需要足够的历史数据）
            start_time = df.iloc[0]['datetime'].strftime("%Y-%m-%d")
            end_time = df.iloc[target_idx]['datetime'].strftime("%Y-%m-%d %H:%M:%S")
            
            # 创建handler（使用分钟级数据）
            # 注意：Alpha158Handler默认是日频，我们需要使用高频版本的handler
            # 但这里我们先尝试直接使用Alpha158Handler，看看能否工作
            
            # 实际上，我们应该使用highfreq_handler_alpha158fz7中的handler
            # 因为它已经适配了分钟级数据
            from highfreq_handler_alpha158fz7 import HighFreqHandler
            
            handler = HighFreqHandler(
                instruments=[code],
                start_time=start_time,
                end_time=end_time,
                infer_processors=[],
                learn_processors=[],
            )
            
            # 准备数据
            handler.setup_data()
            
            # 获取特征数据
            feature_data = handler.prepare("test", col_set="feature", data_key=handler.DK_I)
            
            if feature_data is None or feature_data.empty:
                return np.zeros(158, dtype=np.float32)
            
            # 提取Alpha158特征（从第15个特征开始，到第173个特征，共158个）
            target_time_pd = pd.to_datetime(df.iloc[target_idx]['datetime'])
            
            if isinstance(feature_data.index, pd.MultiIndex):
                code_data = feature_data.xs(code, level='instrument', drop_level=False)
                times = pd.to_datetime(code_data.index.get_level_values('datetime'))
                time_mask = times == target_time_pd
                
                if time_mask.any():
                    row = code_data[time_mask].iloc[0]
                else:
                    time_diffs = times - target_time_pd
                    valid_diffs = time_diffs[time_diffs <= pd.Timedelta(0)]
                    if len(valid_diffs) > 0:
                        closest_idx = valid_diffs.abs().argmin()
                        row = code_data.iloc[closest_idx]
                    else:
                        closest_idx = time_diffs.abs().argmin()
                        row = code_data.iloc[closest_idx]
            else:
                row = feature_data.iloc[0]
            
            # 提取特征值
            if isinstance(row, pd.Series):
                all_feat_vals = row.values
            elif isinstance(row, pd.DataFrame):
                if ('feature',) in row.columns.names:
                    all_feat_vals = row.xs('feature', level=0, axis=1).values.flatten()
                else:
                    all_feat_vals = row.values.flatten()
            else:
                all_feat_vals = np.array(row).flatten()
            
            # 提取Alpha158特征（索引15到172，共158个）
            if len(all_feat_vals) >= 173:
                alpha158_features = all_feat_vals[15:173].astype(np.float32)
            elif len(all_feat_vals) == 180:
                # 如果正好是180个特征，提取中间158个
                alpha158_features = all_feat_vals[15:173].astype(np.float32)
            else:
                # 如果特征数量不对，用零填充
                print(f"[WARN] 特征数量异常: {len(all_feat_vals)}，期望至少173个")
                alpha158_features = np.zeros(158, dtype=np.float32)
            
            return alpha158_features
            
        except Exception as e:
            print(f"[ERROR] 计算Alpha158特征失败: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(158, dtype=np.float32)
    
    def _create_temp_bin_from_csv(self, code: str, df: pd.DataFrame, qlib_data_dir: Path) -> bool:
        """
        从CSV数据创建临时的qlib bin文件
        这是一个简化版本，只创建必要的字段
        """
        try:
            feature_dir = qlib_data_dir / "features" / code
            feature_dir.mkdir(parents=True, exist_ok=True)
            
            # 为每个字段创建bin文件
            fields = ['open', 'high', 'low', 'close', 'volume']
            
            for field in fields:
                bin_path = feature_dir / f"{field}.1min.bin"
                
                # 提取该字段的数据
                values = df[field].values.astype(np.float32)
                
                # qlib的bin格式：第一个值是start_index（int32），然后是数据（float32）
                start_index = np.array([0], dtype=np.int32)
                data = values.astype(np.float32)
                
                # 写入bin文件
                with open(bin_path, 'wb') as f:
                    start_index.tofile(f)
                    data.tofile(f)
            
            # 创建或更新日历文件
            calendar_path = qlib_data_dir / "calendars" / "1min.txt"
            calendar_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 读取现有日历（如果存在）
            existing_minutes = set()
            if calendar_path.exists():
                with open(calendar_path, 'r') as f:
                    existing_minutes = set(line.strip() for line in f)
            
            # 添加新的分钟时间戳
            new_minutes = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S').unique()
            all_minutes = sorted(existing_minutes | set(new_minutes))
            
            with open(calendar_path, 'w') as f:
                for minute in all_minutes:
                    f.write(f"{minute}\n")
            
            return True
        except Exception as e:
            print(f"[WARN] 创建临时bin文件失败: {e}")
            return False
    
    def _calculate_fz7_features(self, df: pd.DataFrame, target_idx: int) -> np.ndarray:
        """
        计算7个FZ7特征
        """
        if target_idx >= len(df) or target_idx < 240:  # 需要至少240分钟的历史数据
            return np.zeros(7, dtype=np.float32)
        
        # 获取最近240分钟的数据（1个交易日）
        window_data = df.iloc[max(0, target_idx - 239):target_idx + 1]
        
        close = window_data['close'].values
        high = window_data['high'].values
        low = window_data['low'].values
        volume = window_data['volume'].values
        
        # 当前收盘价
        current_close = df.iloc[target_idx]['close']
        
        # FZ1: Max($high, 240) / Min($low, 240) - 1
        fz1 = (high.max() / low.min() - 1) if low.min() > 0 else 0.0
        
        # FZ2: Mad($high, 240) / $close
        high_mad = np.median(np.abs(high - np.median(high)))
        fz2 = (high_mad / current_close) if current_close > 0 else 0.0
        
        # FZ3: Std($close / Ref($close, 4) - 1, 240)
        if len(close) >= 5:
            ret_4 = close[4:] / close[:-4] - 1
            fz3 = ret_4.std() if len(ret_4) > 1 else 0.0
        else:
            fz3 = 0.0
        
        # FZ4: Kurt($close / Ref($close, 4) - 1, 240)
        if len(close) >= 5:
            ret_4 = close[4:] / close[:-4] - 1
            if len(ret_4) > 3:
                from scipy.stats import kurtosis
                fz4 = float(kurtosis(ret_4, fisher=False))  # Fisher=False表示Pearson定义
            else:
                fz4 = 0.0
        else:
            fz4 = 0.0
        
        # FZ5: Corr(Ref($high, 1), $volume, 237)
        if len(high) >= 2 and len(volume) >= 2:
            high_prev = high[:-1]
            volume_curr = volume[1:]
            min_len = min(len(high_prev), len(volume_curr), 237)
            if min_len > 1:
                fz5 = float(np.corrcoef(high_prev[:min_len], volume_curr[:min_len])[0, 1])
                if np.isnan(fz5):
                    fz5 = 0.0
            else:
                fz5 = 0.0
        else:
            fz5 = 0.0
        
        # FZ6: Max($close, 240)
        fz6 = close.max()
        
        # FZ7: Min($close / Ref($close, 7) - 1, 240)
        if len(close) >= 8:
            ret_7 = close[7:] / close[:-7] - 1
            fz7 = ret_7.min() if len(ret_7) > 0 else 0.0
        else:
            fz7 = 0.0
        
        fz_features = np.array([fz1, fz2, fz3, fz4, fz5, fz6, fz7], dtype=np.float32)
        return fz_features
    
    def calculate_features_from_csv(self, code: str, target_time: datetime, 
                                   history_days: int = 30, qlib_data_dir: str = None) -> Optional[np.ndarray]:
        """
        从CSV数据计算180个特征
        
        Args:
            code: 股票代码
            target_time: 目标时间（用于计算特征的时间点）
            history_days: 需要多少天的历史数据
            
        Returns:
            180个特征的numpy数组，如果失败返回None
        """
        try:
            # 1. 读取足够的历史数据
            start_date = target_time - timedelta(days=history_days)
            end_date = target_time + timedelta(hours=1)  # 多读1小时确保有目标时间的数据
            
            df = self._read_csv_data(code, start_date, end_date)
            if df is None or df.empty:
                return None
            
            # 2. 找到目标时间对应的索引
            target_time_pd = pd.to_datetime(target_time)
            time_diffs = (df['datetime'] - target_time_pd).abs()
            target_idx = time_diffs.idxmin()
            
            # 如果时间差太大（超过10分钟），认为没有精确匹配
            if time_diffs.iloc[target_idx] > pd.Timedelta(minutes=10):
                return None
            
            # 3. 计算15个基础特征
            basic_features = self._calculate_basic_features(df, target_idx)
            if basic_features is None:
                return None
            
            # 4. 计算158个Alpha158特征（使用qlib的Alpha158Handler）
            # 保存当前股票代码，供_calculate_alpha158_features使用
            self._current_code = code
            if qlib_data_dir is None:
                qlib_data_dir = str(self.csv_data_dir.parent)  # 使用CSV目录的父目录作为qlib数据目录
            alpha158_features = self._calculate_alpha158_features(df, target_idx, qlib_data_dir)
            
            # 5. 计算7个FZ7特征
            fz7_features = self._calculate_fz7_features(df, target_idx)
            
            # 6. 合并所有特征
            all_features = np.hstack([basic_features, alpha158_features, fz7_features])
            
            if len(all_features) != 180:
                print(f"[WARN] {code} 特征数量异常: {len(all_features)}，期望180")
                return None
            
            return all_features.astype(np.float32)
                
        except Exception as e:
            print(f"[ERROR] {code} 计算特征失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_features_batch(self, codes: List[str], target_time: datetime,
                                history_days: int = 30, qlib_data_dir: str = None) -> Dict[str, np.ndarray]:
        """
        批量计算多个股票的特征
        
        Returns:
            {code: features} 字典
        """
        results = {}
        for i, code in enumerate(codes):
            if (i + 1) % 50 == 0:
                print(f"   进度: {i+1}/{len(codes)}")
            features = self.calculate_features_from_csv(code, target_time, history_days, qlib_data_dir)
            if features is not None:
                results[code] = features
        return results
