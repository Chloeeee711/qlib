import numpy as np
import pandas as pd
from qlib.constant import EPS
from qlib.data.dataset.processor import Processor
from qlib.data.dataset.utils import fetch_df_by_index


class HighFreqNorm(Processor):
    def __init__(self, fit_start_time, fit_end_time):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.feature_med = {}
        self.feature_std = {}
        self.feature_vmax = {}
        self.feature_vmin = {}

    def fit(self, df_features):
        """Fit robust scaling stats (median/MAD) on TRAIN window only to avoid leakage."""
        fetch_df = fetch_df_by_index(
            df_features,
            slice(self.fit_start_time, self.fit_end_time),
            level="datetime",
        )
        if fetch_df.empty:
            # Fallback to no-op stats
            self.feature_med = {"price": 0.0, "volume": 0.0}
            self.feature_std = {"price": 1.0, "volume": 1.0}
            self.feature_vmax = {"price": 10.0, "volume": 10.0}
            self.feature_vmin = {"price": -10.0, "volume": -10.0}
            return

        # Use only feature columns; detect volume-like columns by name
        feature_cols = [col for col in fetch_df.columns if col[0] == "feature"]
        if not feature_cols:
            self.feature_med = {"price": 0.0, "volume": 0.0}
            self.feature_std = {"price": 1.0, "volume": 1.0}
            self.feature_vmax = {"price": 10.0, "volume": 10.0}
            self.feature_vmin = {"price": -10.0, "volume": -10.0}
            return

        feat = fetch_df[feature_cols]
        col_names = [c[1] for c in feat.columns]
        vol_mask = np.array([(isinstance(n, str) and ("volume" in n.lower() or n in ("$volume", "$volume_1"))) for n in col_names])
        price_mask = ~vol_mask

        def _robust_stats(values, max_samples=10_000_000):
            """
            使用采样方法计算鲁棒统计量，避免内存溢出
            对于大数据集，采样足够的数据点就能得到准确的统计量
            """
            # 如果数据量太大，进行采样
            if len(values) > max_samples:
                # 随机采样
                indices = np.random.choice(len(values), size=max_samples, replace=False)
                sampled_values = values[indices]
            else:
                sampled_values = values
            
            # 移除 NaN
            sampled_values = sampled_values[~np.isnan(sampled_values)]
            
            if len(sampled_values) == 0:
                return 0.0, 1.0, 10.0, -10.0
            
            med = np.median(sampled_values)
            mad = np.median(np.abs(sampled_values - med)) * 1.4826 + EPS
            # 防止除零
            if mad < 1e-8:
                mad = 1.0
            
            # 对于 vmax 和 vmin，需要在整个数据集上计算（但可以分块处理）
            # 或者使用采样数据的统计量
            scaled = (sampled_values - med) / mad
            scaled = np.clip(scaled, -8, 8)
            vmax = np.max(scaled)
            vmin = np.min(scaled)
            
            return med, mad, vmax, vmin

        X = feat.values.astype(np.float32)

        # Price-like group
        if price_mask.any():
            price_data = X[:, price_mask]
            total_elements = price_data.size
            
            # 如果数据太大，直接采样行和列，避免创建巨大的扁平数组
            if total_elements > 10_000_000:
                # 采样行和列索引，然后提取数据
                n_rows, n_cols = price_data.shape
                max_samples = 10_000_000
                # 计算需要采样多少行
                n_sample_rows = min(n_rows, int(np.sqrt(max_samples)))
                n_sample_cols = min(n_cols, max_samples // n_sample_rows)
                
                # 随机采样行和列
                row_indices = np.random.choice(n_rows, size=n_sample_rows, replace=False)
                col_indices = np.random.choice(n_cols, size=n_sample_cols, replace=False)
                
                # 提取采样数据（不创建完整数组）
                sampled_data = price_data[np.ix_(row_indices, col_indices)].ravel()
            else:
                sampled_data = price_data.ravel()
            
            med, mad, vmax, vmin = _robust_stats(sampled_data)
            self.feature_med["price"], self.feature_std["price"], self.feature_vmax["price"], self.feature_vmin["price"] = med, mad, vmax, vmin
        else:
            self.feature_med["price"], self.feature_std["price"], self.feature_vmax["price"], self.feature_vmin["price"] = 0.0, 1.0, 10.0, -10.0

        # Volume-like group (log1p)
        if vol_mask.any():
            vol_data = X[:, vol_mask]
            total_elements = vol_data.size
            
            # 如果数据太大，直接采样行和列，避免创建巨大的扁平数组
            if total_elements > 10_000_000:
                # 采样行和列索引，然后提取数据
                n_rows, n_cols = vol_data.shape
                max_samples = 10_000_000
                # 计算需要采样多少行
                n_sample_rows = min(n_rows, int(np.sqrt(max_samples)))
                n_sample_cols = min(n_cols, max_samples // n_sample_rows)
                
                # 随机采样行和列
                row_indices = np.random.choice(n_rows, size=n_sample_rows, replace=False)
                col_indices = np.random.choice(n_cols, size=n_sample_cols, replace=False)
                
                # 提取采样数据（不创建完整数组）
                sampled_data = vol_data[np.ix_(row_indices, col_indices)].ravel()
            else:
                sampled_data = vol_data.ravel()
            
            vol_vals = np.log1p(sampled_data)
            med, mad, vmax, vmin = _robust_stats(vol_vals)
            self.feature_med["volume"], self.feature_std["volume"], self.feature_vmax["volume"], self.feature_vmin["volume"] = med, mad, vmax, vmin
        else:
            self.feature_med["volume"], self.feature_std["volume"], self.feature_vmax["volume"], self.feature_vmin["volume"] = 0.0, 1.0, 10.0, -10.0

    def __call__(self, df_features):
        """Apply the fitted scaling to features only; labels are passed through."""
        if df_features.empty:
            return df_features.copy()

        feature_cols = [col for col in df_features.columns if col[0] == "feature"]
        label_cols = [col for col in df_features.columns if col[0] == "label"]

        result = df_features.copy()
        if feature_cols:
            for col in feature_cols:
                name = str(col[1])
                vals = result[col].astype(np.float32).values
                if ("volume" in name.lower()) or (name in ("$volume", "$volume_1")):
                    vals = np.log1p(vals)
                    med, std = self.feature_med.get("volume", 0.0), self.feature_std.get("volume", 1.0)
                else:
                    med, std = self.feature_med.get("price", 0.0), self.feature_std.get("price", 1.0)
                scaled = (vals - med) / (std if std != 0 else 1.0)
                # 应用截断：避免极端值传导到模型
                scaled = np.clip(scaled, -8, 8)
                result[col] = scaled

        # keep labels untouched
        return result
