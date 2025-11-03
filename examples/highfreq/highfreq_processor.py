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

        def _robust_stats(values):
            med = np.nanmedian(values)
            mad = np.nanmedian(np.abs(values - med)) * 1.4826 + EPS
            # 防止除零
            if mad < 1e-8:
                mad = 1.0
            scaled = (values - med) / mad
            # 轻量截断：避免极端值被放大
            scaled = np.clip(scaled, -8, 8)
            vmax = np.nanmax(scaled)
            vmin = np.nanmin(scaled)
            return med, mad, vmax, vmin

        X = feat.values.astype(np.float32)

        # Price-like group
        if price_mask.any():
            med, mad, vmax, vmin = _robust_stats(X[:, price_mask].ravel())
            self.feature_med["price"], self.feature_std["price"], self.feature_vmax["price"], self.feature_vmin["price"] = med, mad, vmax, vmin
        else:
            self.feature_med["price"], self.feature_std["price"], self.feature_vmax["price"], self.feature_vmin["price"] = 0.0, 1.0, 10.0, -10.0

        # Volume-like group (log1p)
        if vol_mask.any():
            vol_vals = np.log1p(X[:, vol_mask].ravel())
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
