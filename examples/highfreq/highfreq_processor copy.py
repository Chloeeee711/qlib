import numpy as np
import pandas as pd
from qlib.constant import EPS
from qlib.data.dataset.processor import Processor
from qlib.data.dataset.utils import fetch_df_by_index


class HighFreqNorm(Processor):
    def __init__(self, fit_start_time, fit_end_time):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time

    def fit(self, df_features):
        fetch_df = fetch_df_by_index(df_features, slice(self.fit_start_time, self.fit_end_time), level="datetime")
        del df_features
        df_values = fetch_df.values
        names = {
            "price": slice(0, 10),
            "volume": slice(10, 12),
        }
        self.feature_med = {}
        self.feature_std = {}
        self.feature_vmax = {}
        self.feature_vmin = {}
        for name, name_val in names.items():
            part_values = df_values[:, name_val].astype(np.float32)
            if name == "volume":
                part_values = np.log1p(part_values)
            self.feature_med[name] = np.nanmedian(part_values)
            part_values = part_values - self.feature_med[name]
            self.feature_std[name] = np.nanmedian(np.absolute(part_values)) * 1.4826 + EPS
            part_values = part_values / self.feature_std[name]
            self.feature_vmax[name] = np.nanmax(part_values)
            self.feature_vmin[name] = np.nanmin(part_values)

    def __call__(self, df_features):
        df_features["date"] = pd.to_datetime(
            df_features.index.get_level_values(level="datetime").to_series().dt.date.values
        )
        df_features.set_index("date", append=True, drop=True, inplace=True)
        df_values = df_features.values
        names = {
            "price": slice(0, 10),
            "volume": slice(10, 12),
        }

        for name, name_val in names.items():
            if name == "volume":
                df_values[:, name_val] = np.log1p(df_values[:, name_val])
            df_values[:, name_val] -= self.feature_med[name]
            df_values[:, name_val] /= self.feature_std[name]
            slice0 = df_values[:, name_val] > 3.0
            slice1 = df_values[:, name_val] > 3.5
            slice2 = df_values[:, name_val] < -3.0
            slice3 = df_values[:, name_val] < -3.5

            df_values[:, name_val][slice0] = (
                3.0 + (df_values[:, name_val][slice0] - 3.0) / (self.feature_vmax[name] - 3) * 0.5
            )
            df_values[:, name_val][slice1] = 3.5
            df_values[:, name_val][slice2] = (
                -3.0 - (df_values[:, name_val][slice2] + 3.0) / (self.feature_vmin[name] + 3) * 0.5
            )
            df_values[:, name_val][slice3] = -3.5
        # Reshape is specifically for adapting to RL high-freq executor
        #动态reshape避免长度不一致
        n_samples = df_values.shape[0] // 6
        feat = df_values[:, [0, 1, 2, 3, 4, 10]].reshape(n_samples, -1)
        feat_1 = df_values[:, [5, 6, 7, 8, 9, 11]].reshape(n_samples, -1)
        
        # 正确创建索引：每6行取一行，对应reshape后的数据
        # 先获取原始索引（在添加date之前）
        original_idx = df_features.index.droplevel("date")  # 移除date级别
        idx = original_idx[::6]  # 每6行取一行
        idx = idx.drop_duplicates()
        idx.set_names(["instrument", "datetime"], inplace=True)

        """df_new_features = pd.DataFrame(
            data=np.concatenate((feat, feat_1), axis=1),
            index=idx,
            columns=["FEATURE_%d" % i for i in range(12 * 240)],
        ).sort_index()
        return df_new_features"""

        
           # 合并特征和标签
        df_new_features = pd.DataFrame(
            data=np.concatenate((feat, feat_1), axis=1),
            index=idx,
            columns=pd.MultiIndex.from_tuples(
                [("feature", f"FEATURE_{i}") for i in range(feat.shape[1] + feat_1.shape[1])]
            ),
        ).sort_index()
    
        # 处理标签列
        label_name = ("label", "LABEL0")
        if label_name in df_features.columns:
            # 按 n_samples 取值，如果 label 是逐分钟的，需要 reshape 或 groupby
            label_data = df_features[label_name].values.reshape(n_samples, -1)[:, 0]  # 取每段的第一个作为 label
            df_new_features[label_name] = label_data
        else:
            print("⚠️ LABEL0 不存在，y_train 仍然会是 None")
    
        return df_new_features
        
        
        
        """
        df_new_features = pd.DataFrame(
            data=np.concatenate((feat, feat_1), axis=1),
            index=idx,
            columns=pd.MultiIndex.from_tuples(
                [("feature", f"FEATURE_{i}") for i in range(feat.shape[1] + feat_1.shape[1])
            ]),
        ).sort_index()
        return df_new_features
        """