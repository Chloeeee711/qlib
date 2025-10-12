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
        # 检查输入数据是否为空
        if df_features.empty:
            print("警告: 输入数据为空，返回空DataFrame")
            return df_features.copy()
        
        # 分离特征列和标签列
        feature_cols = [col for col in df_features.columns if col[0] == "feature"]
        label_cols = [col for col in df_features.columns if col[0] == "label"]
        
        # 处理特征列
        if feature_cols:
            feature_data = df_features[feature_cols]
            df_values = feature_data.values.copy()
            
            # 简单的标准化处理
            for i in range(df_values.shape[1]):
                col_data = df_values[:, i]
                # 检查列数据是否为空或全为NaN
                if len(col_data) == 0 or np.all(np.isnan(col_data)):
                    print(f"警告: 第{i}列数据为空或全为NaN，跳过处理")
                    continue
                    
                # 简单的z-score标准化
                mean_val = np.nanmean(col_data)
                std_val = np.nanstd(col_data)
                if std_val > 0 and not np.isnan(mean_val) and not np.isnan(std_val):
                    df_values[:, i] = (col_data - mean_val) / std_val
                else:
                    print(f"警告: 第{i}列标准化参数无效，保持原值")
            
            # 创建处理后的特征列，保持多级列名结构
            processed_features = pd.DataFrame(
                data=df_values,
                index=feature_data.index,
                columns=feature_data.columns,
            )
        else:
            processed_features = pd.DataFrame(index=df_features.index)
        
        # 处理标签列（保持原样）
        if label_cols:
            processed_labels = df_features[label_cols]
        else:
            processed_labels = pd.DataFrame(index=df_features.index)
        
        # 合并特征和标签
        if not processed_features.empty and not processed_labels.empty:
            result = pd.concat([processed_features, processed_labels], axis=1)
        elif not processed_features.empty:
            result = processed_features
        elif not processed_labels.empty:
            result = processed_labels
        else:
            result = pd.DataFrame(index=df_features.index)
        
        return result
