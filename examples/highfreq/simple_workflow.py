#!/usr/bin/env python3
"""
简化的高频工作流，避免内存问题
"""
import qlib
import pandas as pd
import numpy as np
from qlib.constant import REG_CN
from qlib.config import HIGH_FREQ_CONFIG
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.data import Cal
from qlib.tests.data import GetData
from qlib.data import D

# 简化的处理器
class SimpleProcessor:
    def __init__(self, fit_start_time, fit_end_time):
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
    
    def fit(self, df):
        pass
    
    def __call__(self, df):
        # 简单标准化
        df = df.copy()
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
        return df

# 简化的数据处理器
class SimpleHandler(DataHandlerLP):
    def __init__(self, instruments="csi300", start_time=None, end_time=None, 
                 infer_processors=[], learn_processors=[], fit_start_time=None, fit_end_time=None):
        
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self.get_feature_config(),
                "swap_level": False,
                "freq": "1min",
                "inst_processors": [],
            },
        }
        
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
        )
    
    def get_feature_config(self):
        # 只使用最基本的特征
        fields = ["$close", "$volume"]
        names = ["close", "volume"]
        return fields, names

def run_simple_workflow():
    """运行简化的高频工作流"""
    print("🔹 开始简化高频工作流")
    
    # 初始化qlib
    provider_uri = "~/.qlib/qlib_data/cn_data_1min"
    if not os.path.exists(os.path.expanduser(provider_uri)):
        print("下载分钟数据...")
        GetData().qlib_data(target_dir=provider_uri, interval="1min", region=REG_CN, exists_skip=True)
    
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    print("✅ Qlib初始化完成")
    
    # 设置时间范围（很短的时间段）
    start_time = "2020-12-01 00:00:00"
    end_time = "2020-12-03 16:00:00"  # 只取3天数据
    test_start = "2020-12-02 00:00:00"
    
    # 创建简化的处理器
    processor = SimpleProcessor(start_time, test_start)
    
    # 创建数据集配置
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "SimpleHandler",
                "module_path": "simple_workflow",
                "kwargs": {
                    "instruments": "csi300",
                    "start_time": start_time,
                    "end_time": end_time,
                    "infer_processors": [processor],
                    "learn_processors": [],
                },
            },
            "segments": {
                "train": (start_time, test_start),
                "test": (test_start, end_time),
            },
        },
    }
    
    # 创建数据集
    dataset = init_instance_by_config(dataset_config)
    xtest = dataset.prepare("test")
    
    print(f"✅ 测试数据形状: {xtest.shape}")
    print("前5行数据:")
    print(xtest.head())
    
    # 构建简单信号
    signals = xtest.mean(axis=1).to_frame('score').sort_index()
    print(f"✅ 信号数据形状: {signals.shape}")
    
    return signals

if __name__ == "__main__":
    signals = run_simple_workflow()


















