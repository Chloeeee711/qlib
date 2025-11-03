"""
模型集成模块：从notebook加载模型和生成信号
需要在notebook中运行后调用此模块
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime, date
import qlib
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.model.trainer import task_train
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord

class ModelSignalGenerator:
    """模型信号生成器"""
    
    def __init__(self, recorder_id=None, qlib_data_path=None):
        """
        初始化
        
        参数:
            recorder_id: recorder的ID，None则自动查找最新的
            qlib_data_path: Qlib数据路径
        """
        self.qlib_data_path = qlib_data_path or r"C:\Users\ASUS\qlib_data"
        self.recorder_id = recorder_id
        self.recorder = None
        self.model = None
        self.pred_processed = None
        
        # 初始化Qlib
        qlib.init(provider_uri=self.qlib_data_path, region="cn")
    
    def load_latest_model(self):
        """加载最新的模型和预测结果"""
        try:
            # 如果指定了recorder_id，加载指定的
            if self.recorder_id:
                self.recorder = R.get_recorder(recorder_id=self.recorder_id)
            else:
                # 否则加载最新的recorder
                experiment_name = "experiment_name"  # 您需要根据实际情况修改
                records = R.list_recorders(experiment_name=experiment_name)
                if records:
                    self.recorder = R.get_recorder(records[-1])
            
            # 加载模型
            self.model = self.recorder.load_object("model.pkl")
            
            # 加载预测结果
            self.pred_processed = self.recorder.load_object("pred.pkl")
            
            print("✅ 模型和预测数据加载成功")
            return True
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return False
    
    def get_signals_for_date(self, target_date):
        """
        获取特定日期的交易信号
        
        参数:
            target_date: datetime.date 对象
            
        返回:
            dict: {'SH600519': 0.85, 'SZ000001': 0.78, ...}
        """
        if self.pred_processed is None:
            print("❌ 预测数据未加载")
            return {}
        
        try:
            # 筛选指定日期的信号
            if isinstance(self.pred_processed.index, pd.MultiIndex):
                date_mask = self.pred_processed.index.get_level_values('datetime').date == target_date
                signals = self.pred_processed[date_mask]
            else:
                # 如果索引是时间序列，按日期筛选
                date_mask = pd.to_datetime(self.pred_processed.index).date == target_date
                signals = self.pred_processed[date_mask]
            
            # 转换为字典格式 {stock: score}
            signals_dict = signals.to_dict()
            
            print(f"✅ 获取到 {len(signals_dict)} 只股票的信号")
            return signals_dict
            
        except Exception as e:
            print(f"❌ 获取信号失败: {e}")
            return {}
    
    def get_topk_signals(self, target_date, topk=50, threshold=0.15):
        """
        获取topk信号，应用阈值过滤
        
        参数:
            target_date: 目标日期
            topk: 返回前k只股票
            threshold: 信号阈值
            
        返回:
            list: [(stock, score), ...]，按分数降序
        """
        signals = self.get_signals_for_date(target_date)
        
        # 应用阈值
        if threshold > 0:
            signals = {k: v for k, v in signals.items() if v >= threshold}
        
        # 按分数降序排序，取topk
        sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        topk_signals = sorted_signals[:topk]
        
        return topk_signals
    
    def get_real_time_price(self, stock, target_time):
        """
        获取实时价格
        
        参数:
            stock: 股票代码，如 'SH600519'
            target_time: 目标时间
            
        返回:
            float: 价格
        """
        try:
            # 从Qlib获取价格数据
            data = D.features(
                ['close'],
                instruments=[stock],
                start_time=target_time,
                end_time=target_time,
                freq='1min'
            )
            
            if len(data) > 0:
                return data.iloc[-1].values[0]
            else:
                return None
                
        except Exception as e:
            print(f"❌ 获取价格失败: {e}")
            return None


def get_model_signals_for_buy(target_date=None, topk=50, threshold=0.15):
    """
    便捷函数：获取买入信号
    
    使用示例:
    >>> signals = get_model_signals_for_buy()
    >>> for stock, score in signals:
    >>>     print(f"{stock}: {score}")
    """
    generator = ModelSignalGenerator()
    
    if not generator.load_latest_model():
        return []
    
    if target_date is None:
        target_date = date.today()
    
    return generator.get_topk_signals(target_date, topk=topk, threshold=threshold)


if __name__ == "__main__":
    # 测试
    print("🧪 测试模型信号生成")
    
    generator = ModelSignalGenerator()
    if generator.load_latest_model():
        signals = generator.get_topk_signals(date.today(), topk=10)
        print("\n📊 今日Top 10信号:")
        for stock, score in signals:
            print(f"  {stock}: {score:.2f}")

