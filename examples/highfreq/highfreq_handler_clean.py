from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc
import highfreq_ops
import pandas as pd 
import numpy as np

class HighFreqHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        drop_raw=True,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        
        # 存储自定义标签数据
        self._custom_labels = None

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self.get_feature_config(),
                "swap_level": False,
                "freq": "1min",
                "inst_processors": [],  # 明确指定为空，避免参数冲突
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            drop_raw=drop_raw,
        )

    def get_feature_config(self):
        fields = []
        names = []

        template_if = "If(IsNull({1}), {0}, {1})"
        template_paused = "Select(Or(IsNull($paused), Eq($paused, 0.0)), {0})"
        template_fillnan = "BFillNan(FFillNan({0}))"
        # Because there is no vwap field in the yahoo data, a method similar to Simpson integration is used to approximate vwap
        template_vwap = "Div(Add(Add(Mul($close, 2), Add($high, $low)), Add($open, $close)), 6)"
        template_returns = "Ref(Div(Sub($close, Ref($close, 1)), Ref($close, 1)), -1)"
        template_volume = "Mul($volume, Div(Add($close, $open), 2))"

        # 基础特征
        fields.extend([
            template_fillnan.format("$open"),
            template_fillnan.format("$high"),
            template_fillnan.format("$low"),
            template_fillnan.format("$close"),
            template_fillnan.format("$volume"),
        ])
        names.extend(["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])

        # 技术指标
        fields.extend([
            template_fillnan.format(template_vwap),
            template_fillnan.format("Mean($close, 5)"),
            template_fillnan.format("Mean($close, 10)"),
            template_fillnan.format("Mean($close, 20)"),
            template_fillnan.format("Std($close, 5)"),
            template_fillnan.format("Std($close, 10)"),
            template_fillnan.format("Std($close, 20)"),
        ])
        names.extend(["VWAP", "MA5", "MA10", "MA20", "STD5", "STD10", "STD20"])

        # 收益率特征
        fields.extend([
            template_fillnan.format(template_returns),
            template_fillnan.format("Ref($close, -1) / $close - 1"),
            template_fillnan.format("Ref($close, -2) / $close - 1"),
            template_fillnan.format("Ref($close, -3) / $close - 1"),
        ])
        names.extend(["RET_1", "RET_2", "RET_3", "RET_4"])

        # 成交量特征
        fields.extend([
            template_fillnan.format(template_volume),
            template_fillnan.format("Mean($volume, 5)"),
            template_fillnan.format("Mean($volume, 10)"),
            template_fillnan.format("Mean($volume, 20)"),
        ])
        names.extend(["VOLUME_AMOUNT", "VOLUME_MA5", "VOLUME_MA10", "VOLUME_MA20"])

        # 隔夜收益率标签：使用字符串表达式
        # 当日 14:41-14:50 VWAP
        today_vwap = "IntradayWindowVWAP($close, 14.41, 14.50, '$volume')"
        
        # 次日 10:11-10:20 VWAP（通过 DayShift 移位）
        next_vwap = "DayShift(IntradayWindowVWAP($close, 10.11, 10.20, '$volume'), 1)"
        
        # 隔夜收益率
        label_expr = f"{next_vwap} / {today_vwap} - 1"
        
        # 返回多组配置，支持多级列名
        return {
            "feature": (fields, names),
            "label": ([label_expr], ["LABEL0"]),
        }
    
    def _load_custom_labels(self, instruments, start_time, end_time):
        """计算自定义标签"""
        if self._custom_labels is not None:
            return self._custom_labels
            
        print("计算自定义隔夜收益率标签...")
        parts = []
        
        for inst in instruments:
            try:
                # 使用你的compute_label_single逻辑
                label_data = self._compute_label_single(inst, start_time, end_time)
                if not label_data.empty:
                    parts.append(label_data)
            except Exception as e:
                print(f"计算 {inst} 标签失败: {e}")
                continue
        
        if parts:
            self._custom_labels = pd.concat(parts)
        else:
            self._custom_labels = pd.DataFrame(columns=["LABEL0"])
            
        print(f"自定义标签计算完成，形状: {self._custom_labels.shape}")
        return self._custom_labels
    
    def _compute_label_single(self, inst, start_time, end_time):
        """单只股票的标签计算逻辑"""
        from qlib.data import D
        import pandas as pd
        import numpy as np
        
        def empty_label_df(inst):
            return pd.DataFrame(
                index=pd.MultiIndex.from_product([[inst], []], names=["instrument","datetime"]),
                columns=["LABEL0"]
            )
          
        # 拉分钟数据
        try:
            df = D.features([inst], ["$open", "$high", "$low", "$close", "$volume"],
                            start_time=start_time, end_time=end_time, freq="1min")
        except Exception:
            return empty_label_df(inst)

        if df is None or df.empty:
            return empty_label_df(inst)

        # MultiIndex 安全切片
        if isinstance(df.index, pd.MultiIndex):
            lvl = "instrument" if "instrument" in df.index.names else df.index.names[0]
            inst_levels = df.index.get_level_values(lvl).unique()
            if inst not in inst_levels:
                return empty_label_df(inst)
            df = df.xs(inst, level=lvl, drop_level=True)

        if df.empty:
            return empty_label_df(inst)

        # 时间窗口掩码
        dt = pd.to_datetime(df.index)
        t_int = dt.hour * 100 + dt.minute
        mask_today = (t_int >= int(14.41 * 100)) & (t_int <= int(14.50 * 100))
        mask_next  = (t_int >= int(10.11 * 100)) & (t_int <= int(10.20 * 100))

        # 典型价格与成交量
        tp  = (df["$open"] + df["$high"] + df["$low"] + df["$close"]) / 4.0
        vol = df["$volume"]
        
        # 按"日期"聚合成日级 VWAP
        day_key = pd.Index(dt.date)
        def window_vwap(mask):
            num = (tp.where(mask, 0.0) * vol.where(mask, 0.0)).groupby(day_key).sum()
            den = vol.where(mask, 0.0).groupby(day_key).sum()
            wv  = (num / den.replace(0.0, np.nan)).astype("float64")
            wv = wv.fillna(tp.where(mask).groupby(day_key).mean())
            wv = wv.fillna(df["$close"].groupby(day_key).last())
            return wv

        today_day = window_vwap(mask_today)
        next_day  = window_vwap(mask_next).shift(1)

        # 广播回分钟
        today_map = today_day.to_dict()
        next_map  = next_day.to_dict()
        days = pd.Index(dt.date)
        today_series = pd.Series([today_map.get(d, np.nan) for d in days], index=df.index, dtype="float64")
        next_series  = pd.Series([next_map.get(d, np.nan)  for d in days], index=df.index, dtype="float64")

        lbl = next_series / today_series - 1
        mi = pd.MultiIndex.from_arrays([[inst]*len(lbl), df.index], names=["instrument","datetime"])
        return pd.DataFrame({"LABEL0": lbl.values}, index=mi)
    
    def setup_data(self, *args, **kwargs):
        """重写setup_data，在数据加载时计算自定义标签"""
        print(f"setup_data 开始，_data 存在: {hasattr(self, '_data')}")
        if hasattr(self, '_data'):
            print(f"_data 形状: {self._data.shape if self._data is not None else 'None'}")
        
        # 先调用父类方法加载特征数据
        super().setup_data(*args, **kwargs)
        
        print(f"父类setup_data后，_data 存在: {hasattr(self, '_data')}")
        if hasattr(self, '_data'):
            print(f"_data 形状: {self._data.shape if self._data is not None else 'None'}")
        else:
            print("父类setup_data后，_data属性不存在！")
            # 手动设置_data属性
            print("尝试手动设置_data属性...")
            try:
                self._data = self.data_loader.load(self.instruments, self.start_time, self.end_time)
                print(f"手动设置_data成功，形状: {self._data.shape if self._data is not None else 'None'}")
            except Exception as e:
                print(f"手动设置_data失败: {e}")
        
        # 计算自定义标签
        custom_labels = self._load_custom_labels(
            self.instruments, 
            self.start_time, 
            self.end_time
        )
        
        # 将自定义标签添加到数据中
        if not custom_labels.empty and hasattr(self, '_data') and self._data is not None:
            # 对齐索引
            common_index = self._data.index.intersection(custom_labels.index)
            if len(common_index) > 0:
                # 创建多级列名的标签数据
                label_multi = pd.DataFrame(
                    custom_labels.loc[common_index, "LABEL0"].values,
                    index=common_index,
                    columns=pd.MultiIndex.from_tuples([('label', 'LABEL0')])
                )
                
                # 合并到现有数据中
                self._data = pd.concat([self._data, label_multi], axis=1)
                print(f"自定义标签已添加到数据中，形状: {self._data.shape}")
            else:
                print(f"警告：没有共同的索引，_data索引: {self._data.index[:5]}, custom_labels索引: {custom_labels.index[:5]}")
        else:
            print(f"警告：无法添加自定义标签，_data属性存在: {hasattr(self, '_data')}, _data为None: {self._data is None if hasattr(self, '_data') else 'N/A'}")


class HighFreqBacktestHandler(DataHandler):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        drop_raw=True,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

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
            drop_raw=drop_raw,
        )

    def get_feature_config(self):
        fields = []
        names = []

        template_if = "If(IsNull({1}), {0}, {1})"
        template_paused = "Select(Or(IsNull($paused), Eq($paused, 0.0)), {0})"
        template_fillnan = "BFillNan(FFillNan({0}))"
        template_vwap = "Div(Add(Add(Mul($close, 2), Add($high, $low)), Add($open, $close)), 6)"
        template_returns = "Ref(Div(Sub($close, Ref($close, 1)), Ref($close, 1)), -1)"
        template_volume = "Mul($volume, Div(Add($close, $open), 2))"

        # 基础特征
        fields.extend([
            template_fillnan.format("$open"),
            template_fillnan.format("$high"),
            template_fillnan.format("$low"),
            template_fillnan.format("$close"),
            template_fillnan.format("$volume"),
        ])
        names.extend(["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])

        # 技术指标
        fields.extend([
            template_fillnan.format(template_vwap),
            template_fillnan.format("Mean($close, 5)"),
            template_fillnan.format("Mean($close, 10)"),
            template_fillnan.format("Mean($close, 20)"),
            template_fillnan.format("Std($close, 5)"),
            template_fillnan.format("Std($close, 10)"),
            template_fillnan.format("Std($close, 20)"),
        ])
        names.extend(["VWAP", "MA5", "MA10", "MA20", "STD5", "STD10", "STD20"])

        # 收益率特征
        fields.extend([
            template_fillnan.format(template_returns),
            template_fillnan.format("Ref($close, -1) / $close - 1"),
            template_fillnan.format("Ref($close, -2) / $close - 1"),
            template_fillnan.format("Ref($close, -3) / $close - 1"),
        ])
        names.extend(["RET_1", "RET_2", "RET_3", "RET_4"])

        # 成交量特征
        fields.extend([
            template_fillnan.format(template_volume),
            template_fillnan.format("Mean($volume, 5)"),
            template_fillnan.format("Mean($volume, 10)"),
            template_fillnan.format("Mean($volume, 20)"),
        ])
        names.extend(["VOLUME_AMOUNT", "VOLUME_MA5", "VOLUME_MA10", "VOLUME_MA20"])

        return {
            "feature": (fields, names),
            "label": (["Ref($close, -1) / $close - 1"], ["LABEL0"]),
        }


class SafeHighFreqHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        drop_raw=True,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

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
            drop_raw=drop_raw,
        )

    def get_feature_config(self):
        fields = []
        names = []

        template_if = "If(IsNull({1}), {0}, {1})"
        template_paused = "Select(Or(IsNull($paused), Eq($paused, 0.0)), {0})"
        template_fillnan = "BFillNan(FFillNan({0}))"
        template_vwap = "Div(Add(Add(Mul($close, 2), Add($high, $low)), Add($open, $close)), 6)"
        template_returns = "Ref(Div(Sub($close, Ref($close, 1)), Ref($close, 1)), -1)"
        template_volume = "Mul($volume, Div(Add($close, $open), 2))"

        # 基础特征
        fields.extend([
            template_fillnan.format("$open"),
            template_fillnan.format("$high"),
            template_fillnan.format("$low"),
            template_fillnan.format("$close"),
            template_fillnan.format("$volume"),
        ])
        names.extend(["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])

        # 技术指标
        fields.extend([
            template_fillnan.format(template_vwap),
            template_fillnan.format("Mean($close, 5)"),
            template_fillnan.format("Mean($close, 10)"),
            template_fillnan.format("Mean($close, 20)"),
            template_fillnan.format("Std($close, 5)"),
            template_fillnan.format("Std($close, 10)"),
            template_fillnan.format("Std($close, 20)"),
        ])
        names.extend(["VWAP", "MA5", "MA10", "MA20", "STD5", "STD10", "STD20"])

        # 收益率特征
        fields.extend([
            template_fillnan.format(template_returns),
            template_fillnan.format("Ref($close, -1) / $close - 1"),
            template_fillnan.format("Ref($close, -2) / $close - 1"),
            template_fillnan.format("Ref($close, -3) / $close - 1"),
        ])
        names.extend(["RET_1", "RET_2", "RET_3", "RET_4"])

        # 成交量特征
        fields.extend([
            template_fillnan.format(template_volume),
            template_fillnan.format("Mean($volume, 5)"),
            template_fillnan.format("Mean($volume, 10)"),
            template_fillnan.format("Mean($volume, 20)"),
        ])
        names.extend(["VOLUME_AMOUNT", "VOLUME_MA5", "VOLUME_MA10", "VOLUME_MA20"])

        return {
            "feature": (fields, names),
            "label": (["Ref($close, -1) / $close - 1"], ["LABEL0"]),
        }

    def _load_internal(self, instrument, start_index, end_index, freq):
        import warnings
        import logging
        
        try:
            data = super()._load_internal(instrument, start_index, end_index, freq)
            
            if data is None or data.empty:
                msg = f"[SafeHighFreqHandler] 股票 {instrument} 数据为空"
                warnings.warn(msg)
                logging.warning(msg)
                return None

            return data

        except Exception as e:
            msg = f"[SafeHighFreqHandler] 加载股票 {instrument} 数据时出错: {e}, 跳过该股票"
            warnings.warn(msg)
            logging.warning(msg)
            return None




