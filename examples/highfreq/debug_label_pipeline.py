# =====================================================
# 文件名: highfreq_full.py
# 功能: 集成 highfreq_ops、highfreq_handler 和数据加载配置
# =====================================================

import os
import sys
import qlib
import pandas as pd
import numpy as np
from qlib.constant import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.ops import ElemOperator

# =====================================================
# 🔹 自定义算子: IntradayWindowVWAP
# =====================================================
class IntradayWindowVWAP(ElemOperator):
    def __init__(self, feature, start_hm: float, end_hm: float, volume_field: str = "$volume"):
        self.start_hm = start_hm
        self.end_hm = end_hm
        self.volume_field = volume_field
        super().__init__(feature)

    def _load_internal(self, instrument, start_index, end_index, freq):
        from qlib.data import D
        import pandas as pd
        import numpy as np

        if isinstance(self.feature, str):
            from qlib.data.data import ExpressionD
            try:
                idx_series = ExpressionD.expression(instrument, self.feature, start_index, end_index, freq)
            except:
                idx_series = pd.Series(index=pd.date_range(start_index, end_index, freq=freq), dtype="float64")
        else:
            idx_series = self.feature.load(instrument, start_index, end_index, freq)

        if idx_series is None or len(idx_series) == 0:
            return idx_series

        df = D.features(
            [instrument],
            ["$open", "$high", "$low", "$close", self.volume_field],
            start_time=start_index,
            end_time=end_index,
            freq=freq,
        )
        if df is None or len(df) == 0:
            return pd.Series(np.nan, index=idx_series.index, dtype="float64")

        if isinstance(df.index, pd.MultiIndex):
            lvl_name = 'instrument' if 'instrument' in df.index.names else df.index.names[0]
            if instrument in df.index.get_level_values(lvl_name):
                df = df.xs(instrument, level=lvl_name, drop_level=True)
            else:
                return pd.Series(np.nan, index=idx_series.index, dtype="float64")

        tp = (df["$open"] + df["$high"] + df["$low"] + df["$close"]) / 4.0
        vol = df[self.volume_field]
        t_int = df.index.hour * 100 + df.index.minute
        mask = (t_int >= int(round(self.start_hm * 100))) & (t_int <= int(round(self.end_hm * 100)))
        valid = mask & tp.notna() & vol.notna()
        day = pd.Index(df.index.date)
        num = (tp * vol).where(valid, 0.0).groupby(day).sum()
        den = vol.where(valid, 0.0).groupby(day).sum()
        day_vwap = (num / den.replace(0.0, np.nan)).astype("float64")

        out = pd.Series(np.nan, index=idx_series.index, dtype="float64")
        if len(day_vwap) > 0:
            vwap_value = day_vwap.iloc[0]
            if not pd.isna(vwap_value):
                out[:] = vwap_value
        return out

    def get_extended_window_size(self):
        return self.feature.get_extended_window_size()


# =====================================================
# 🔹 自定义算子: DayShift
# =====================================================
class DayShift(ElemOperator):
    def __init__(self, feature, k: int = 1):
        self.k = int(k)
        super().__init__(feature)

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        if series is None or len(series) == 0:
            return series

        dt_index = pd.to_datetime(series.index)
        day_key = pd.Index(dt_index.date)
        last_per_day = pd.Series(series.values, index=day_key).groupby(level=0).last()
        shifted = last_per_day.shift(self.k)
        shifted_map = shifted.to_dict()
        out = pd.Series([shifted_map.get(d, np.nan) for d in day_key], index=series.index, dtype="float64")
        return out

    def get_extended_window_size(self):
        return self.feature.get_extended_window_size()


# =====================================================
# 🔹 自定义 DataHandler
# =====================================================
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
        from highfreq_processor import check_transform_proc
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
        # Because there is no vwap field in the yahoo data, a method similar to Simpson integration is used to approximate vwap
        simpson_vwap = "($open + 2*$high + 2*$low + $close)/6"

        def get_normalized_price_feature(price_field, shift=0):
            """Get normalized price feature ops"""
            if shift == 0:
                template_norm = "Cut({0}/Ref(DayLast({1}), 240), 240, None)"
            else:
                template_norm = "Cut(Ref({0}, " + str(shift) + ")/Ref(DayLast({1}), 240), 240, None)"

            feature_ops = template_norm.format(
                template_if.format(
                    template_fillnan.format(template_paused.format("$close")),
                    template_paused.format(price_field),
                ),
                template_fillnan.format(template_paused.format("$close")),
            )
            return feature_ops

        fields += [get_normalized_price_feature("$open", 0)]
        fields += [get_normalized_price_feature("$high", 0)]
        fields += [get_normalized_price_feature("$low", 0)]
        fields += [get_normalized_price_feature("$close", 0)]
        fields += [get_normalized_price_feature(simpson_vwap, 0)]
        names += ["$open", "$high", "$low", "$close", "$vwap"]

        fields += [get_normalized_price_feature("$open", 240)]
        fields += [get_normalized_price_feature("$high", 240)]
        fields += [get_normalized_price_feature("$low", 240)]
        fields += [get_normalized_price_feature("$close", 240)]
        fields += [get_normalized_price_feature(simpson_vwap, 240)]
        names += ["$open_1", "$high_1", "$low_1", "$close_1", "$vwap_1"]

        fields += [
            "Cut({0}/Ref(DayLast(Mean({0}, 7200)), 240), 240, None)".format(
                "If(IsNull({0}), 0, If(Or(Gt({1}, Mul(1.001, {3})), Lt({1}, Mul(0.999, {2}))), 0, {0}))".format(
                    template_paused.format("$volume"),
                    template_paused.format(simpson_vwap),
                    template_paused.format("$low"),
                    template_paused.format("$high"),
                )
            )
        ]
        names += ["$volume"]
        fields += [
            "Cut(Ref({0}, 240)/Ref(DayLast(Mean({0}, 7200)), 240), 240, None)".format(
                "If(IsNull({0}), 0, If(Or(Gt({1}, Mul(1.001, {3})), Lt({1}, Mul(0.999, {2}))), 0, {0}))".format(
                    template_paused.format("$volume"),
                    template_paused.format(simpson_vwap),
                    template_paused.format("$low"),
                    template_paused.format("$high"),
                )
            )
        ]
        names += ["$volume_1"]

        # 附加原始辅助字段，便于作为特征参与训练（保持与现有模板一致的填充与暂停过滤）
        fields += [
            template_fillnan.format(template_paused.format("$factor")),
            template_fillnan.format(template_paused.format("$paused")),
            template_fillnan.format(template_paused.format("$change")),
            template_fillnan.format(template_paused.format("$paused_num")),
        ]
        names += ["$factor", "$paused", "$change", "$paused_num"]
       
        """
       # 严格区间 VWAP（当日 14:41-14:50）
        vwap_pm = "IntradayWindowVWAP($close, 14.41, 14.50, '$volume')"
       # 严格区间 VWAP（次日 10:11-10:20）
        vwap_am = "IntradayWindowVWAP($close, 10.11, 10.20, '$volume')"

        
        # LABEL：次日窗口 VWAP / 当日窗口 VWAP - 1；用 Ref(...,240) 跨日对齐
        #label_expr = "Cut(Ref({next_v}, -240) / {today_v} - 1, 240, None)".format(
        #next_v=vwap_1011_1020, today_v=vwap_1441_1450
        #)
        # 将“当日/次日”都抽成“日值”（同日恒等 → 取日末值即可），保证仅 1 个日标量
        #today_vwap_day = "DayLast({})".format(vwap_pm)
        today_vwap_day = "IntradayWindowVWAP($close, 14.41, 14.50, '$volume')"
        #am_vwap_day    = "DayLast({})".format(vwap_am)

        # 对“上午窗口的日值”做交易日+1 移位（次日），再广播回分钟
        #next_vwap_day  = "DayShift({}, 1)".format(am_vwap_day)
        next_vwap_day = "DayShift(IntradayWindowVWAP($close, 10.11, 10.20, '$volume'), 1)"

        # 标签：不再使用 Ref/Cut（按日移位已经对好了日），直接在分钟维度上计算
        today_vwap_day = vwap_pm
        next_vwap_day  = f"DayShift({vwap_am}, 1)"

         """
        # 隔夜收益率标签：使用字符串表达式（需要 Monkey Patch 支持）
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

import os
import sys
import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.utils import flatten_dict, init_instance_by_config
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.model.gbdt import LGBModel
from qlib.backtest import backtest
from workflow import HighfreqWorkflow
from highfreq_ops import IntradayWindowVWAP, DayShift
from qlib.data.ops import Operators
from qlib.contrib.ops.high_freq import Cut
# 确保当前路径包含 workflow.py
WORKDIR = r"C:\Users\ASUS\qlib\examples\highfreq"
sys.path.append(WORKDIR)

# 导入自定义操作符
from highfreq_ops import get_calendar_day, DayLast, FFillNan, BFillNan, Date, Select, IsNull, DayFirst,IntradayWindowVWAP, DayShift
Operators.register([IntradayWindowVWAP, DayShift, Cut])

# 配置自定义操作符
# Do NOT include built-in Cut here; only custom ops need registration
SPEC_CONF = {"custom_ops": [DayLast, DayFirst, FFillNan, BFillNan, Date, Select, IsNull, DayShift, IntradayWindowVWAP, Cut], "expression_cache": None}

print("✅ 自定义操作符已配置")

# 天勤数据配置
KQ_DATA_DIR = r"C:\Users\ASUS\qlib_data"
STOCK_POOL_CSV = r"C:\Users\ASUS\qlib\examples\highfreq\sorted_high_preclose_ratio_2025.csv"

# 天勤数据时间段配置
START_TIME = "2025-01-02 09:30:00"
END_TIME = "2025-09-29 14:57:00"
TRAIN_END_TIME = "2025-04-30 14:59:00"  # 1-4月用于训练
TEST_START_TIME = "2025-05-06 09:30:00"  # 5-9月用于测试和回测

CUSTOM_QLIB_CONFIG = {
    "provider_uri": KQ_DATA_DIR,  # 强制使用天勤数据路径
    "dataset_cache": None,
    "expression_cache": "DiskExpressionCache",
    "region": REG_CN,
    **SPEC_CONF  # 添加自定义操作符配置
}

qlib.init(**CUSTOM_QLIB_CONFIG)


stock_pool_df = pd.read_csv(STOCK_POOL_CSV)
stock_codes = stock_pool_df["code"].tolist()
print(f"📊 股票池包含 {len(stock_codes)} 支股票")

wf = HighfreqWorkflow()

wf.start_time = START_TIME
wf.train_end_time = TRAIN_END_TIME
wf.test_start_time = TEST_START_TIME
wf.end_time = END_TIME


# 拷贝 handler 配置并指定 provider_uri
handler_config = wf.DATA_HANDLER_CONFIG0.copy()
handler_config["instruments"] = stock_codes
#handler_config["provider_uri"] = KQ_DATA_DIR   # ⚠️ 关键
handler_config["infer_processors"] = [
    {"class": "HighFreqNorm", "module_path": "highfreq_processor",
     "kwargs": {"fit_start_time": wf.start_time,
                "fit_end_time": wf.train_end_time}}
]

# 2. 从 handler_config 中移除 class 和 module_path，避免传递给构造函数
if "class" in handler_config:
    del handler_config["class"]
if "module_path" in handler_config:
    del handler_config["module_path"]

from qlib.workflow import R
from qlib.utils import init_instance_by_config
from qlib.contrib.model.gbdt import LGBModel
import pandas as pd

# 明确指定训练特征和标签
features_columns = ["FEATURE_%d" % i for i in range(12 * 240)]  # HighFreqNorm 输出的列
#labels_columns = ["Ref($close, -1) / $close - 1"]  # 次日收益率作为示例
labels_columns = ["LABEL0"]
# dataset task
dataset_task = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "HighFreqHandler",   ## 使用我们修改过的 handler
            "module_path": "highfreq_handler",
            "kwargs": handler_config,
        },
        "segments": {
            "train": (wf.start_time, wf.train_end_time),
            "test": (wf.test_start_time, wf.end_time),
        },
        "infer_processors": handler_config["infer_processors"],
        "learn_processors": handler_config["infer_processors"],
        #"label": labels_columns,
        #"feature": features_columns,
        # ⚠️ 关键，给模型提供默认列映射
        #"use_cols": {
        #    "feature": features_columns,
        #   "label": labels_columns
        #}
    },
}


handler_cfg = dataset_task['kwargs']['handler']


# 修正时间段 和自定义数据时间段一致 否则会用默认路径时间 重要！！！
handler_cfg['kwargs'].update({
    'start_time': '2025-01-02 09:30:00',
    'end_time': '2025-09-29 14:57:00',
    'fit_start_time': '2025-01-02 09:30:00',
    'fit_end_time': '2025-04-30 14:59:00'
})

dataset_task['kwargs']['handler'] = handler_cfg

# 模型 task
model_task = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "learning_rate": 0.05,
        "n_estimators": 200,
        "num_leaves": 63,
    },
}

# 整体 task
task = {
    "dataset": dataset_task,
    "model": model_task,
}

from qlib.utils import init_instance_by_config

dataset = init_instance_by_config(dataset_task)

from qlib.data.dataset.handler import DataHandlerLP
train_df = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
xtrain = train_df["feature"]
ytrain = train_df["label"]

test_df = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
xtest = test_df["feature"]
ytest = test_df["label"]

if xtrain.size == 0 or xtest.size == 0:
    raise ValueError("❌ 数据为空！请检查股票池与 provider_uri 是否匹配天勤数据")
else:
    print(f"✅ 数据检查通过: 训练集 {xtrain.shape}, 测试集 {xtest.shape}")

print("训练集:", xtrain.shape, ytrain.shape)
print("样本预览:")
print(xtrain.head(300))
print(xtest.head(300))

train_df = dataset.prepare("train", col_set=["feature","label"], data_key="learn")
train_df["label"].head(3500)
