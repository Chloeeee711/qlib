import os
import pickle
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.config import HIGH_FREQ_CONFIG
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.ops import Operators
from qlib.data.data import Cal
from qlib.tests.data import GetData

from highfreq_ops import get_calendar_day, DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut

from qlib.contrib.model.gbdt import LGBModel
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord

class HighfreqWorkflowAuto:
    SPEC_CONF = {"custom_ops": [DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut], "expression_cache": None}
    MARKET = "all"

    start_time = "2020-09-15 00:00:00"
    end_time = "2021-01-18 16:00:00"
    train_end_time = "2020-11-30 16:00:00"
    test_start_time = "2020-12-01 00:00:00"

    DATA_HANDLER_CONFIG0 = {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": start_time,
        "fit_end_time": train_end_time,
        "instruments": MARKET,
        "infer_processors": [
            {"class": "HighFreqNorm", "module_path": "highfreq_processor"},
            {"class": "VWAPLabelProcessor", "module_path": "highfreq_vwaplabel_processor"}  # 自定义Label
        ],
    }
    DATA_HANDLER_CONFIG1 = {
    "start_time": start_time,
    "end_time": end_time,
    "fit_start_time": start_time,
    "fit_end_time": train_end_time,
    "instruments": MARKET,
    "infer_processors": [
        {"class": "HighFreqNorm", "module_path": "highfreq_processor"},
        {"class": "VWAPLabelProcessor", "module_path": "highfreq_vwaplabel_processor"}  # 自定义 label
    ],
}

    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 8,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "HighFreqHandler",
                    "module_path": "highfreq_handler",
                    "kwargs": DATA_HANDLER_CONFIG0,
                },
                "segments": {
                    "train": (start_time, train_end_time),
                    "test": (test_start_time, end_time),
                },
            },
        },
        "dataset_backtest": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "HighFreqBacktestHandler",
                    "module_path": "highfreq_handler",
                    "kwargs": DATA_HANDLER_CONFIG1,
                },
                "segments": {
                    "train": (start_time, train_end_time),
                    "test": (test_start_time, end_time),
                },
            },
        },
    }

    def _init_qlib(self):
        print("Initializing Qlib...")
        QLIB_INIT_CONFIG = {**HIGH_FREQ_CONFIG, **self.SPEC_CONF}
        provider_uri = QLIB_INIT_CONFIG.get("provider_uri")
        GetData().qlib_data(target_dir=provider_uri, interval="1min", region=REG_CN, exists_skip=True)
        qlib.init(**QLIB_INIT_CONFIG)

    def _prepare_calender_cache(self):
        print("Preparing calendar cache...")
        Cal.calendar(freq="1min")
        get_calendar_day(freq="1min")

    def train_and_backtest(self):
        self._init_qlib()
        self._prepare_calender_cache()

        print("Preparing dataset...")
        dataset = init_instance_by_config(self.task["dataset"])
        dataset_backtest = init_instance_by_config(self.task["dataset_backtest"])

        print("Training model...")
        model = init_instance_by_config(self.task["model"])
        with R.start(experiment_name="highfreq_train_model"):
            R.log_params(**flatten_dict({"model": self.task["model"], "dataset": {"handler": "HighFreqHandler"}}))
            model.fit(dataset)
            R.save_objects(trained_model=model)
            rid = R.get_recorder().id
        print("Training completed. RID:", rid)

        print("Generating feature importance...")
        rec_train = R.get_recorder(recorder_id=rid, experiment_name="highfreq_train_model")
        model = rec_train.load_object("trained_model")
        fi = model.get_feature_importance()
        df_imp = pd.DataFrame(fi.items(), columns=["feature", "gain"]).sort_values("gain", ascending=False)
        print(df_imp.head(20))

        print("Running backtest...")
        port_analysis_config = {
            "executor": {
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {"time_per_step": "minute", "generate_portfolio_metrics": True},
            },
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {"model": model, "dataset": dataset, "topk": 50, "n_drop": 5},
            },
            "backtest": {
                "start_time": self.test_start_time,
                "end_time": self.end_time,
                "account": 100000000,
                "benchmark": "SH000300",
                "exchange_kwargs": {
                    "freq": "minute",
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            },
        }

        with R.start(experiment_name="highfreq_backtest_analysis"):
            rec_bt = R.get_recorder()
            sr = SignalRecord(model, dataset, rec_bt)
            sr.generate()
            par = PortAnaRecord(rec_bt, port_analysis_config, "minute")
            par.generate()

        print("Backtest finished.")

if __name__ == "__main__":
    workflow = HighfreqWorkflowAuto()
    workflow.train_and_backtest()
    