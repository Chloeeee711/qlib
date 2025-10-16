# -*- coding: utf-8 -*-
import sys, os
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE)

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.ops import Operators
from qlib.config import C
import pandas as pd

def main():
    # register custom ops and force threading backend to keep registration in workers
    from highfreq_ops import DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut, DayFirst, IntradayWindowVWAP, DayShift
    Operators.register([DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut, DayFirst, IntradayWindowVWAP, DayShift])
    C.kernels = 1
    C.joblib_backend = 'threading'
    KQ_DATA_DIR = r"C:\\Users\\ASUS\\qlib_data"
    print("init qlib...")
    qlib.init(provider_uri=KQ_DATA_DIR, region=REG_CN)

    handler_config = {
        "instruments": ["SH000300"],
        "start_time": "2025-05-06 09:30:00",
        "end_time":   "2025-05-08 14:59:00",
        "drop_raw": False,
    }

    dataset_task = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "HighFreqHandler",
                "module_path": "highfreq_handler",
                "kwargs": handler_config,
            },
            "segments": {
                "train": ("2025-05-06", "2025-05-07"),
                "test":  ("2025-05-08", "2025-05-08"),
            },
        },
    }

    print("init dataset...")
    dataset = init_instance_by_config(dataset_task)
    hd = dataset.handler
    assert hasattr(hd, "_data") and hd._data is not None, "handler._data is None!"

    # handler-level label (ground truth)
    lbl_h = hd._data[("label","LABEL0")]
    d_h = pd.to_datetime(hd._data.index.get_level_values("datetime")).date
    nunq_h = pd.Series(lbl_h.values, index=d_h).groupby(level=0).nunique()
    print("handler per-day nunique unique:", nunq_h.unique())
    print("handler first-day all NaN:", lbl_h[d_h == d_h.min()].isna().all())
    print("handler head label:\n", lbl_h.head(12))

    # dataset-level
    train_df = dataset.prepare("train", col_set=["feature","label"], data_key=DataHandlerLP.DK_L)
    test_df  = dataset.prepare("test",  col_set=["feature","label"], data_key=DataHandlerLP.DK_I)

    lab_all = lbl_h.to_frame()
    lab_all.columns = pd.MultiIndex.from_tuples([("label","LABEL0")])

    # join to align (non-unique multiindex safe)
    train_join = train_df.drop(columns=[("label","LABEL0")], errors="ignore").join(lab_all, how="left")
    d_tr = pd.to_datetime(train_join.index.get_level_values("datetime")).date
    nunq_tr = pd.Series(train_join[("label","LABEL0")].values, index=d_tr).groupby(level=0).nunique()
    print("train per-day nunique unique (joined):", nunq_tr.unique())
    print("train first-day all NaN (joined):", train_join[("label","LABEL0")][d_tr == d_tr.min()].isna().all())
    print("train head label (joined):\n", train_join[("label","LABEL0")].head(12))

    # raw train df
    if ("label","LABEL0") in train_df.columns:
        d_raw = pd.to_datetime(train_df.index.get_level_values("datetime")).date
        nunq_raw = pd.Series(train_df[("label","LABEL0")].values, index=d_raw).groupby(level=0).nunique()
        print("train per-day nunique unique (raw df):", nunq_raw.unique())
        print("train first-day all NaN (raw df):", train_df[("label","LABEL0")][d_raw == d_raw.min()].isna().all())
        print("train head label (raw df):\n", train_df[("label","LABEL0")].head(12))

    print("\nDONE")

if __name__ == "__main__":
    main()


