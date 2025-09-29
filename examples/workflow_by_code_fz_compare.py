import os
import sys
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.workflow import R
from qlib.utils import init_instance_by_config

# Ensure local repo import works when running from terminal
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ------------------------------
# Utility
# ------------------------------
def clean_Xy(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    Xc = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    yc = y.reindex(Xc.index).dropna()
    Xc = Xc.reindex(yc.index)
    return Xc, yc


def compute_daily_ic(pred: pd.Series, y: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"pred": pred, "label": y}).dropna()
    ic_by_day = df.groupby(level="datetime").apply(lambda d: d["pred"].corr(d["label"]))
    ic = ic_by_day.dropna()
    res = pd.DataFrame({
        "ic_mean": [ic.mean()],
        "ic_std": [ic.std(ddof=1)],
    })
    res["ic_ir"] = res["ic_mean"] / (res["ic_std"].replace(0, np.nan))
    return res, ic


def print_feature_importance(model, title: str, topn: int = 30):
    booster = model.model
    fi = pd.DataFrame({
        "feature": booster.dump_model().get("feature_names", []),
        "gain": booster.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    fi["is_fz"] = fi["feature"].astype(str).str.startswith("FZ_")
    total_gain = max(fi["gain"].sum(), 1e-12)
    fz_share = fi.loc[fi["is_fz"], "gain"].sum() / total_gain
    print(f"\n=== {title} ===")
    print(f"FZ_ 因子重要性占比: {fz_share:.2%}")
    print(fi.head(topn).to_string(index=False))


def maybe_backtest(pred_series: pd.Series, dataset: DatasetH, title: str):
    try:
        # Build a minimal workflow to evaluate portfolio
        from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
        from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
        from qlib.contrib.evaluate import risk_analysis

        # Align prediction to dataset index
        pred = pred_series.rename("score").sort_index()

        # Recorder
        R.start(exp_name=f"FZ_COMPARE_{title}")
        recorder = R.get_recorder()

        # Save signal
        sr_conf = {
            "class": "SignalRecord",
            "module_path": "qlib.workflow.record_temp",
            "kwargs": {
                "model": None,
                "dataset": dataset,
                "pred": pred,
            },
        }
        sr: SignalRecord = init_instance_by_config(sr_conf)
        sr.generate()

        # Portfolio analysis
        par_conf = {
            "class": "PortAnaRecord",
            "module_path": "qlib.workflow.record_temp",
            "kwargs": {
                "strategy": TopkDropoutStrategy(topk=50, n_drop=5),
                "executor": {
                    "class": "SimulatorExecutor",
                    "module_path": "qlib.backtest.executor",
                    "kwargs": {
                        "time_per_step": "day",
                        "generate_report": True,
                    },
                },
                "backtest": {
                    "start_time": dataset.segments["test"].start,  # use test period
                    "end_time": dataset.segments["test"].end,
                    "account": 1e9,
                    "benchmark": "SH000300",
                },
            },
        }
        par: PortAnaRecord = init_instance_by_config(par_conf)
        par.generate()

        # Risk metrics
        report = recorder.list_metrics()
        print(f"\n=== 回测结果（{title}）===")
        for k, v in report.items():
            print(k, ":", v)

    except Exception as e:
        warnings.warn(f"Backtest skipped: {e}")


def main():
    # Init qlib
    if not qlib.is_initialized():
        qlib.init(provider_uri=None, region=REG_CN)

    # Data segments
    segments = {
        "train": ("2008-01-01", "2014-12-31"),
        "valid": ("2015-01-01", "2016-12-31"),
        "test": ("2017-01-01", "2020-12-31"),
    }

    # Handler with only 7 FZ factors added in library; disable infer processors to avoid ProcessInf path
    handler = Alpha158(
        instruments="csi300",
        fit_start_time=segments["train"][0],
        fit_end_time=segments["train"][1],
        infer_processors=[],
        # keep default learn processors
    )

    dataset = DatasetH(handler, segments=segments)

    # Prepare data
    y_train = dataset.prepare("train", col_set="label")
    y_valid = dataset.prepare("valid", col_set="label")
    y_test = dataset.prepare("test", col_set="label")

    X_train_full = dataset.prepare("train", col_set="feature")
    X_valid_full = dataset.prepare("valid", col_set="feature")
    X_test_full = dataset.prepare("test", col_set="feature")

    # Baseline: filter out FZ_ cols
    cols_all = sorted(set(X_train_full.columns) | set(X_valid_full.columns) | set(X_test_full.columns))
    cols_no_fz = [c for c in cols_all if not str(c).startswith("FZ_")]

    Xb_train = X_train_full[cols_no_fz]
    Xb_valid = X_valid_full[cols_no_fz]
    Xb_test = X_test_full[cols_no_fz]

    # Clean
    Xb_train, yb_train = clean_Xy(Xb_train, y_train)
    Xb_valid, yb_valid = clean_Xy(Xb_valid, y_valid)
    Xb_test, yb_test = clean_Xy(Xb_test, y_test)

    Xf_train, yf_train = clean_Xy(X_train_full, y_train)
    Xf_valid, yf_valid = clean_Xy(X_valid_full, y_valid)
    Xf_test, yf_test = clean_Xy(X_test_full, y_test)

    # Models
    from qlib.contrib.model.gbdt import LGBModel

    lgb_params = dict(
        loss="mse",
        colsample_bytree=0.8879,
        learning_rate=0.05,
        subsample=0.8789,
        lambda_l1=205.6999,
        lambda_l2=580.9768,
        max_depth=8,
        num_leaves=210,
        num_threads=8,
        seed=42,
        min_data_in_leaf=50,
    )

    model_base = LGBModel(**lgb_params)
    model_fz = LGBModel(**lgb_params)

    print("\nTraining BASE (Alpha158 only)...")
    model_base.fit(Xb_train, yb_train, eval_set=[(Xb_valid, yb_valid)])

    print("\nTraining FZ (Alpha158 + 7 FZ)...")
    model_fz.fit(Xf_train, yf_train, eval_set=[(Xf_valid, yf_valid)])

    # Feature importance
    print_feature_importance(model_base, "BASE Top-30", topn=30)
    print_feature_importance(model_fz, "FZ Top-30", topn=30)

    # Predictions for IC
    pred_base = pd.Series(model_base.predict(Xb_test), index=Xb_test.index)
    pred_fz = pd.Series(model_fz.predict(Xf_test), index=Xf_test.index)

    # IC
    ic_base, ic_series_base = compute_daily_ic(pred_base, yb_test)
    ic_fz, ic_series_fz = compute_daily_ic(pred_fz, yf_test)
    print("\n=== IC (test) ===")
    print("BASE:")
    print(ic_base.to_string(index=False))
    print("FZ:")
    print(ic_fz.to_string(index=False))

    # Backtest (optional, best-effort)
    print("\nAttempting backtest (best-effort)...")
    maybe_backtest(pred_base, dataset, title="BASE")
    maybe_backtest(pred_fz, dataset, title="FZ")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()


