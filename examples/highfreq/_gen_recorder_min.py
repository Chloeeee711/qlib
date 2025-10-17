import warnings
import pandas as pd
from qlib import init
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.executor import SimulatorExecutor
from qlib.data import D


def main():
    warnings.filterwarnings("ignore", message=".*获取 .* 价格失败.*")
    warnings.filterwarnings("ignore", message="object of type 'numpy.float64' has no len()")

    # Init qlib (use local qlib_data if present)
    init(provider_uri="qlib_data")

    # Build a dummy prediction series directly from calendar and instruments (minute freq)
    start, end = "2024-02-16", "2024-03-01"
    cal = D.calendar(start_time=start, end_time=end, freq="1min")
    # Keep times at or around 14:40 to ensure trading logic availability
    # But we will provide full-minute preds; strategy uses its own times
    ins_raw = D.instruments(market="all")
    try:
        ins_list = list(ins_raw)[:50]
    except Exception:
        # Fallback: try common pools
        for pool in ["csi300", "csi500", "all"]:
            try:
                ins_list = list(D.instruments(market=pool))
                break
            except Exception:
                continue
        ins_list = ins_list[:50]
    # Construct MultiIndex and zero predictions
    idx_tuples = []
    for dt in cal:
        for ins in ins_list:
            idx_tuples.append((ins, pd.Timestamp(dt)))
    mi = pd.MultiIndex.from_tuples(idx_tuples, names=["instrument", "datetime"])
    pred = pd.Series(0.0, index=mi)

    # Create experiment and recorder explicitly (context manager avoids API differences)
    with R.start(experiment_name="HF_MIN_BACKTEST", recorder_name="recorder"):
        recorder = R.get_recorder()

        # Log prediction directly as artifact 'pred.pkl'
        recorder.save_objects(**{"pred.pkl": pred})

        # Backtest with topk=5 to create positions artifacts using PandasSignal
        signal_cfg = {
            "class": "PandasSignal",
            "module_path": "qlib.backtest.signal",
            "kwargs": {"signal": pred},
        }
        strategy = TopkDropoutStrategy(signal=signal_cfg, topk=5, n_drop=0)
        executor = SimulatorExecutor(time_per_step="1min", generate_portfolio_metrics=True)
        par = PortAnaRecord(executor=executor, strategy=strategy, recorder=recorder)
        par.generate()

        rid = getattr(recorder, "id", None)
        print(f"Recorder ready: {rid}")


if __name__ == "__main__":
    main()


