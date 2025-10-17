import os
import sys
import warnings
from typing import Tuple, Optional

import numpy as np
import pandas as pd


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        # Avoid issues with emojis on some terminals
        print(*[str(a).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore") for a in args], **kwargs)


def init_qlib():
    # Lazy import to avoid requiring qlib before needed
    import qlib
    from qlib.config import C

    warnings.filterwarnings("ignore", message=".*获取 .* 价格失败.*")
    warnings.filterwarnings("ignore", message="object of type 'numpy.float64' has no len()")

    # Try to detect a data path; fall back to default
    provider_uri = None
    # Prefer an explicit env var if present
    if os.environ.get("QLIB_DATA_PATH"):
        provider_uri = os.environ["QLIB_DATA_PATH"]
    else:
        # If repo has qlib_data/, use it
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        candidate = os.path.join(repo_root, "qlib_data")
        if os.path.isdir(candidate):
            provider_uri = candidate

    qlib.init(provider_uri=provider_uri)
    return qlib


def load_latest_recorder(preferred_experiment: str = "HF_MIN_BACKTEST"):
    from qlib.workflow import R

    # Try preferred experiment first
    try:
        R.start(experiment_name=preferred_experiment)
        recorders = R.list_recorders(experiment_name=preferred_experiment)
        if recorders:
            rid = recorders[-1].id if hasattr(recorders[-1], "id") else recorders[-1]
            return R.get_recorder(rid)
    except Exception:
        pass

    # Fallback: scan all experiments and pick most recent recorder
    exps = R.list_experiments()
    for exp in reversed(exps):
        try:
            R.start(experiment_name=exp)
            recs = R.list_recorders(experiment_name=exp)
            if recs:
                rid = recs[-1].id if hasattr(recs[-1], "id") else recs[-1]
                return R.get_recorder(rid)
        except Exception:
            continue
    raise RuntimeError("No recorders found in any experiment.")


def load_pred(recorder) -> pd.Series:
    # SignalRecord saves 'pred.pkl' by default
    try:
        pred = recorder.load_object("pred.pkl")
    except Exception:
        # Some setups nest under 'signals'
        pred = recorder.load_object("signals/pred.pkl")
    if isinstance(pred, pd.DataFrame) and pred.shape[1] == 1:
        pred = pred.iloc[:, 0]
    return pred


def compute_realized_return(c_close: pd.Series, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> float:
    # pick first bar on or after the timestamp
    s_buy = c_close[c_close.index >= start_ts]
    s_sell = c_close[c_close.index >= end_ts]
    if len(s_buy) == 0 or len(s_sell) == 0:
        return np.nan
    buy = float(s_buy.iloc[0])
    sell = float(s_sell.iloc[0])
    if not np.isfinite(buy) or buy <= 0 or not np.isfinite(sell):
        return np.nan
    return sell / buy - 1.0


def correlation_checks(pred: pd.Series, sample_size: int = 2000) -> Tuple[Optional[float], Optional[float]]:
    from qlib.data import D

    # Ensure MultiIndex with (instrument, datetime)
    if not isinstance(pred.index, pd.MultiIndex) or set(pred.index.names) != {"instrument", "datetime"}:
        pred.index = pred.index.set_names(["instrument", "datetime"])  # best effort

    # Keep only 14:40 timestamps
    idx_dt = pred.index.get_level_values("datetime")
    mask = idx_dt.time == pd.Timestamp("14:40").time()
    pred_1440 = pred[mask]
    if len(pred_1440) == 0:
        safe_print("No 14:40 signals found in predictions; cannot run alignment check.")
        return None, None

    if len(pred_1440) > sample_size:
        pred_1440 = pred_1440.sample(sample_size, random_state=42)

    vals = []
    reals_0 = []  # same-day window: [14:45, next 10:46]
    reals_1 = []  # next-day-shifted window: [next 14:45, next+1 10:46]

    for (ins, ts), v in pred_1440.items():
        # fetch minute close for instrument around two days
        start = pd.Timestamp(ts.date()) - pd.Timedelta(days=1)
        end = pd.Timestamp(ts.date()) + pd.Timedelta(days=3)
        try:
            df = D.features([ins], ["$close"], start_time=start, end_time=end, freq="1min")
        except Exception:
            continue
        if df is None or df.empty:
            continue
        c = df.droplevel("feature").xs(ins, level="instrument")["$close"].rename("close").astype(float)

        # Window 0: same-day → next morning
        buy0 = pd.Timestamp(ts.date()) + pd.Timedelta(hours=14, minutes=45)
        sell0 = (pd.Timestamp(ts.date()) + pd.Timedelta(days=1)) + pd.Timedelta(hours=10, minutes=46)
        r0 = compute_realized_return(c, buy0, sell0)

        # Window 1: next-day → next+1 morning
        buy1 = (pd.Timestamp(ts.date()) + pd.Timedelta(days=1)) + pd.Timedelta(hours=14, minutes=45)
        sell1 = (pd.Timestamp(ts.date()) + pd.Timedelta(days=2)) + pd.Timedelta(hours=10, minutes=46)
        r1 = compute_realized_return(c, buy1, sell1)

        if np.isfinite(r0) and np.isfinite(r1):
            vals.append(float(v))
            reals_0.append(float(r0))
            reals_1.append(float(r1))

    if len(vals) < 50:
        safe_print("Insufficient valid samples for correlation checks.")
        return None, None

    corr0 = float(np.corrcoef(vals, reals_0)[0, 1])
    corr1 = float(np.corrcoef(vals, reals_1)[0, 1])
    return corr0, corr1


def estimate_daily_trade_counts(recorder) -> pd.DataFrame:
    # Use positions snapshot to approximate trade counts at 14:45 and 10:46 per day
    try:
        pos_obj = recorder.load_object("portfolio_analysis/positions_normal_1min.pkl")
    except Exception as e:
        raise RuntimeError(f"Positions artifact not found: {e}")

    pos_df = pos_obj.get("position") if isinstance(pos_obj, dict) else pos_obj
    if not isinstance(pos_df, pd.DataFrame) or pos_df.empty:
        raise RuntimeError("Positions data invalid or empty.")

    pos_df = pos_df.copy()
    pos_df.index = pd.to_datetime(pos_df.index)

    def changes_at_time(df: pd.DataFrame, hhmm: str) -> int:
        snap = df[df.index.time == pd.Timestamp(hhmm).time()].sort_index()
        if len(snap) < 2:
            return 0
        # number of symbols with changed position
        return int((snap.iloc[-1] != snap.iloc[0]).sum())

    rows = []
    for d, day_df in pos_df.groupby(pos_df.index.date):
        cnt = 0
        cnt += changes_at_time(day_df, "14:45")
        cnt += changes_at_time(day_df, "10:46")
        rows.append((pd.Timestamp(d), cnt))

    out = pd.DataFrame(rows, columns=["date", "trade_count"]).set_index("date").sort_index()
    return out


def main():
    init_qlib()
    rec = load_latest_recorder(preferred_experiment="HF_MIN_BACKTEST")
    safe_print("Loaded recorder:", rec.id if hasattr(rec, "id") else rec)

    # 1) Load predictions and check time buckets
    pred = load_pred(rec)
    if pred is None or len(pred) == 0:
        safe_print("No predictions found in recorder. Aborting alignment check.")
    else:
        times = set(pred.index.get_level_values("datetime").time)
        has_1440 = pd.Timestamp("14:40").time() in times
        safe_print("Signals include 14:40:", has_1440)

        # Correlation checks to detect effective alignment
        corr_same, corr_next = correlation_checks(pred, sample_size=2000)
        safe_print("corr(pred, realized [14:45 -> next 10:46]):", corr_same)
        safe_print("corr(pred, realized [next 14:45 -> next+1 10:46]):", corr_next)
        if corr_same is not None and corr_next is not None:
            if corr_next > corr_same + 0.02:
                safe_print("LIKELY OFFSET: predictions align better with next-day-shifted window.")
            elif corr_same > corr_next + 0.02:
                safe_print("NO OFFSET: predictions align with same-day window as intended.")
            else:
                safe_print("UNCLEAR: correlations are similar; no strong evidence of offset.")

    # 2) Estimate daily trade counts from positions (reflects strategy execution, thus topk)
    try:
        dc = estimate_daily_trade_counts(rec)
        safe_print("Daily trade_count (first 10 days):\n", dc.head(10))
        mean_cnt = float(dc["trade_count"].mean()) if len(dc) else np.nan
        safe_print("Mean daily trade_count:", mean_cnt)
    except Exception as e:
        safe_print("Could not estimate daily trade counts:", e)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        safe_print("Verification failed:", e)
        sys.exit(1)
    sys.exit(0)


