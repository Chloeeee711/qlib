import os
import sys
import warnings
import numpy as np
import pandas as pd


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        print(*[str(a).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore") for a in args], **kwargs)


def corr_alignment(pred_path: str, sample_size: int = 2000):
    from qlib import init
    from qlib.data import D

    init(provider_uri="qlib_data")

    pred = pd.read_pickle(pred_path)
    if isinstance(pred, pd.DataFrame) and pred.shape[1] == 1:
        pred = pred.iloc[:, 0]
    # ensure MultiIndex names
    if not isinstance(pred.index, pd.MultiIndex):
        raise ValueError("pred.pkl must have a MultiIndex [instrument, datetime]")
    pred.index = pred.index.set_names(["instrument", "datetime"])

    # keep 14:40
    mask = pred.index.get_level_values("datetime").time == pd.Timestamp("14:40").time()
    pred_1440 = pred[mask]
    if len(pred_1440) == 0:
        safe_print("No 14:40 entries in pred.pkl")
        return None, None
    if len(pred_1440) > sample_size:
        pred_1440 = pred_1440.sample(sample_size, random_state=42)

    def realized(c_close: pd.Series, buy_ts: pd.Timestamp, sell_ts: pd.Timestamp):
        s_buy = c_close[c_close.index >= buy_ts]
        s_sell = c_close[c_close.index >= sell_ts]
        if len(s_buy) == 0 or len(s_sell) == 0:
            return np.nan
        buy = float(s_buy.iloc[0])
        sell = float(s_sell.iloc[0])
        if not np.isfinite(buy) or buy <= 0 or not np.isfinite(sell):
            return np.nan
        return sell / buy - 1.0

    vals, r0s, r1s = [], [], []
    for (ins, ts), v in pred_1440.items():
        start = pd.Timestamp(ts.date()) - pd.Timedelta(days=1)
        end = pd.Timestamp(ts.date()) + pd.Timedelta(days=3)
        try:
            df = D.features([ins], ["$close"], start_time=start, end_time=end, freq="1min")
        except Exception:
            continue
        if df is None or df.empty:
            continue
        c = df.droplevel("feature").xs(ins, level="instrument")["$close"].rename("close").astype(float)
        buy0 = pd.Timestamp(ts.date()) + pd.Timedelta(hours=14, minutes=45)
        sell0 = (pd.Timestamp(ts.date()) + pd.Timedelta(days=1)) + pd.Timedelta(hours=10, minutes=46)
        buy1 = (pd.Timestamp(ts.date()) + pd.Timedelta(days=1)) + pd.Timedelta(hours=14, minutes=45)
        sell1 = (pd.Timestamp(ts.date()) + pd.Timedelta(days=2)) + pd.Timedelta(hours=10, minutes=46)
        r0 = realized(c, buy0, sell0)
        r1 = realized(c, buy1, sell1)
        if np.isfinite(r0) and np.isfinite(r1):
            vals.append(float(v))
            r0s.append(float(r0))
            r1s.append(float(r1))

    if len(vals) < 50:
        safe_print("Insufficient samples for correlation analysis")
        return None, None
    corr0 = float(np.corrcoef(vals, r0s)[0, 1])
    corr1 = float(np.corrcoef(vals, r1s)[0, 1])
    return corr0, corr1


def daily_topk_count(pred_path: str, topk: int = 5) -> pd.DataFrame:
    pred = pd.read_pickle(pred_path)
    if isinstance(pred, pd.DataFrame) and pred.shape[1] == 1:
        pred = pred.iloc[:, 0]
    pred.index = pred.index.set_names(["instrument", "datetime"])  # ensure names
    # use 14:40 snapshot and count how many topk per day
    dt = pred.index.get_level_values("datetime")
    mask = dt.time == pd.Timestamp("14:40").time()
    p = pred[mask]
    if len(p) == 0:
        raise RuntimeError("No 14:40 signals found in pred.pkl for daily topk count")
    # group by date and take topk per day
    rows = []
    for d, df in p.groupby(p.index.get_level_values("datetime").date):
        ser = df.droplevel("datetime")
        cnt = int(min(topk, ser.shape[0])) if ser.notna().any() else 0
        rows.append((pd.Timestamp(d), cnt))
    out = pd.DataFrame(rows, columns=["date", "topk_count"]).set_index("date").sort_index()
    return out


def main():
    warnings.filterwarnings("ignore", message=".*获取 .* 价格失败.*")
    warnings.filterwarnings("ignore", message="object of type 'numpy.float64' has no len()")

    pred_path = os.path.join(os.path.dirname(__file__), "pred.pkl")
    if not os.path.exists(pred_path):
        safe_print("pred.pkl not found at examples/highfreq/pred.pkl")
        sys.exit(1)

    # alignment check
    corr0, corr1 = corr_alignment(pred_path)
    safe_print("corr(pred, realized [14:45 -> next 10:46]):", corr0)
    safe_print("corr(pred, realized [next 14:45 -> next+1 10:46]):", corr1)
    if corr0 is not None and corr1 is not None:
        if corr1 > corr0 + 0.02:
            safe_print("LIKELY OFFSET: predictions align better with next-day-shifted window.")
        elif corr0 > corr1 + 0.02:
            safe_print("NO OFFSET: predictions align with same-day window as intended.")
        else:
            safe_print("UNCLEAR: correlations close.")

    # daily expected topk count from signals
    dc = daily_topk_count(pred_path, topk=5)
    safe_print("Daily expected topk selections at 14:40 (first 10 days):\n", dc.head(10))
    safe_print("Mean expected selections per day:", float(dc["topk_count"].mean()) if len(dc) else np.nan)


if __name__ == "__main__":
    main()


