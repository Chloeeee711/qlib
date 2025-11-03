from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


RAW_DIR = Path(r"C:\Users\ASUS\kq_raw_data_ipynb")
QLIB_DIR = Path(r"C:\Users\ASUS\qlib_data_ipynb")


def load_calendar_1min(qdir: Path) -> pd.DatetimeIndex:
    cal_file = qdir / "calendars" / "1min.txt"
    if not cal_file.exists():
        raise FileNotFoundError(f"Calendar not found: {cal_file}")
    # 一行一个时间字符串
    cal = pd.read_csv(cal_file, header=None, names=["dt"], sep=",|\t|\s+", engine="python")
    cal = pd.to_datetime(cal["dt"].astype(str))
    return pd.DatetimeIndex(cal)


def market_prefix_convert(code: str) -> str:
    c = code.upper()
    if c.startswith("SSE_"):
        return "SH" + c.split("_")[1]
    if c.startswith("SZSE_"):
        return "SZ" + c.split("_")[1]
    if c.startswith("SSE."):
        return "SH" + c.split(".")[1]
    if c.startswith("SZSE."):
        return "SZ" + c.split(".")[1]
    return c


def ensure_float32(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.float32:
        return arr.astype(np.float32)
    return arr


def to_bin_with_index(out_path: Path, start_idx: int, data: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # qlib .bin 规范：首个 float32 为起始索引（相对日历），随后为数据
    payload = np.hstack([np.array([start_idx], dtype="<f4"), ensure_float32(data)]).astype("<f4")
    payload.tofile(str(out_path))


def compute_change(close_arr: np.ndarray) -> np.ndarray:
    if len(close_arr) == 0:
        return close_arr
    change = np.empty_like(close_arr)
    change[0] = 0.0
    change[1:] = close_arr[1:] - close_arr[:-1]
    return change


def read_csv_ohlcv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 兼容列名
    cols_map = {c.lower(): c for c in df.columns}
    for k in ["datetime", "open", "high", "low", "close", "volume"]:
        if k not in cols_map:
            # 容错：可能首字母大写
            if k.capitalize() in df.columns:
                cols_map[k] = k.capitalize()
            elif k.upper() in df.columns:
                cols_map[k] = k.upper()
    dt_col = cols_map.get("datetime")
    if dt_col is None:
        raise ValueError(f"{csv_path} 缺少 datetime 列")
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col])
    df = df.sort_values(dt_col)
    df = df.rename(columns={
        dt_col: "datetime",
        cols_map.get("open", "open"): "open",
        cols_map.get("high", "high"): "high",
        cols_map.get("low", "low"): "low",
        cols_map.get("close", "close"): "close",
        cols_map.get("volume", "volume"): "volume",
    })
    # 仅保留需要列
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    return df


def get_start_index(calendar: pd.DatetimeIndex, first_dt: pd.Timestamp) -> int:
    # 使用 searchsorted 加速定位
    pos = calendar.searchsorted(first_dt)
    if pos >= len(calendar) or calendar[pos] != first_dt:
        # 如果第一条分钟不在日历，提示并尽量对齐到最近的索引
        # 这里仍然返回 pos，使得后续读取能在正确位置起读
        pass
    return int(pos)


def convert_one(csv_path: Path, cal_1min: pd.DatetimeIndex, out_root: Path) -> Tuple[str, int]:
    code_raw = csv_path.stem  # e.g., SSE_600519
    inst_dir = out_root / "features" / market_prefix_convert(code_raw)

    df = read_csv_ohlcv(csv_path)
    if df.empty:
        return (code_raw, 0)

    first_dt = pd.Timestamp(df["datetime"].iloc[0])
    start_idx = get_start_index(cal_1min, first_dt)

    open_arr = df["open"].to_numpy(np.float32)
    high_arr = df["high"].to_numpy(np.float32)
    low_arr = df["low"].to_numpy(np.float32)
    close_arr = df["close"].to_numpy(np.float32)
    vol_arr = df["volume"].to_numpy(np.float32)
    factor_arr = np.ones_like(close_arr, dtype=np.float32)
    paused_arr = np.zeros_like(close_arr, dtype=np.float32)
    paused_num_arr = np.zeros_like(close_arr, dtype=np.float32)
    change_arr = compute_change(close_arr)

    series_list: List[Tuple[np.ndarray, str]] = [
        (open_arr, "open"),
        (high_arr, "high"),
        (low_arr, "low"),
        (close_arr, "close"),
        (vol_arr, "volume"),
        (factor_arr, "factor"),
        (paused_arr, "paused"),
        (paused_num_arr, "paused_num"),
        (change_arr, "change"),
    ]

    written = 0
    for arr, name in series_list:
        out_path = inst_dir / f"{name}.1min.bin"
        to_bin_with_index(out_path, start_idx, arr)
        written += 1

    return (code_raw, written)


def main():
    print("====== Qlib Minutes Converter (KQ CSV -> .bin) ======")
    print(f"RAW_DIR = {RAW_DIR}")
    print(f"QLIB_DIR = {QLIB_DIR}")

    if not RAW_DIR.exists():
        raise FileNotFoundError(RAW_DIR)
    if not QLIB_DIR.exists():
        raise FileNotFoundError(QLIB_DIR)

    cal = load_calendar_1min(QLIB_DIR)
    print(f"Loaded 1min calendar: {len(cal)} entries, range: {cal[0]} ~ {cal[-1]}")

    features_root = QLIB_DIR / "features"
    features_root.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in RAW_DIR")

    ok, fail = 0, 0
    for i, csv_path in enumerate(csv_files, 1):
        try:
            code, n = convert_one(csv_path, cal, QLIB_DIR)
            ok += 1
            if i <= 10:
                print(f"[{i}/{len(csv_files)}] OK {code}: wrote {n} bins")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(csv_files)}] FAIL {csv_path.name}: {e}")

    print(f"\nDone. success={ok}, failed={fail}")


if __name__ == "__main__":
    main()


