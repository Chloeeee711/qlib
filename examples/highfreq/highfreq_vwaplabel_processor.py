import numpy as np
import pandas as pd
from qlib.data import D
from qlib.data.dataset.processor import Processor


class VWAPLabelProcessor(Processor):
    def __init__(self, buy_window=("14:41", "14:50"), sell_window=("10:11", "10:20")):
        self.buy_window = buy_window
        self.sell_window = sell_window

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # 根据输入索引范围与标的，拉取原始分钟收盘与成交量，计算VWAP标签并合并回去
        if df.empty:
            return df.copy()

        # 解析时间范围与标的集合
        try:
            dt_index = df.index.get_level_values("datetime")
            inst_index = df.index.get_level_values("instrument")
        except Exception:
            # 若索引级别顺序不同，尝试互换
            if isinstance(df.index, pd.MultiIndex) and "datetime" in df.index.names and "instrument" in df.index.names:
                dt_index = df.index.get_level_values(df.index.names.index("datetime"))
                inst_index = df.index.get_level_values(df.index.names.index("instrument"))
            else:
                # 无法解析，多返回原样
                out = df.copy()
                out["LABEL"] = np.nan
                return out

        start_time = pd.Timestamp(dt_index.min()).strftime("%Y-%m-%d %H:%M:%S")
        end_time = pd.Timestamp(dt_index.max()).strftime("%Y-%m-%d %H:%M:%S")
        instruments = sorted(pd.Index(inst_index).unique().tolist())

        # 拉取原始数据
        raw = D.features(
            instruments,
            ["$close", "$volume"],
            start_time=start_time,
            end_time=end_time,
            freq="1min",
        )
        if raw.empty:
            out = df.copy()
            out["LABEL"] = np.nan
            return out

        # 规范列名与索引顺序为 (instrument, datetime)
        raw.columns = ["close", "volume"]
        if isinstance(raw.index, pd.MultiIndex) and raw.index.names == ["datetime", "instrument"]:
            raw = raw.swaplevel().sort_index()
        else:
            raw = raw.sort_index()

        # 计算每个标的、每个交易日的买入/卖出时段VWAP，并生成次日相对收益标签
        idx_inst = raw.index.get_level_values("instrument")
        idx_dt = raw.index.get_level_values("datetime")
        df_work = raw.copy()
        df_work["date"] = pd.to_datetime(idx_dt.date)
        df_work["time"] = pd.to_datetime(idx_dt).time

        buy_start = pd.to_datetime(self.buy_window[0]).time()
        buy_end = pd.to_datetime(self.buy_window[1]).time()
        sell_start = pd.to_datetime(self.sell_window[0]).time()
        sell_end = pd.to_datetime(self.sell_window[1]).time()

        labels_map = {}

        # 分组粒度: (instrument, date)
        for (inst, date), group in df_work.groupby([idx_inst, "date"], sort=False):
            buy_df = group[(group["time"] >= buy_start) & (group["time"] <= buy_end)]
            sell_df = group[(group["time"] >= sell_start) & (group["time"] <= sell_end)]

            if buy_df.empty or sell_df.empty:
                continue

            buy_vol = buy_df["volume"].sum()
            sell_vol = sell_df["volume"].sum()
            if buy_vol <= 0 or sell_vol <= 0:
                continue

            buy_vwap = (buy_df["close"] * buy_df["volume"]).sum() / buy_vol
            sell_vwap = (sell_df["close"] * sell_df["volume"]).sum() / sell_vol

            # 标签对应到 T+1 的日期
            next_date = (pd.Timestamp(date) + pd.Timedelta(days=1)).date()
            labels_map[(inst, next_date)] = sell_vwap / buy_vwap - 1.0

        # 将标签映射回分钟索引：同一日内的所有分钟都赋同一个 LABEL（对应上一交易日的买入日）
        out = df.copy()
        # 为 out 构造日期键，使用其 datetime 层的日期
        try:
            out_dates = pd.to_datetime(out.index.get_level_values("datetime")).date
            out_insts = out.index.get_level_values("instrument")
        except Exception:
            # 兼容不同顺序
            out_dates = pd.to_datetime(dt_index).date
            out_insts = inst_index

        label_vals = []
        for inst, d in zip(out_insts, out_dates):
            label_vals.append(labels_map.get((inst, d), np.nan))

        out["LABEL"] = np.asarray(label_vals, dtype=float)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df.index.get_level_values("datetime").date)
        df["time"] = pd.to_datetime(df.index.get_level_values("datetime")).time

        # 按日期分组计算 VWAP
        buy_vwap = {}
        sell_vwap = {}

        for date, group in df.groupby("date"):
            buy_mask = group["time"].between(pd.to_datetime(self.buy_window[0]).time(),
                                              pd.to_datetime(self.buy_window[1]).time())
            sell_mask = group["time"].between(pd.to_datetime(self.sell_window[0]).time(),
                                               pd.to_datetime(self.sell_window[1]).time())

            buy_df = group.loc[buy_mask]
            sell_df = group.loc[sell_mask]

            if buy_df.empty or sell_df.empty:
                continue

            buy_vwap[date] = (buy_df["close"] * buy_df["volume"]).sum() / buy_df["volume"].sum()
            sell_vwap[date + pd.Timedelta(days=1)] = (sell_df["close"] * sell_df["volume"]).sum() / sell_df["volume"].sum()

        # 生成 label
        labels = []
        for idx in df.index:
            date = idx[1].date()  # datetime
            if date in buy_vwap and date in sell_vwap:
                labels.append(sell_vwap[date] / buy_vwap[date] - 1)
            else:
                labels.append(np.nan)

        df["LABEL"] = labels
        return df[["LABEL"]]

