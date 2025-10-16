import numpy as np
import pandas as pd
import importlib
from qlib.data.ops import ElemOperator, PairOperator
from qlib.config import C
from qlib.data.cache import H
from qlib.data.data import Cal
from qlib.contrib.ops.high_freq import get_calendar_day


class DayLast(ElemOperator):
    """DayLast Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        a series of that each value equals the last value of its day
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = get_calendar_day(freq=freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.groupby(_calendar[series.index], group_keys=False).transform("last")

#修改label需要新增第二天开盘价

class DayFirst(ElemOperator):
    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = get_calendar_day(freq=freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.groupby(_calendar[series.index], group_keys=False).transform("first")


class FFillNan(ElemOperator):
    """FFillNan Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        a forward fill nan feature
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.ffill()


class BFillNan(ElemOperator):
    """BFillNan Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        a backfoward fill nan feature
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.bfill()


class Date(ElemOperator):
    """Date Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        a series of that each value is the date corresponding to feature.index
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        _calendar = get_calendar_day(freq=freq)
        series = self.feature.load(instrument, start_index, end_index, freq)
        return pd.Series(_calendar[series.index], index=series.index)


class Select(PairOperator):
    """Select Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance, select condition
    feature_right : Expression
        feature instance, select value

    Returns
    ----------
    feature:
        value(feature_right) that meets the condition(feature_left)

    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series_condition = self.feature_left.load(instrument, start_index, end_index, freq)
        series_feature = self.feature_right.load(instrument, start_index, end_index, freq)
        # Align condition index to feature index to avoid unalignable boolean Series
        mask = series_condition.reindex(series_feature.index, fill_value=False)
        return series_feature[mask]


class IsNull(ElemOperator):
    """IsNull Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    feature:
        A series indicating whether the feature is nan
    """

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.isnull()


class Cut(ElemOperator):
    """Cut Operator

    Parameters
    ----------
    feature : Expression
        feature instance
    l : int
        l > 0, delete the first l elements of feature (default is None, which means 0)
    r : int
        r < 0, delete the last -r elements of feature (default is None, which means 0)
    Returns
    ----------
    feature:
        A series with the first l and last -r elements deleted from the feature.
        Note: It is deleted from the raw data, not the sliced data
    """

    def __init__(self, feature, l=None, r=None):
        self.l = l
        self.r = r
        if (self.l is not None and self.l <= 0) or (self.r is not None and self.r >= 0):
            raise ValueError("Cut operator l should > 0 and r should < 0")

        super(Cut, self).__init__(feature)

    def _load_internal(self, instrument, start_index, end_index, freq):
        series = self.feature.load(instrument, start_index, end_index, freq)
        return series.iloc[self.l : self.r]

    def get_extended_window_size(self):
        ll = 0 if self.l is None else self.l
        rr = 0 if self.r is None else abs(self.r)
        lft_etd, rght_etd = self.feature.get_extended_window_size()
        lft_etd = lft_etd + ll
        rght_etd = rght_etd + rr
        return lft_etd, rght_etd

#修改label第二版 改成vwap版本VWAP≈(sum(close*volume)/sum(volume))要改   
#当日 14:41 - 14:50 区间的 VWAP和 次日 10:11 - 10:20 区间的 VWAP
#  隔夜收益率 = 次日 10:20 VWAP ÷ 当日 14:50 VWAP - 1   


class IntradayWindowVWAP(ElemOperator):
    def __init__(self, feature, start_hm: float, end_hm: float, volume_field: str = "$volume"):
        self.start_hm = start_hm
        self.end_hm = end_hm
        self.volume_field = volume_field
        super().__init__(feature)

    def _load_internal(self, instrument, start_index, end_index, freq):
        from qlib.data import D
        import numpy as np
        import pandas as pd
        from qlib.data.data import ExpressionD

        # 修复：处理 self.feature 可能是字符串的情况
        if isinstance(self.feature, str):
            # 如果 self.feature 是字符串，直接使用 D.features 获取索引
            # 使用原始方法避免 Monkey Patch 递归
            from qlib.data.data import ExpressionD
            original_expression = ExpressionD.expression
            try:
                idx_series = original_expression(instrument, self.feature, start_index, end_index, freq)
            except:
                # 如果原始方法失败，创建简单的 Series
                idx_series = pd.Series(index=pd.date_range(start_index, end_index, freq=freq), dtype="float64")
        else:
            # 如果 self.feature 是对象，调用其 load 方法
            idx_series = self.feature.load(instrument, start_index, end_index, freq)
                
        if idx_series is None or len(idx_series) == 0:
            return idx_series

        # 通过 ExpressionD 加载基础字段，避免在表达式上下文内再次调用 D.features 引起不一致
        def load_field(field: str):
            try:
                return ExpressionD.expression(instrument, field, start_index, end_index, freq)
            except Exception:
                return None

        s_open = load_field("$open")
        s_high = load_field("$high")
        s_low  = load_field("$low")
        s_close= load_field("$close")
        s_vol  = load_field(self.volume_field) or load_field("$volume")
        s_amount = load_field("$amount")

        # 直接对齐到表达式请求的索引，若其不是时间索引则回退为收盘价索引
        target_index = idx_series.index
        if not isinstance(target_index, pd.DatetimeIndex):
            if s_close is not None and isinstance(s_close.index, pd.DatetimeIndex):
                target_index = s_close.index
            else:
                # 无法获得时间索引，返回 NaN
                return pd.Series(np.nan, index=idx_series.index, dtype="float64")
        if s_open is None or s_high is None or s_low is None or s_close is None:
            return pd.Series(np.nan, index=target_index, dtype="float64")

        df = pd.DataFrame({
            "$open": s_open.reindex(target_index),
            "$high": s_high.reindex(target_index),
            "$low":  s_low.reindex(target_index),
            "$close":s_close.reindex(target_index),
        }, index=target_index)

        # 计算 VWAP
        tp = (df["$open"] + df["$high"] + df["$low"] + df["$close"]) / 4.0
        vol = s_vol.reindex(target_index) if s_vol is not None else None
        if (vol is None) or (vol.fillna(0.0).sum() == 0.0 and s_amount is not None):
            with pd.option_context('mode.use_inf_as_na', True):
                vol = (s_amount.reindex(target_index) / df["$close"]).replace([float("inf"), -float("inf")], pd.NA)
        if vol is None:
            vol = pd.Series(0.0, index=target_index)
        # 计算时间窗口掩码
        time_index = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index, errors="coerce")
        if not isinstance(time_index, pd.DatetimeIndex) or time_index.isna().any():
            return pd.Series(np.nan, index=idx_series.index, dtype="float64")
        t_int = time_index.hour * 100 + time_index.minute
        mask = (t_int >= int(round(self.start_hm * 100))) & (t_int <= int(round(self.end_hm * 100)))
        valid = mask & tp.notna() & vol.notna()
        

        # 按交易日聚合
        day = pd.Index(df.index.date)
        num = (tp * vol).where(valid, 0.0).groupby(day).sum()
        den = vol.where(valid, 0.0).groupby(day).sum()
        day_vwap = (num / den.replace(0.0, np.nan)).astype("float64")
        # 兜底：若窗口无成交量，用窗口内收盘简单均值代替；再兜底用当日收盘
        window_mean = tp.where(valid).groupby(day).mean()
        day_close = df["$close"].groupby(day).last()
        day_vwap = day_vwap.fillna(window_mean).fillna(day_close)
        

        # 将日 VWAP 映射回分钟索引（逐日广播）——严格使用 DatetimeIndex 的 target_index
        idx_dates = pd.Index(pd.to_datetime(target_index).date)
        day_map = day_vwap.to_dict()
        out_vals = [day_map.get(d, float("nan")) for d in idx_dates]
        out = pd.Series(out_vals, index=target_index, dtype="float64")
        return out
        

    def get_extended_window_size(self):
        return self.feature.get_extended_window_size()


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