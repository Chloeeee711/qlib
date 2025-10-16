from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc
import highfreq_ops
from qlib.data.ops import Operators
try:
    # Ensure custom ops are registered in any worker importing this module
    from highfreq_ops import DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut, DayFirst, IntradayWindowVWAP, DayShift
    Operators.register([DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut, DayFirst, IntradayWindowVWAP, DayShift])
except Exception:
    # tolerate double registration or partial availability
    pass
import pandas as pd 
import numpy as np

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
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        
        # 存储自定义标签数据
        self._custom_labels = None

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self.get_feature_config(),
                "swap_level": False,
                "freq": "1min",
                "inst_processors": [],  # 明确指定为空，避免参数冲突
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
        """
        return {
            "feature": (fields, names),
            "label": (["Ref($close, -1) / $close - 1"], ["LABEL0"])  # 占位符，会被自定义标签覆盖
        }
    
    def _load_custom_labels(self, instruments, start_time, end_time):
        """计算自定义标签"""
        if self._custom_labels is not None:
            return self._custom_labels
            
        print("计算自定义隔夜收益率标签...")
        parts = []
        
        for inst in instruments:
            try:
                # 使用你的compute_label_single逻辑
                label_data = self._compute_label_single(inst, start_time, end_time)
                if not label_data.empty:
                    parts.append(label_data)
            except Exception as e:
                print(f"计算 {inst} 标签失败: {e}")
                continue
        
        if parts:
            self._custom_labels = pd.concat(parts)
        else:
            self._custom_labels = pd.DataFrame(columns=["LABEL0"])
            
        print(f"自定义标签计算完成，形状: {self._custom_labels.shape}")
        return self._custom_labels
    
    def _compute_label_single(self, inst, start_time, end_time):
        """单只股票的标签计算逻辑"""
        from qlib.data import D
        import pandas as pd
        import numpy as np
        
        def empty_label_df(inst):
            return pd.DataFrame(
                index=pd.MultiIndex.from_product([[inst], []], names=["instrument","datetime"]),
                columns=["LABEL0"]
            )
          
        # 拉分钟数据 - 确保有足够的时间范围来计算隔夜收益率
        try:
            # 扩展时间范围以确保有次日数据
            from datetime import datetime, timedelta
            # 兼容带时分秒/不同格式
            def _parse_dt(s):
                if s is None:
                    return None
                if isinstance(s, (datetime,)):
                    return s
                s = str(s)
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):  # 常见两种
                    try:
                        return datetime.strptime(s, fmt)
                    except Exception:
                        continue
                # 回退：仅取日期部分
                try:
                    return datetime.strptime(s.split(" ")[0], "%Y-%m-%d")
                except Exception:
                    return None

            s_dt = _parse_dt(start_time)
            e_dt = _parse_dt(end_time)
            if e_dt is None or s_dt is None:
                # 无法解析时间时，直接按传入字符串做一次尝试（D.features 仍可接受字符串）
                extended_end = end_time
            else:
                extended_end = (e_dt + timedelta(days=1)).strftime("%Y-%m-%d")

            df = D.features(
                [inst],
                ["$open", "$high", "$low", "$close", "$volume"],
                start_time=start_time,
                end_time=extended_end,
                freq="1min",
            )
        except Exception as e:
            print(f"⚠️ D.features 加载失败: inst={inst}, start={start_time}, end={end_time}, err={e}")
            return empty_label_df(inst)

        if df is None or df.empty:
            return empty_label_df(inst)

        # MultiIndex 安全切片
        if isinstance(df.index, pd.MultiIndex):
            lvl = "instrument" if "instrument" in df.index.names else df.index.names[0]
            inst_levels = df.index.get_level_values(lvl).unique()
            if inst not in inst_levels:
                return empty_label_df(inst)
            df = df.xs(inst, level=lvl, drop_level=True)

        if df.empty:
            return empty_label_df(inst)

        # 时间窗口掩码
        dt = pd.to_datetime(df.index)
        t_int = dt.hour * 100 + dt.minute
        mask_today = (t_int >= int(14.41 * 100)) & (t_int <= int(14.50 * 100))
        mask_next  = (t_int >= int(10.11 * 100)) & (t_int <= int(10.20 * 100))

        # 典型价格与成交量
        tp  = (df["$open"] + df["$high"] + df["$low"] + df["$close"]) / 4.0
        vol = df["$volume"]
        
        # 按"日期"聚合成日级 VWAP
        day_key = pd.Index(dt.date)
        def window_vwap(mask):
            num = (tp.where(mask, 0.0) * vol.where(mask, 0.0)).groupby(day_key).sum()
            den = vol.where(mask, 0.0).groupby(day_key).sum()
            wv  = (num / den.replace(0.0, np.nan)).astype("float64")
            wv = wv.fillna(tp.where(mask).groupby(day_key).mean())
            wv = wv.fillna(df["$close"].groupby(day_key).last())
            return wv

        # 定义“今天标签 = 明天上午 / 今天下午 - 1”
        afternoon_today = window_vwap(mask_today)          # 键为 D（当日）
        morning_today   = window_vwap(mask_next)           # 键为 D（当日）
        morning_tomorrow = morning_today.shift(-1)         # 键映射到 D+1（明日的上午）

        # 广播回分钟（使用 明天上午 / 今天下午 - 1）
        aft_map  = afternoon_today.to_dict()
        morn_map = morning_tomorrow.to_dict()
        days = pd.Index(dt.date)
        aft_series  = pd.Series([aft_map.get(d, np.nan) for d in days], index=df.index, dtype="float64")
        morn_series = pd.Series([morn_map.get(d, np.nan) for d in days], index=df.index, dtype="float64")

        lbl = morn_series / aft_series - 1
        mi = pd.MultiIndex.from_arrays([[inst]*len(lbl), df.index], names=["instrument","datetime"])
        return pd.DataFrame({"LABEL0": lbl.values}, index=mi)
    
    def setup_data(self, *args, **kwargs):
        """重写setup_data，在数据加载时计算自定义标签"""
        print(f"setup_data 开始，_data 存在: {hasattr(self, '_data')}")
        if hasattr(self, '_data'):
            print(f"_data 形状: {self._data.shape if self._data is not None else 'None'}")
        
        # 先调用父类方法加载特征数据
        super().setup_data(*args, **kwargs)
        
        print(f"父类setup_data后，_data 存在: {hasattr(self, '_data')}")
        if hasattr(self, '_data'):
            print(f"_data 形状: {self._data.shape if self._data is not None else 'None'}")
        else:
            print("父类setup_data后，_data属性不存在！")
            # 手动设置_data属性
            print("尝试手动设置_data属性...")
            try:
                self._data = self.data_loader.load(self.instruments, self.start_time, self.end_time)
                print(f"手动设置_data成功，形状: {self._data.shape if self._data is not None else 'None'}")
            except Exception as e:
                print(f"手动设置_data失败: {e}")
        
        # 扩展_data以包含次日数据（用于计算隔夜收益率）
        if hasattr(self, '_data') and self._data is not None:
            from datetime import datetime, timedelta
            # 处理时间格式，支持带时间的格式
            try:
                if ' ' in self.end_time:
                    end_dt = datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
                else:
                    end_dt = datetime.strptime(self.end_time, "%Y-%m-%d")
            except ValueError:
                # 如果格式不匹配，尝试其他格式
                try:
                    end_dt = datetime.strptime(self.end_time.split(' ')[0], "%Y-%m-%d")
                except:
                    print(f"❌ 无法解析时间格式: {self.end_time}")
                    return
            
            extended_end = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            
            print(f"🔍 扩展数据到: {extended_end}")
            try:
                # 加载扩展数据
                extended_data = self.data_loader.load(self.instruments, self.start_time, extended_end)
                if extended_data is not None and not extended_data.empty:
                    print(f"🔍 扩展数据形状: {extended_data.shape}")
                    # 合并数据
                    self._data = pd.concat([self._data, extended_data], axis=0)
                    print(f"🔍 合并后数据形状: {self._data.shape}")
                else:
                    print("❌ 扩展数据为空")
            except Exception as e:
                print(f"❌ 扩展数据失败: {e}")
        
        # 计算自定义标签（使用实际已加载的数据里的标的集合）
        if hasattr(self, "_data") and self._data is not None and len(self._data) > 0:
            try:
                inst_level = 'instrument' if 'instrument' in self._data.index.names else self._data.index.names[0]
                instruments_in_data = self._data.index.get_level_values(inst_level).unique().tolist()
            except Exception:
                instruments_in_data = self.instruments
        else:
            instruments_in_data = self.instruments

        custom_labels = self._load_custom_labels(
            instruments_in_data,
            self.start_time,
            self.end_time,
        )

        # 覆盖式写入标签列，避免产生重复列
        if hasattr(self, '_data') and self._data is not None and not custom_labels.empty:
            # 与 _data 完全对齐
            aligned = custom_labels.reindex(self._data.index)["LABEL0"]

            # 如果已有占位符或旧列，先删除再写入
            if ('label', 'LABEL0') in getattr(self._data, 'columns', []):
                try:
                    self._data.drop(columns=[('label', 'LABEL0')], inplace=True)
                except Exception:
                    pass

            # 将分钟级标签强制为“同日常数”：保证同一交易日内标签恒等，首日应为NaN
            try:
                dt_idx = pd.to_datetime(self._data.index.get_level_values('datetime'))
            except Exception:
                # 兜底：若无命名，取最后一级
                dt_idx = pd.to_datetime(self._data.index.get_level_values(-1))

            inst_idx = self._data.index.get_level_values(0)
            day_key = pd.MultiIndex.from_arrays([inst_idx, pd.Index(dt_idx.date)], names=['instrument','date'])

            # 计算每个(标的, 日)的日标签（取该日第一个非空值）
            aligned_df = pd.DataFrame({'label': aligned.values}, index=day_key)
            day_value = aligned_df.groupby(level=['instrument','date'])['label'].apply(lambda s: s.dropna().iloc[0] if s.dropna().size>0 else np.nan)

            # 广播回分钟索引
            day_map = day_value.to_dict()
            broadcast = pd.Series([day_map.get((i, d), np.nan) for i, d in zip(inst_idx, pd.Index(dt_idx.date))], index=self._data.index)

            # 覆盖式赋值（先删后写，确保唯一）到 _data
            self._data[('label', 'LABEL0')] = broadcast.values

            # 同步到学习/推理数据（不同 Qlib 版本命名可能不同，尽量覆盖常见属性）
            for buf_name in ['_learn', '_infer', '_data_l', '_data_i']:
                if hasattr(self, buf_name) and getattr(self, buf_name) is not None:
                    try:
                        buf = getattr(self, buf_name)
                        if ('label','LABEL0') in getattr(buf, 'columns', []):
                            try:
                                buf.drop(columns=[('label','LABEL0')], inplace=True)
                            except Exception:
                                pass
                        # 对齐并写入
                        aligned_buf = pd.Series(
                            [day_map.get((i, d), np.nan) for i, d in zip(
                                buf.index.get_level_values(0),
                                pd.to_datetime(buf.index.get_level_values('datetime'), errors='coerce').date
                                if 'datetime' in buf.index.names else
                                pd.to_datetime(buf.index.get_level_values(-1), errors='coerce').date
                            )],
                            index=buf.index
                        )
                        buf[('label','LABEL0')] = aligned_buf.values
                    except Exception:
                        continue

            nn = pd.Series(broadcast).notna().sum()
            print(f"自定义标签覆盖写入完成，按日常数化，非NaN数量: {nn}")
        else:
            print(f"警告：无法添加自定义标签，_data属性存在: {hasattr(self, '_data')}, _data为None: {self._data is None if hasattr(self, '_data') else 'N/A'}，custom_labels为空: {custom_labels.empty}")


class HighFreqBacktestHandler(DataHandler):
    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
    ):
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": self.get_feature_config(),
                "swap_level": False,
                "freq": "1min",
                "inst_processors": [],  # 明确指定为空，避免参数冲突
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
        )

    def get_feature_config(self):
        fields = []
        names = []

        template_if = "If(IsNull({1}), {0}, {1})"
        template_paused = "Select(Or(IsNull($paused), Eq($paused, 0.0)), {0})"
        template_fillnan = "BFillNan(FFillNan({0}))"
        # Because there is no vwap field in the yahoo data, a method similar to Simpson integration is used to approximate vwap
        simpson_vwap = "($open + 2*$high + 2*$low + $close)/6"
        fields += [
            "Cut({0}, 240, None)".format(template_fillnan.format(template_paused.format("$close"))),
        ]
        names += ["$close0"]
        fields += [
            "Cut({0}, 240, None)".format(
                template_if.format(
                    template_fillnan.format(template_paused.format("$close")),
                    template_paused.format(simpson_vwap),
                )
            )
        ]
        names += ["$vwap0"]
        return fields, names



import warnings
import os
import numpy as np
from qlib.data.dataset.handler import DataHandler
from highfreq_handler import HighFreqHandler  # 确保引入你原来的 HighFreqHandler
import logging

# 创建日志记录器
log_path = os.path.expanduser("~/.qlib/safe_handler_skip.log")
logging.basicConfig(
    filename=log_path,
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class SafeHighFreqHandler(HighFreqHandler):
    def _load_internal(self, instrument, start_index, end_index, *args, **kwargs):
        try:
            data = super()._load_internal(instrument, start_index, end_index, *args, **kwargs)

            if data is None or len(data) == 0 or (hasattr(data, "shape") and np.prod(data.shape) == 0):
                msg = f"[SafeHighFreqHandler] 股票 {instrument} 数据为空，跳过该股票"
                warnings.warn(msg)
                logging.warning(msg)
                return None

            return data

        except Exception as e:
            msg = f"[SafeHighFreqHandler] 加载股票 {instrument} 数据时出错: {e}, 跳过该股票"
            warnings.warn(msg)
            logging.warning(msg)
            return None
