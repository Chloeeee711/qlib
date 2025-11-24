from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc, Alpha158 as Alpha158Handler
import highfreq_ops
from qlib.data.ops import Operators
try:
    # Ensure custom ops are registered in any worker importing this module
    # 注意：If 会覆盖 Qlib 默认的 If 操作符，修复索引不一致问题
    from highfreq_ops import DayLast, FFillNan, Date, Select, IsNull, Cut, DayFirst, IntradayWindowVWAP, DayShift, If, Gt, Lt
    Operators.register([DayLast, FFillNan, Date, Select, IsNull, Cut, DayFirst, IntradayWindowVWAP, DayShift, If, Gt, Lt])
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

    def _assign_label_column(self, df, values):
        """将 LABEL0 列安全写回，避免频繁插入导致的碎片化"""
        if df is None or len(df) == 0:
            return
        arr = np.asarray(values)
        if isinstance(df.columns, pd.MultiIndex):
            col_key = ('label', 'LABEL0')
        else:
            col_key = 'LABEL0'
        if col_key in df.columns:
            try:
                df.drop(columns=[col_key], inplace=True)
            except Exception:
                pass
        try:
            df._consolidate_inplace()
        except Exception:
            pass
        df[col_key] = arr

    def get_feature_config(self):
        fields = []
        names = []

        template_if = "If(IsNull({1}), {0}, {1})"
        template_paused = "Select(Or(IsNull($paused), Eq($paused, 0.0)), {0})"
        template_fillnan = "FFillNan({0})"
        # Because there is no vwap field in the yahoo data, a method similar to Simpson integration is used to approximate vwap
        simpson_vwap = "($open + 2*$high + 2*$low + $close)/6"

        def get_normalized_price_feature(price_field, shift=0):
            """Get normalized price feature ops"""
            if shift == 0:
                template_norm = "Cut({0}/Ref(Ref(DayLast({1}), 240), -1), 240, None)"
            else:
                template_norm = "Cut(Ref({0}, " + str(shift) + ")/Ref(Ref(DayLast({1}), 240), -1), 240, None)"

            # 彻底修复索引不一致：使用统一的索引基准
            # 方案：所有字段都先基于 $close 的 Select 结果进行对齐
            # 通过使用 $close 作为 Select 的条件基准，确保所有字段都有相同的索引集合
            close_select_base = template_paused.format("$close")
            close_field = template_fillnan.format(close_select_base)
            
            # 关键修复：对于其他字段，先 Select，然后对齐到 $close 的索引
            # 方法：使用相同的 Select 条件，确保索引一致
            if price_field == simpson_vwap or "(open" in str(price_field).lower() or "+" in str(price_field):
                # 表达式：先处理组成字段，都基于相同的 Select 条件
                open_base = template_fillnan.format(template_paused.format("$open"))
                high_base = template_fillnan.format(template_paused.format("$high"))
                low_base = template_fillnan.format(template_paused.format("$low"))
                close_base = template_fillnan.format(template_paused.format("$close"))
                price_field_processed = "({0} + {1} * 2 + {2} * 2 + {3}) / 6".format(
                    open_base, high_base, low_base, close_base
                )
            else:
                # 普通字段：使用相同的 Select 模式和 FFillNan
                price_field_processed = template_fillnan.format(template_paused.format(price_field))
            
            # 如果 price_field 就是 $close，直接使用，不需要 If
            if price_field == "$close":
                selected_field = close_field
            else:
                # 使用 If 选择，但确保两个分支基于相同的处理流程
                # 注意：即使都经过 Select + FFillNan，如果原始字段的索引不同，结果索引仍可能不同
                # 所以我们需要依赖 Qlib 在 If 内部的索引对齐机制（通过 reindex）
                selected_field = template_if.format(close_field, price_field_processed)
            
            feature_ops = template_norm.format(selected_field, close_field)
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

        # 修复索引不一致：volume 字段的所有组成部分都要经过相同的处理
        # simpson_vwap 需要特殊处理：先对组成字段进行 Select + FFillNan，然后计算表达式
        open_base = template_fillnan.format(template_paused.format("$open"))
        high_base = template_fillnan.format(template_paused.format("$high"))
        low_base = template_fillnan.format(template_paused.format("$low"))
        close_base = template_fillnan.format(template_paused.format("$close"))
        # 基于已处理的字段计算 vwap（确保索引一致）
        vwap_field_processed = "({0} + {1} * 2 + {2} * 2 + {3}) / 6".format(
            open_base, high_base, low_base, close_base
        )
        
        volume_field = template_fillnan.format(template_paused.format("$volume"))
        low_field = template_fillnan.format(template_paused.format("$low"))
        high_field = template_fillnan.format(template_paused.format("$high"))
        
        fields += [
            "Cut({0}/Ref(DayLast(Mean({0}, 7200)), 240), 240, None)".format(
                "If(IsNull({0}), 0, If(Or(Gt({1}, Mul(1.001, {3})), Lt({1}, Mul(0.999, {2}))), 0, {0}))".format(
                    volume_field, vwap_field_processed, low_field, high_field
                )
            )
        ]
        names += ["$volume"]
        fields += [
            "Cut(Ref({0}, 240)/Ref(DayLast(Mean({0}, 7200)), 240), 240, None)".format(
                "If(IsNull({0}), 0, If(Or(Gt({1}, Mul(1.001, {3})), Lt({1}, Mul(0.999, {2}))), 0, {0}))".format(
                    volume_field, vwap_field_processed, low_field, high_field
                )
            )
        ]
        names += ["$volume_1"]

        # 附加原始辅助字段，便于作为特征参与训练（保持与现有模板一致的填充与暂停过滤）
        # 移除 $change 特征，因为它包含当日价格变化信息，会导致数据泄露
        fields += [
            template_fillnan.format(template_paused.format("$factor")),
            template_fillnan.format(template_paused.format("$paused")),
            # template_fillnan.format(template_paused.format("$change")),  # 移除，防止数据泄露
            template_fillnan.format(template_paused.format("$paused_num")),
        ]
        names += ["$factor", "$paused", "$paused_num"]  # 移除 "$change"
        
        # ========== 添加官方 Alpha158 因子 ==========
        print("[Alpha158] 开始加载官方 Alpha158 特征配置")
        alpha158_fields, alpha158_names = Alpha158Handler.get_feature_config(self)
        print(f"[Alpha158] 官方因子数量: {len(alpha158_fields)}")

        fields += alpha158_fields
        names += alpha158_names
        print(f"[Alpha158] 总特征数量: {len(fields)}")
       
        return {
            "feature": (fields, names),
            "label": (["Ref($close, -1) / $close - 1"], ["LABEL0"])  # 占位符，会被自定义标签覆盖
        }
    
    def _load_custom_labels(self, instruments, start_time, end_time):
        """计算自定义标签"""
        if self._custom_labels is not None:
            return self._custom_labels
            
        # print("计算自定义隔夜收益率标签...")  # 减少输出
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
            
        # print(f"自定义标签计算完成，形状: {self._custom_labels.shape}")  # 减少输出
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
          
        # 拉分钟数据 - 只加载指定时间范围的数据，避免数据泄露
        try:
            df = D.features(
                [inst],
                ["$open", "$high", "$low", "$close", "$volume"],
                start_time=start_time,
                end_time=end_time,
                freq="1min",
            )
            
            # 单独加载次日数据用于标签计算
            from datetime import datetime, timedelta
            def _parse_dt(s):
                if s is None:
                    return None
                if isinstance(s, (datetime,)):
                    return s
                s = str(s)
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                    try:
                        return datetime.strptime(s, fmt)
                    except Exception:
                        continue
                try:
                    return datetime.strptime(s.split(" ")[0], "%Y-%m-%d")
                except Exception:
                    return None

            e_dt = _parse_dt(end_time)
            if e_dt is not None:
                # 需要加载次日数据用于标签计算（计算当日标签需要次日上午VWAP）
                # 加载次日全天数据，确保包含次日上午时间段（10:11-10:20）
                next_day_date = e_dt + timedelta(days=1)
                extended_start = next_day_date.strftime("%Y-%m-%d") + " 09:30:00"
                extended_end = next_day_date.strftime("%Y-%m-%d") + " 15:00:00"
                
                # 只在交易日尝试加载（简化：先尝试，如果失败就用当日数据计算）
                try:
                    extended_df = D.features(
                        [inst],
                        ["$open", "$high", "$low", "$close", "$volume"],
                        start_time=extended_start,
                        end_time=extended_end,
                        freq="1min",
                    )
                    
                    # 合并数据用于标签计算
                    if extended_df is not None and not extended_df.empty:
                        # 确保没有重复索引
                        df = pd.concat([df, extended_df], axis=0)
                        df = df[~df.index.duplicated(keep='first')]  # 去除重复索引
                    # 如果次日数据为空，不打印警告（可能是交易日边界或数据未更新，这是正常的）
                except Exception:
                    # 静默处理，如果次日数据不存在，标签计算会使用已有的当日数据
                    # 这种情况下，标签会在最后一天缺失，这是正常的
                    pass
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
        # 严格使用半开区间，避免窗口上界包含未来一分钟
        # 下午窗口 [14:41, 14:50) - 当日下午
        mask_afternoon = (t_int >= 1441) & (t_int < 1450)
        # 上午窗口 [10:11, 10:20) - 当日上午（用于计算次日上午VWAP）
        mask_morning = (t_int >= 1011) & (t_int < 1020)

        # 典型价格与成交量
        tp  = (df["$open"] + df["$high"] + df["$low"] + df["$close"]) / 4.0
        vol = df["$volume"]
        
        # 按"日期"聚合成日级 VWAP
        day_key = pd.Index(dt.date)
        def window_vwap(mask):
            num = (tp.where(mask, 0.0) * vol.where(mask, 0.0)).groupby(day_key).sum()
            den = vol.where(mask, 0.0).groupby(day_key).sum()
            # 严格：若该日窗口无数据，则返回 NaN，不再使用同日价格或均值回填，避免泄露
            wv  = (num / den.replace(0.0, np.nan)).astype("float64")
            wv = wv.ffill()  # 前向填充，使用新的API
            return wv

        # 定义"今天标签 = 明天上午 / 今天下午 - 1"
        #afternoon_today = window_vwap(mask_today)          # 键为 D（当日）
        afternoon_vwap = window_vwap(mask_afternoon)
        #morning_today   = window_vwap(mask_next)           # 键为 D（当日）
        morning_vwap   = window_vwap(mask_morning)  
    
        # 正确的隔夜收益率计算
        # 当日标签 = 次日上午VWAP / 当日下午VWAP - 1
        # 需要将次日上午VWAP向前对齐到当日
        morning_next_day = morning_vwap.shift(-1)  # 次日上午VWAP对齐到当日
        label_vwap = morning_next_day / afternoon_vwap - 1  # 正确的隔夜收益率
        # 截断异常值 (保留99.5%的数据)
        label_vwap = label_vwap.clip(lower=label_vwap.quantile(0.01), 
                                 upper=label_vwap.quantile(0.99))
    
        
        # 计算隔夜收益：当日 -> 次日
        #overnight_return = morning_tomorrow / afternoon_today - 1  # 索引为当日日期
        # 将当日值顺延到“次日”的日期索引
        #overnight_return_nextday = overnight_return.shift(1)

        # 广播回分钟：次日的每一分钟都使用前一日的隔夜收益
        #days = pd.Index(pd.to_datetime(df.index).date)
        #ov_map = overnight_return_nextday.to_dict()  # 键：次日日期
        #broadcast = pd.Series([ov_map.get(d, np.nan) for d in days], index=df.index, dtype="float64")

        days = pd.Index(dt.date)
        label_map = label_vwap.to_dict()  # 键：当日日期（已对齐）
        broadcast = pd.Series([label_map.get(d, np.nan) for d in days], index=df.index, dtype="float64")

        # 创建MultiIndex并返回
        mi = pd.MultiIndex.from_arrays([[inst]*len(broadcast), df.index], names=["instrument","datetime"])
        return pd.DataFrame({"LABEL0": broadcast.values}, index=mi)

    def setup_data(self, *args, **kwargs):
        """重写setup_data，在数据加载时计算自定义标签"""
        # print(f"setup_data 开始，_data 存在: {hasattr(self, '_data')}")  # 减少输出
        if hasattr(self, '_data'):
            print(f"_data 形状: {self._data.shape if self._data is not None else 'None'}")
        
        # 先调用父类方法加载特征数据
        super().setup_data(*args, **kwargs)
        
        # print(f"父类setup_data后，_data 存在: {hasattr(self, '_data')}")  # 减少输出
        if hasattr(self, '_data'):
            print(f"_data 形状: {self._data.shape if self._data is not None else 'None'}")
        else:
            print("父类setup_data后，_data属性不存在！")
            # 手动设置_data属性
            print("尝试手动设置_data属性...")
            try:
                self._data = self.data_loader.load(self.instruments, self.start_time, self.end_time)
            # print(f"手动设置_data成功，形状: {self._data.shape if self._data is not None else 'None'}")  # 减少输出
            except Exception as e:
                print(f"手动设置_data失败: {e}")
        
        # 不再扩展整个数据集，避免数据泄露
        # 标签计算时会在_compute_label_single中单独加载次日数据
        
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
            self._assign_label_column(self._data, broadcast.values)

            # 同步到学习/推理数据（不同 Qlib 版本命名可能不同，尽量覆盖常见属性）
            for buf_name in ['_learn', '_infer', '_data_l', '_data_i']:
                if hasattr(self, buf_name) and getattr(self, buf_name) is not None:
                    try:
                        buf = getattr(self, buf_name)
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
                        self._assign_label_column(buf, aligned_buf.values)
                    except Exception:
                        continue

            nn = pd.Series(broadcast).notna().sum()
            # print(f"自定义标签覆盖写入完成，按日常数化，非NaN数量: {nn}")  # 减少输出
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
        template_fillnan = "FFillNan({0})"
        # Because there is no vwap field in the yahoo data, a method similar to Simpson integration is used to approximate vwap
        simpson_vwap = "($open + 2*$high + 2*$low + $close)/6"
        fields += [
            "Cut({0}, 240, None)".format(template_fillnan.format(template_paused.format("$close"))),
        ]
        names += ["$close0"]
        # 修复索引不一致：确保 If 的两个分支都经过相同处理
        # simpson_vwap 需要特殊处理
        close_field = template_fillnan.format(template_paused.format("$close"))
        open_base = template_fillnan.format(template_paused.format("$open"))
        high_base = template_fillnan.format(template_paused.format("$high"))
        low_base = template_fillnan.format(template_paused.format("$low"))
        vwap_field_processed = "({0} + {1} * 2 + {2} * 2 + {3}) / 6".format(
            open_base, high_base, low_base, close_field
        )
        fields += [
            "Cut({0}, 240, None)".format(
                template_if.format(close_field, vwap_field_processed)
            )
        ]
        names += ["$vwap0"]
        return fields, names



import warnings
import os
import numpy as np
from qlib.data.dataset.handler import DataHandler
# 注释掉这行，避免覆盖本文件定义的 HighFreqHandler 类
# from highfreq_handler import HighFreqHandler  # 确保引入你原来的 HighFreqHandler
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
