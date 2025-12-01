# ============================================
# 基于TQSDK的实时数据更新器
# 功能：使用tqsdk实时下载数据并转换为qlib格式
# ============================================

import os
import sys
import pandas as pd
import numpy as np
from tqsdk import TqApi, TqAuth
from datetime import datetime, timedelta
import time
import struct
from pathlib import Path
import shutil
from bisect import bisect_left


class TQSDKRealtimeUpdater:
    def __init__(self):
        # 天勤账号
        self.username = "xclight"
        self.password = "xclight666"
        
        # 路径配置 - 实时数据路径
        self.raw_dir = Path(r"E:\kq_raw_data_recent")  # 实时原始数据
        self.qlib_dir = Path(r"E:\qlib_data_recent")  # 实时qlib数据
        self.pool_csv = Path(r"E:\qlib\examples\highfreq\sorted_high_preclose_ratio_2025.csv")
        
        # 确保目录存在
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.qlib_dir.mkdir(parents=True, exist_ok=True)
        (self.qlib_dir / "features").mkdir(parents=True, exist_ok=True)
        (self.qlib_dir / "calendars").mkdir(parents=True, exist_ok=True)
        
        self.calendar_file = self.qlib_dir / "calendars" / "1min.txt"
        self.calendar = []
        self.calendar_index = {}
        self.calendar_dirty = False
        self._load_calendar()
        
        # 读取股票池
        try:
            stock_pool_df = pd.read_csv(self.pool_csv)
            # 转换股票代码格式
            self.stocks = []
            for code in stock_pool_df["code"].tolist():
                # 支持多种输入格式：sh.600000, SH.600000, SH600000
                code_upper = code.upper()
                
                # sh.600000 -> SSE.600000 (天勤标准格式)
                # sz.000001 -> SZSE.000001
                if code_upper.startswith("SH."):
                    self.stocks.append("SSE." + code[3:])  # 跳过 "SH."
                elif code_upper.startswith("SZ."):
                    self.stocks.append("SZSE." + code[3:])  # 跳过 "SZ."
                elif code_upper.startswith("SH") and len(code) > 2:
                    self.stocks.append("SSE." + code[2:])  # SH600000 -> SSE.600000
                elif code_upper.startswith("SZ") and len(code) > 2:
                    self.stocks.append("SZSE." + code[2:])  # SZ000001 -> SZSE.000001
                else:
                    # 保持原样
                    self.stocks.append(code)
            
            print(f"[OK] 实时数据更新器初始化完成")
            print(f"[DIR] 原始数据目录: {self.raw_dir}")
            print(f"[DIR] Qlib数据目录: {self.qlib_dir}")
            print(f"[INFO] 股票池: {len(self.stocks)} 支")
            
        except Exception as e:
            print(f"[ERROR] 初始化失败: {e}")
            raise
    
    def _disable_proxy_env(self):
        """禁用代理"""
        keys = [
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy",
        ]
        for k in keys:
            if k in os.environ:
                os.environ.pop(k, None)
        
        no_proxy_hosts = [
            ".shinnytech.com", ".shinnytech.com.cn",
            "shinnytech.com", "auth.shinnytech.com",
            "api.shinnytech.com", "account.shinnytech.com",
        ]
        existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
        merged = ",".join([h for h in (existing.split(",") if existing else []) if h])
        suffix = ("," + merged) if merged else ""
        os.environ["NO_PROXY"] = ",".join(no_proxy_hosts) + suffix
        os.environ["no_proxy"] = os.environ["NO_PROXY"]
    
    def _normalize_code_for_file(self, code):
        """将代码转换为文件名格式"""
        # SSE.600000 -> SSE_600000
        # SZSE.000001 -> SZSE_000001
        return code.replace(".", "_")
    
    def download_today_data(self, test_mode=False):
        """下载今天的数据"""
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"\n[DOWNLOAD] 开始下载今天数据: {today}")
        
        # 测试模式：生成模拟数据
        if test_mode:
            print("[TEST] 测试模式：生成模拟数据...")
            return self._generate_test_data()
        
        self._disable_proxy_env()
        
        try:
            api = TqApi(auth=TqAuth(self.username, self.password))
            print("[OK] TqApi连接成功")
        except Exception as e:
            print(f"[ERROR] 初始化 TqApi 失败: {e}")
            return False
        
        try:
            # 使用get_quote方式快速获取实时数据，不阻塞
            downloaded_count = 0
            failed_stocks = []
            temp_data_dir = self.raw_dir / "temp_today"
            temp_data_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"[INFO] 开始快速获取实时数据，共 {len(self.stocks)} 个股票...")
            
            # 策略：快速批量订阅，每个股票获取最近5条K线（5分钟数据）
            # 优化：增大批次大小，减少等待时间，提高并发度
            current_minute = datetime.now().replace(second=0, microsecond=0)
            print(f"[DEBUG] 当前时间: {current_minute}")
            
            print(f"[INFO] 开始快速批量订阅（共 {len(self.stocks)} 个股票）...")
            start_time = time.time()
            
            # 步骤1: 快速批量订阅K线（优化：减少等待时间，提高并发）
            print(f"[INFO] 步骤1: 批量订阅K线（data_length=5，快速模式）...")
            kline_dict = {}
            
            # 策略：快速订阅所有股票，然后一次性等待（避免分批等待的延迟）
            print(f"[INFO] 快速订阅所有股票（不等待响应，让服务器并行处理）...")
            subscribe_start = time.time()
            for stock in self.stocks:
                try:
                    fmt = stock
                    try:
                        # 快速订阅，不等待数据返回（让TQSDK在后台处理）
                        klines = api.get_kline_serial(fmt, duration_seconds=60, data_length=5)
                        if klines is not None:
                            kline_dict[stock] = {
                                'klines': klines,
                                'format': fmt
                            }
                    except:
                        pass
                except:
                    continue
            
            subscribe_elapsed = time.time() - subscribe_start
            print(f"[INFO] 订阅请求完成（耗时 {subscribe_elapsed:.1f}秒），等待数据更新...")
            
            # 一次性等待所有数据更新（减少等待次数）
            wait_start = time.time()
            try:
                # 等待数据更新，但设置合理的超时时间
                api.wait_update(deadline=time.time() + 3)  # 最多等待3秒（从5秒减少到3秒）
            except:
                pass
            wait_elapsed = time.time() - wait_start
            print(f"[INFO] 数据更新等待完成（耗时 {wait_elapsed:.1f}秒）")
            
            elapsed = time.time() - start_time
            success_rate = len(kline_dict) / len(self.stocks) * 100
            print(f"[OK] 订阅完成: {len(kline_dict)}/{len(self.stocks)} 成功 ({success_rate:.1f}%) (耗时 {elapsed:.1f}秒)")
            
            # 检查是否有遗漏的股票
            missing_stocks = set(self.stocks) - set(kline_dict.keys())
            if missing_stocks:
                print(f"[WARN] 未订阅成功的股票: {len(missing_stocks)} 个")
                if len(missing_stocks) <= 10:
                    print(f"  前10个: {list(missing_stocks)[:10]}")
            
            # 步骤2: 快速验证数据（不额外等待，直接验证）
            if len(kline_dict) > 0:
                print(f"[INFO] 步骤2: 验证数据完整性...")
                # 验证订阅的股票是否有数据（过滤掉空的）
                valid_kline_dict = {}
                for stock, info in kline_dict.items():
                    try:
                        if info['klines'] is not None and len(info['klines']) > 0:
                            valid_kline_dict[stock] = info
                    except:
                        continue
                kline_dict = valid_kline_dict
                print(f"[INFO] 有效订阅: {len(kline_dict)} 个股票有数据")
            else:
                print(f"[WARN] 没有成功订阅任何股票，可能有问题")
            
            # 步骤3: 快速收集所有数据
            print(f"[INFO] 步骤3: 收集数据...")
            
            for stock_idx, (stock, kline_info) in enumerate(kline_dict.items()):
                if stock_idx % 100 == 0 and stock_idx > 0:
                    print(f"  收集进度: {stock_idx}/{len(kline_dict)}...")
                
                try:
                    klines = kline_info['klines']
                    
                    if len(klines) > 0:
                        # 处理TQSDK返回的列名格式：可能是 'SSE.600105.open' 或 'open'
                        # 尝试多种可能的列名格式
                        def get_field(bar, field_name):
                            """获取字段值，支持多种列名格式"""
                            # 尝试标准列名
                            if field_name in bar.index:
                                return bar[field_name]
                            # 尝试带股票代码前缀的列名（如 SSE.600105.open）
                            stock_code = stock  # SSE.600000
                            for col in bar.index:
                                if col.endswith(f'.{field_name}') or col == field_name:
                                    return bar[col]
                            return None
                        
                        # 处理最近5条K线数据（5分钟）
                        data_list = []
                        for i in range(len(klines)):
                            bar = klines.iloc[i]
                            
                            # 获取datetime（可能是datetime对象或timestamp）
                            bar_datetime_raw = get_field(bar, 'datetime')
                            if bar_datetime_raw is None:
                                continue
                            
                            # 转换为Timestamp
                            if isinstance(bar_datetime_raw, (int, float)):
                                # 如果是纳秒时间戳（TQSDK格式）
                                bar_datetime = pd.Timestamp.fromtimestamp(bar_datetime_raw / 1e9)
                            else:
                                bar_datetime = pd.Timestamp(bar_datetime_raw)
                            
                            # 放宽时间条件：保存最近10分钟内的数据（确保连贯性）
                            time_diff = abs((bar_datetime - current_minute).total_seconds())
                            if time_diff <= 600:  # 允许10分钟误差，确保数据连贯
                                # 获取OHLCV数据
                                open_price = get_field(bar, 'open')
                                high_price = get_field(bar, 'high')
                                low_price = get_field(bar, 'low')
                                close_price = get_field(bar, 'close')
                                volume_value = get_field(bar, 'volume')
                                
                                if all(x is not None for x in [open_price, high_price, low_price, close_price, volume_value]):
                                    data_list.append({
                                        'datetime': bar_datetime,
                                        'open': float(open_price),
                                        'high': float(high_price),
                                        'low': float(low_price),
                                        'close': float(close_price),
                                        'volume': float(volume_value)
                                    })
                        
                        # 如果有数据，保存到CSV
                        if data_list:
                            # 追加或更新CSV（增量模式）
                            file_name = self._normalize_code_for_file(stock)
                            csv_file = temp_data_dir / f"{file_name}.csv"
                            
                            # 读取现有数据或创建新DataFrame
                            if csv_file.exists():
                                existing_df = pd.read_csv(csv_file)
                                existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
                                
                                # 合并新数据
                                new_df = pd.DataFrame(data_list)
                                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                                # 去重并排序
                                combined_df = combined_df.drop_duplicates(subset=['datetime'], keep='last')
                                combined_df = combined_df.sort_values('datetime')
                                combined_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                                downloaded_count += 1
                            else:
                                # 新建文件
                                df = pd.DataFrame(data_list)
                                df = df.drop_duplicates(subset=['datetime'], keep='last')
                                df = df.sort_values('datetime')
                                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                                downloaded_count += 1
                except Exception as e:
                    failed_stocks.append((stock, str(e)[:50]))
                    continue
            
            if failed_stocks and len(failed_stocks) <= 20:
                print(f"[WARN] 失败的股票 ({len(failed_stocks)} 个):")
                for stock, reason in failed_stocks[:10]:
                    print(f"  {stock}: {reason}")
            elif failed_stocks:
                print(f"[WARN] 失败的股票: {len(failed_stocks)} 个 (不显示详情)")
            
            total_elapsed = time.time() - start_time
            print(f"[OK] 下载完成: {downloaded_count} 支股票数据已更新 (总耗时: {total_elapsed:.1f}秒)")
            
            # 将临时数据移动到主目录
            self._merge_to_main_dir(temp_data_dir)
            
            return downloaded_count > 0
            
        except Exception as e:
            print(f"[ERROR] 下载过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            api.close()
            # 清理临时目录
            if temp_data_dir.exists():
                shutil.rmtree(temp_data_dir)
    
    def _generate_test_data(self):
        """生成测试数据（无需连接tqsdk）"""
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"[TEST] 生成模拟数据...")
        
        # 生成最近10分钟的数据
        base_time = datetime.now().replace(second=0, microsecond=0)
        
        temp_data_dir = self.raw_dir / "temp_today"
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        
        generated_count = 0
        
        for stock in self.stocks[:10]:  # 只生成前10个股票的测试数据
            try:
                file_name = self._normalize_code_for_file(stock)
                csv_file = temp_data_dir / f"{file_name}.csv"
                
                # 生成模拟数据
                data_list = []
                base_price = 10.0 + hash(stock) % 100  # 基于股票代码生成基础价格
                
                current_price = base_price
                
                # 生成最近10分钟的数据
                for i in range(10):
                    dt = base_time - timedelta(minutes=10-i)
                    
                    # 模拟价格波动
                    change = np.random.normal(0, 0.1)
                    current_price = max(1.0, current_price + change)
                    
                    high = current_price + abs(np.random.normal(0, 0.05))
                    low = current_price - abs(np.random.normal(0, 0.05))
                    volume = int(np.random.normal(1000000, 200000))
                    
                    data_list.append({
                        'datetime': dt,
                        'open': current_price,
                        'high': high,
                        'low': low,
                        'close': current_price,
                        'volume': volume
                    })
                
                # 保存为CSV
                df = pd.DataFrame(data_list)
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                
                print(f"[OK] 生成模拟数据: {file_name} ({len(data_list)} 条)")
                generated_count += 1
                
            except Exception as e:
                print(f"[WARN] 生成 {stock} 数据失败: {e}")
                continue
        
        print(f"[OK] 测试数据生成完成: {generated_count} 支股票")
        
        # 将临时数据移动到主目录
        self._merge_to_main_dir(temp_data_dir)
        
        # 清理临时目录
        if temp_data_dir.exists():
            shutil.rmtree(temp_data_dir)
        
        return generated_count > 0
    
    def _merge_to_main_dir(self, temp_data_dir):
        """将临时数据合并到主目录"""
        print(f"[MERGE] 合并数据到主目录: {self.qlib_dir / 'features'}")
        
        main_data_dir = self.qlib_dir / "features"
        main_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        total_files = len(list(temp_data_dir.glob("*.csv")))
        print(f"[MERGE] 临时目录文件数: {total_files}")
        
        success_count = 0
        error_count = 0
        
        for csv_file in temp_data_dir.glob("*.csv"):
            try:
                # 读取临时CSV
                temp_df = pd.read_csv(csv_file)
                if temp_df.empty:
                    continue
                
                # 确保有 datetime 列
                if 'datetime' not in temp_df.columns:
                    continue
                
                # 将 datetime 转换为 datetime 类型
                temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], errors='coerce')
                temp_df = temp_df.dropna(subset=['datetime'])
                
                if temp_df.empty:
                    continue
                
                # 获取股票代码
                code = csv_file.stem
                
                # 转换为Qlib格式（文件夹名）
                if code.startswith("SSE_"):
                    folder_name = "SH" + code[4:]
                elif code.startswith("SZSE_"):
                    folder_name = "SZ" + code[5:]
                else:
                    folder_name = code
                
                # 创建股票目录
                stock_dir = main_data_dir / folder_name
                stock_dir.mkdir(parents=True, exist_ok=True)
                
                # 增量合并到目标文件
                existing_csv = stock_dir / "data.csv"
                
                new_rows_for_bin = None
                if existing_csv.exists():
                    # 读取现有数据
                    try:
                        existing_df = pd.read_csv(existing_csv)
                        existing_df['datetime'] = pd.to_datetime(existing_df['datetime'], errors='coerce')
                        existing_df = existing_df.dropna(subset=['datetime'])
                        
                        # 只添加新的数据（避免重复）
                        existing_datetimes = set(existing_df['datetime'])
                        new_data = temp_df[~temp_df['datetime'].isin(existing_datetimes)]
                        
                        if not new_data.empty:
                            # 合并数据
                            combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                            combined_df = combined_df.sort_values('datetime').drop_duplicates(subset=['datetime'], keep='last')
                            combined_df.to_csv(existing_csv, index=False, encoding='utf-8-sig')
                            new_rows_for_bin = new_data.sort_values('datetime')
                        else:
                            # 没有新数据，跳过
                            continue
                    except Exception as e:
                        # 如果读取失败，直接覆盖
                        temp_df.to_csv(existing_csv, index=False, encoding='utf-8-sig')
                        new_rows_for_bin = temp_df.sort_values('datetime')
                else:
                    # 文件不存在，直接保存
                    temp_df.to_csv(existing_csv, index=False, encoding='utf-8-sig')
                    new_rows_for_bin = temp_df.sort_values('datetime')
                
                # 强制刷新文件系统
                import os
                os.utime(existing_csv, None)
                
                if new_rows_for_bin is not None and not new_rows_for_bin.empty:
                    self._ensure_lowercase_symlink(stock_dir)
                    self._update_bin_files(folder_name, new_rows_for_bin)
                
                success_count += 1
                if total_files <= 50:  # 只在文件数少时显示详细信息
                    print(f"[OK] 保存 {folder_name}: {len(temp_df)} 条数据")
                
            except Exception as e:
                error_count += 1
                print(f"[WARN] 保存 {csv_file.name} 失败: {e}")
        
        # 打印统计信息
        print(f"\n[MERGE] 合并统计: 成功 {success_count}, 失败 {error_count}, 总计 {total_files}")
        self._save_calendar_if_needed()
    
    def _load_calendar(self):
        if self.calendar_file.exists():
            with open(self.calendar_file, "r", encoding="utf-8") as f:
                self.calendar = [line.strip() for line in f if line.strip()]
        else:
            self.calendar = []
        self.calendar_index = {ts: idx for idx, ts in enumerate(self.calendar)}
        self.calendar_dirty = False
    
    def _save_calendar_if_needed(self):
        if not self.calendar_dirty:
            return
        self.calendar_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.calendar_file, "w", encoding="utf-8") as f:
            for ts in self.calendar:
                f.write(f"{ts}\n")
        self.calendar_dirty = False
    
    def _rebuild_calendar_index(self):
        self.calendar_index = {ts: idx for idx, ts in enumerate(self.calendar)}
    
    def _get_or_create_calendar_index(self, dt_value):
        ts = pd.Timestamp(dt_value).strftime("%Y-%m-%d %H:%M:%S")
        existing = self.calendar_index.get(ts)
        if existing is not None:
            return existing
        if not self.calendar or ts > self.calendar[-1]:
            idx = len(self.calendar)
            self.calendar.append(ts)
            self.calendar_index[ts] = idx
        else:
            insert_pos = bisect_left(self.calendar, ts)
            self.calendar.insert(insert_pos, ts)
            self._rebuild_calendar_index()
            idx = self.calendar_index[ts]
        self.calendar_dirty = True
        return idx
    
    def _ensure_lowercase_symlink(self, stock_dir: Path):
        try:
            lowercase_path = stock_dir.parent / stock_dir.name.lower()
            if lowercase_path.exists() or lowercase_path.is_symlink():
                return
            lowercase_path.symlink_to(stock_dir.name)
        except Exception:
            pass
    
    def _update_bin_files(self, folder_name: str, new_rows: pd.DataFrame):
        if new_rows is None or new_rows.empty:
            return
        stock_dir = self.qlib_dir / "features" / folder_name
        stock_dir.mkdir(parents=True, exist_ok=True)
        df = new_rows.copy()
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        if df.empty:
            return
        df = df.drop_duplicates(subset=['datetime'], keep='last').sort_values('datetime')
        minute_indices = [self._get_or_create_calendar_index(ts) for ts in df['datetime'].tolist()]
        values_map = {
            'open': df['open'].astype(np.float32).to_numpy(),
            'high': df['high'].astype(np.float32).to_numpy(),
            'low': df['low'].astype(np.float32).to_numpy(),
            'close': df['close'].astype(np.float32).to_numpy(),
            'volume': df['volume'].astype(np.float32).to_numpy(),
        }
        for field, vals in values_map.items():
            bin_path = stock_dir / f"{field}.1min.bin"
            self._write_bin_series(bin_path, minute_indices, vals)
    
    def _write_bin_series(self, bin_path: Path, minute_indices, values):
        if not minute_indices:
            return
        dedup = {}
        for idx, val in zip(minute_indices, values):
            dedup[int(idx)] = float(val)
        sorted_indices = np.array(sorted(dedup.keys()), dtype=np.int64)
        sorted_values = np.array([dedup[i] for i in sorted_indices], dtype=np.float32)
        
        if bin_path.exists():
            existing = np.fromfile(bin_path, dtype=np.float32)
            start_idx = int(existing[0])
            data = existing[1:]
        else:
            start_idx = int(sorted_indices[0])
            data = np.array([], dtype=np.float32)
        
        if start_idx > sorted_indices[0]:
            prepend_len = start_idx - sorted_indices[0]
            data = np.concatenate([np.full(prepend_len, np.nan, dtype=np.float32), data])
            start_idx = int(sorted_indices[0])
        
        target_len = sorted_indices[-1] - start_idx + 1
        if len(data) < target_len:
            data = np.concatenate([data, np.full(target_len - len(data), np.nan, dtype=np.float32)])
        
        for idx, val in zip(sorted_indices, sorted_values):
            pos = idx - start_idx
            if pos < 0:
                continue
            data[pos] = val
        
        out = np.concatenate([[float(start_idx)], data.astype(np.float32)])
        out.astype(np.float32).tofile(bin_path)
    
    def update_data(self):
        """执行数据更新"""
        return self.download_today_data()
    
    def start_scheduler(self):
        """启动定时任务"""
        import schedule
        
        print("\n" + "="*70)
        print("[SCHEDULER] TQSDK实时数据更新定时任务")
        print("="*70)
        print("[INFO] 更新频率: 交易时间内每3分钟更新一次")
        print("[INFO] 交易时间: 09:30-11:30, 13:00-15:00")
        print("="*70 + "\n")
        
        # 设置定时任务 - 每3分钟更新一次（只在交易时间）
        # 上午：9:30, 9:33, 9:36, ..., 11:27, 11:30
        # 下午：13:00, 13:03, 13:06, ..., 14:57
        
        # 上午时段：9:30-11:30，每3分钟
        for hour in [9, 10, 11]:
            if hour == 9:
                # 9:30, 9:33, 9:36, ..., 9:57
                for minute in range(30, 60, 3):
                    time_str = f"{hour:02d}:{minute:02d}"
                    schedule.every().day.at(time_str).do(self.update_data)
            elif hour == 10:
                # 10:00, 10:03, 10:06, ..., 10:57
                for minute in range(0, 60, 3):
                    time_str = f"{hour:02d}:{minute:02d}"
                    schedule.every().day.at(time_str).do(self.update_data)
            elif hour == 11:
                # 11:00, 11:03, 11:06, ..., 11:30
                for minute in range(0, 31, 3):
                    time_str = f"{hour:02d}:{minute:02d}"
                    schedule.every().day.at(time_str).do(self.update_data)
        
        # 下午时段：13:00-15:00，每3分钟
        for hour in [13, 14]:
            for minute in range(0, 60, 3):
                time_str = f"{hour:02d}:{minute:02d}"
                schedule.every().day.at(time_str).do(self.update_data)
        
        # 统计设置的任务数
        job_count = len(schedule.jobs)
        print(f"[INFO] 已设置 {job_count} 个定时任务（每3分钟执行一次）")
        
        # 检查当前时间并立即执行（使用北京时间）
        current_time = datetime.now()
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        # 判断是否在交易时间内
        is_trading_time = False
        if 9 <= current_hour < 12:  # 上午
            if current_hour == 9 and current_minute >= 30:
                is_trading_time = True
            elif current_hour == 11 and current_minute <= 30:
                is_trading_time = True
            elif 10 <= current_hour < 11:
                is_trading_time = True
        elif 13 <= current_hour < 15:  # 下午
            is_trading_time = True
        
        if is_trading_time:
            print("[TIME] 检测到交易时间，立即执行一次更新...")
            self.update_data()
        else:
            print(f"[TIME] 当前时间 {current_hour:02d}:{current_minute:02d} 不在交易时间内，等待交易时间...")
        
        print("[START] 定时任务已启动，按 Ctrl+C 停止")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(10)  # 每10秒检查一次（更频繁，确保不遗漏）
        except KeyboardInterrupt:
            print("\n[STOP] 实时数据更新器已停止")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TQSDK实时数据更新器')
    parser.add_argument('--test', action='store_true', help='测试模式：生成模拟数据')
    parser.add_argument('--once', action='store_true', help='只执行一次更新（不进入定时循环）')
    args = parser.parse_args()
    
    updater = TQSDKRealtimeUpdater()
    
    if args.test:
        print("\n[TEST] 测试模式：生成模拟数据并测试合并流程...")
        updater.download_today_data(test_mode=True)
        print("\n[OK] 测试完成！可以查看数据是否正确合并")
    elif args.once:
        print("\n[UPDATE] 执行一次数据更新...")
        updater.download_today_data()
        print("\n[OK] 更新完成")
    else:
        updater.start_scheduler()

if __name__ == "__main__":
    main()
