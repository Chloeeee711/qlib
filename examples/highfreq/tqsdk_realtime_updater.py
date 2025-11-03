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



class TQSDKRealtimeUpdater:
    def __init__(self):
        # 天勤账号
        self.username = "xclight"
        self.password = "xclight666"
        
        # 路径配置 - 实时数据路径
        self.raw_dir = Path(r"C:\Users\ASUS\kq_raw_data_recent")  # 实时原始数据
        self.qlib_dir = Path(r"C:\Users\ASUS\qlib_data_recent")  # 实时qlib数据
        self.pool_csv = Path(r"C:\Users\ASUS\qlib\examples\highfreq\sorted_high_preclose_ratio_2025.csv")
        
        # 确保目录存在
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.qlib_dir.mkdir(parents=True, exist_ok=True)
        
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
            # 使用较小的批次避免超时，但增加并发度
            current_minute = datetime.now().replace(second=0, microsecond=0)
            print(f"[DEBUG] 当前时间: {current_minute}")
            
            print(f"[INFO] 开始快速批量订阅（共 {len(self.stocks)} 个股票）...")
            start_time = time.time()
            
            # 步骤1: 快速批量订阅K线（关键：使用data_length=1只获取最近1条）
            print(f"[INFO] 步骤1: 批量订阅K线（data_length=1，速度最快）...")
            kline_dict = {}
            batch_size = 50  # 减小批次大小，避免超时
            
            for batch_idx in range(0, len(self.stocks), batch_size):
                batch = self.stocks[batch_idx:batch_idx+batch_size]
                batch_num = batch_idx // batch_size + 1
                total_batches = (len(self.stocks) + batch_size - 1) // batch_size
                
                if batch_num % 2 == 1 or batch_num == total_batches:  # 打印部分批次
                    print(f"  订阅进度: 批次 {batch_num}/{total_batches} ({len(batch)} 个股票)...")
                
                # 记录批次开始时间
                batch_start_time = time.time()
                
                # 批量订阅这一批（快速订阅，不阻塞）
                # 策略：先快速订阅所有股票（不等待响应），然后再统一等待
                for stock in batch:
                    try:
                        # 优先使用SSE.600000格式（天勤标准格式）
                        fmt = stock
                        
                        # 关键：获取最近5条K线（5分钟数据），确保连贯性
                        try:
                            klines = api.get_kline_serial(fmt, duration_seconds=60, data_length=5)
                            if klines is not None:
                                # 先保存引用，不检查数据长度（减少阻塞）
                                kline_dict[stock] = {
                                    'klines': klines,
                                    'format': fmt
                                }
                        except:
                            # 如果标准格式失败，不尝试其他格式（节省时间）
                            pass
                    except:
                        continue
                
                # 批次之间快速等待一次更新（让服务器有时间处理，但不阻塞太久）
                if batch_idx + batch_size < len(self.stocks):
                    try:
                        api.wait_update(deadline=time.time() + 1)  # 最多等待1秒
                    except:
                        pass
                    time.sleep(0.1)  # 短暂等待，减少延迟
            
            elapsed = time.time() - start_time
            success_rate = len(kline_dict) / len(self.stocks) * 100
            print(f"[OK] 订阅完成: {len(kline_dict)}/{len(self.stocks)} 成功 ({success_rate:.1f}%) (耗时 {elapsed:.1f}秒)")
            
            # 检查是否有遗漏的股票
            missing_stocks = set(self.stocks) - set(kline_dict.keys())
            if missing_stocks:
                print(f"[WARN] 未订阅成功的股票: {len(missing_stocks)} 个")
                if len(missing_stocks) <= 10:
                    print(f"  前10个: {list(missing_stocks)[:10]}")
            
            # 步骤2: 等待一次更新确保数据完整（快速）
            if len(kline_dict) > 0:
                print(f"[INFO] 步骤2: 等待数据更新（最多3秒）...")
                try:
                    api.wait_update(deadline=time.time() + 3)  # 最多等待3秒
                except:
                    pass
                time.sleep(0.2)  # 额外等待0.2秒
                
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
                        else:
                            # 没有新数据，跳过
                            continue
                    except Exception as e:
                        # 如果读取失败，直接覆盖
                        temp_df.to_csv(existing_csv, index=False, encoding='utf-8-sig')
                else:
                    # 文件不存在，直接保存
                    temp_df.to_csv(existing_csv, index=False, encoding='utf-8-sig')
                
                # 强制刷新文件系统
                import os
                os.utime(existing_csv, None)
                
                success_count += 1
                if total_files <= 50:  # 只在文件数少时显示详细信息
                    print(f"[OK] 保存 {folder_name}: {len(temp_df)} 条数据")
                
            except Exception as e:
                error_count += 1
                print(f"[WARN] 保存 {csv_file.name} 失败: {e}")
        
        # 打印统计信息
        print(f"\n[MERGE] 合并统计: 成功 {success_count}, 失败 {error_count}, 总计 {total_files}")
    
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
        
        # 设置定时任务 - 每3分钟更新一次
        for hour in range(9, 15):
            if hour == 9:
                # 9:30-9:59
                for minute in range(30, 60):
                    time_str = f"{hour:02d}:{minute:02d}"
                    schedule.every().day.at(time_str).do(self.update_data)
            elif hour == 11:
                # 11:00-11:30
                for minute in range(0, 31):
                    time_str = f"{hour:02d}:{minute:02d}"
                    schedule.every().day.at(time_str).do(self.update_data)
            elif hour == 13:
                # 13:00-13:59
                for minute in range(0, 60):
                    time_str = f"{hour:02d}:{minute:02d}"
                    schedule.every().day.at(time_str).do(self.update_data)
            elif hour == 14:
                # 14:00-14:59
                for minute in range(0, 60):
                    time_str = f"{hour:02d}:{minute:02d}"
                    schedule.every().day.at(time_str).do(self.update_data)
        
        # 检查当前时间并立即执行
        current_hour = datetime.now().hour
        current_minute = datetime.now().minute
        
        if 9 <= current_hour < 15:
            print("[TIME] 检测到交易时间，立即执行一次更新...")
            self.update_data()
        
        print("[START] 定时任务已启动，按 Ctrl+C 停止")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)  # 每30秒检查一次
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
