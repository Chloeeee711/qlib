import sys
from pathlib import Path
from contextlib import closing
from typing import List
import pandas as pd
from loguru import logger
import fire

# --- Qlib 转换依赖 ---
CUR_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = CUR_DIR.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dump_bin import DumpDataUpdate  # 用于 csv → qlib 二进制转换


def normalize_stock_code(code: str) -> str:
    """标准化股票代码"""
    c = str(code).strip().upper().replace("\t", "").replace(" ", "")
    if c.startswith("SH"):
        return f"SSE.{c[-6:]}" if len(c) > 2 else c
    if c.startswith("SZ"):
        return f"SZSE.{c[-6:]}" if len(c) > 2 else c
    if c.endswith(".SH"):
        return f"SSE.{c[:-3]}"
    if c.endswith(".SZ"):
        return f"SZSE.{c[:-3]}"
    return c


class KQDownloaderFixed:
    """TQSDK 数据下载器（兼容 Qlib 官方格式）"""

    def run(
        self,
        source_dir: str,
        target_dir: str,
        csv_stock_pool: str,
        start: str,
        end: str,
        interval: str = "1min",
        username: str = None,
        password: str = None,
        limit_nums: int = None,
        benchmark: str = None,           # ⚡ 新增
        benchmark_dir: str = None,       # ⚡ 新增
        year_segment: int = None,         # ⚡ 新增：指定下载哪一年（如2020），None表示下载全部
        append_mode: bool = True,         # ⚡ 新增：是否追加模式（True=追加，False=覆盖）
        max_workers: int = None           # ⚡ 新增：并发数（None=自动，Windows 建议 1-2）
    ):
        try:
            from tqsdk import TqApi, TqAuth
            from tqsdk.tools import DataDownloader
        except Exception:
            logger.error("请先安装 TQSDK 专业版: pip install tqsdk")
            raise

        save_dir = Path(source_dir)
        qlib_dir = Path(target_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        qlib_dir.mkdir(parents=True, exist_ok=True)

        def normalize_stock_code(code: str) -> str:
            """
            格式化股票代码为 TQSDK 要求的格式：
            SSE.600000 或 SZSE.000001
            支持各种输入形式：
              - 600000
              - SH600000 / SZ000001
              - sh.600000 / sz.000001
              - sh_600000 / sz_000001
              - 600000.SH / 000001.SZ
            """
            c = str(code).strip().replace("\t", "").replace(" ", "").upper()

            # 🔹 统一下划线为点号，方便后续统一处理
            c = c.replace("_", ".")

            # 处理形如 SH.600000 或 SZ.000001
            if "." in c:
                prefix, num = c.split(".", 1)
                num = num.zfill(6)
                if prefix in {"SH", "SSE"}:
                    return f"SSE.{num}"
                elif prefix in {"SZ", "SZSE"}:
                    return f"SZSE.{num}"
                elif prefix in {"SHSE"}:
                    return f"SSE.{num}"
                elif prefix in {"SZSE"}:
                    return f"SZSE.{num}"

            # 处理形如 600000.SH 或 000001.SZ
            if c.endswith(".SH") or c.endswith(".SZ"):
                num = c[:-3].zfill(6)
                if c.endswith(".SH"):
                    return f"SSE.{num}"
                else:
                    return f"SZSE.{num}"

            # 处理形如 SH600000 / SZ000001
            if c.startswith("SH") and c[2:].isdigit():
                return f"SSE.{c[2:].zfill(6)}"
            if c.startswith("SZ") and c[2:].isdigit():
                return f"SZSE.{c[2:].zfill(6)}"

            # 纯数字，默认上交所
            if c.isdigit() and len(c) <= 6:
                return f"SSE.{c.zfill(6)}"

            # 无法识别，直接返回原始
            return c



   

        codes_df = pd.read_csv(csv_stock_pool)
        codes = codes_df["code"].drop_duplicates().astype(str).tolist()
        codes = [normalize_stock_code(c) for c in codes]


        # limit_nums 控制逻辑
        if limit_nums is not None and limit_nums > 0:
            logger.info(f"⚙️ 限制股票数量为 {limit_nums}")
            codes = codes[: int(limit_nums)]
        else:
            logger.info("⚙️ 未设置 limit_nums，默认下载全部股票")

        logger.info(f"📊 股票数: {len(codes)}")

                # --- 自动加入 benchmark ---
        if benchmark:
            benchmark_code = benchmark
            if benchmark_code not in codes:
                codes.append(benchmark_code)
                logger.info(f"✅ 自动加入 benchmark: {benchmark_code}")

            # benchmark_dir 存在则创建空文件夹
            if benchmark_dir:
                benchmark_dir_path = Path(benchmark_dir)
                benchmark_dir_path.mkdir(parents=True, exist_ok=True)
                benchmark_file = benchmark_dir_path / f"{benchmark_code}.csv"
                if not benchmark_file.exists():
                    benchmark_file.touch()
                logger.info(f"✅ benchmark 文件生成: {benchmark_file}")

        # 最终股票池
        logger.info(f"最终股票池共 {len(codes)} 支（含 benchmark）")


        kq_symbols = [normalize_stock_code(c) for c in codes]
        dur_sec = {"1min": 60, "5min": 300, "day": 86400}.get(interval, 60)
        
        # 如果指定了year_segment，只下载该年份
        if year_segment is not None:
            segment_start = f"{year_segment}-01-01"
            segment_end = f"{year_segment}-12-31"
            start_dt = pd.Timestamp(segment_start)
            end_dt = pd.Timestamp(segment_end)
            logger.info(f"📅 分段下载模式: 只下载 {year_segment} 年 ({segment_start} ~ {segment_end})")
        else:
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end)
            logger.info(f"📅 全量下载模式: {start_dt} ~ {end_dt}")

        logger.info(f"下载区间: {start_dt} ~ {end_dt}, 频率: {interval}")
        logger.info(f"追加模式: {append_mode} (True=追加到已有文件, False=覆盖)")

        auth = TqAuth(username, password) if username and password else None
        api = TqApi(auth=auth)

        # --- 创建下载任务（支持追加模式） ---
        tasks = {}
        skipped_count = 0
        backup_dir = save_dir / "_backup" if append_mode else None
        if backup_dir and append_mode:
            backup_dir.mkdir(exist_ok=True)
        
        for sym, kq in zip(codes, kq_symbols):
            out_csv = save_dir / f"{sym.replace('.', '_')}.csv"
            temp_csv = save_dir / f"{sym.replace('.', '_')}_temp.csv"  # 临时文件用于下载
            
            # 追加模式：检查是否需要下载，并备份已有文件
            existing_data = None
            if append_mode and out_csv.exists():
                try:
                    # 读取现有文件
                    existing_df = pd.read_csv(out_csv, parse_dates=['datetime'], 
                                             on_bad_lines='skip', encoding='utf-8')
                    if not existing_df.empty:
                        existing_start = existing_df['datetime'].min()
                        existing_end = existing_df['datetime'].max()
                        
                        # 检查该年份的数据是否已存在
                        if year_segment is not None:
                            year_start = pd.Timestamp(f"{year_segment}-01-01")
                            year_end = pd.Timestamp(f"{year_segment}-12-31")
                            if year_start >= existing_start and year_end <= existing_end:
                                logger.info(f"⏭️  跳过 {sym}: {year_segment}年数据已存在 ({existing_start.date()} ~ {existing_end.date()})")
                                skipped_count += 1
                                continue
                        
                        # 备份已有文件
                        existing_data = existing_df
                        backup_file = backup_dir / f"{sym.replace('.', '_')}_backup.csv"
                        existing_df.to_csv(backup_file, index=False, encoding='utf-8-sig')
                        logger.info(f"💾 备份 {sym}: 现有数据 ({existing_start.date()} ~ {existing_end.date()})，将追加新数据")
                except Exception as e:
                    logger.warning(f"⚠️  读取 {sym} 现有文件失败: {e}，将重新下载")
            
            # 创建下载任务（下载到临时文件，避免覆盖）
            download_target = temp_csv if append_mode and existing_data is not None else out_csv
            tasks[sym] = {
                'downloader': DataDownloader(
                    api,
                    symbol_list=kq,
                    dur_sec=dur_sec,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    csv_file_name=str(download_target),
                ),
                'out_csv': out_csv,
                'temp_csv': temp_csv,
                'symbol': sym,
                'kq_symbol': kq,
                'has_existing': existing_data is not None
            }
            logger.info(f"创建任务: {sym} -> {kq} -> {download_target}")
        
        if skipped_count > 0:
            logger.info(f"⏭️  已跳过 {skipped_count} 个已有完整数据的股票")
        logger.info(f"📊 待下载任务数: {len(tasks)}")

        # --- 执行下载（带进度显示和超时检测） ---
        if len(tasks) == 0:
            logger.info("✅ 所有股票都已下载完成，无需下载")
            return
        
        import time
        start_time = time.time()
        last_progress_time = start_time
        timeout_seconds = 3600 * 24  # 24小时超时
        progress_interval = 300  # 每5分钟显示一次进度
        
        # 提取下载器对象
        downloaders = {sym: task['downloader'] for sym, task in tasks.items()}
        
        with closing(api):
            while not all(t.is_finished() for t in downloaders.values()):
                api.wait_update()
                
                # 检查超时
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    logger.error(f"⏰ 下载超时（{timeout_seconds/3600:.1f}小时），已运行 {elapsed/3600:.1f} 小时")
                    break
                
                # 定期显示进度
                if time.time() - last_progress_time > progress_interval:
                    finished_count = sum(1 for t in downloaders.values() if t.is_finished())
                    total_count = len(downloaders)
                    progress = finished_count / total_count * 100
                    elapsed_hours = elapsed / 3600
                    logger.info(f"📊 进度: {finished_count}/{total_count} ({progress:.1f}%), 已运行: {elapsed_hours:.1f} 小时")
                    last_progress_time = time.time()
        
        elapsed_total = time.time() - start_time
        finished_count = sum(1 for t in downloaders.values() if t.is_finished())
        logger.info(f"下载完成 ✅ 耗时: {elapsed_total/3600:.1f} 小时, 成功: {finished_count}/{len(tasks)}")
        
        # --- 追加模式：合并新下载的数据到已有文件 ---
        if append_mode:
            logger.info("🔄 开始合并数据（追加模式）...")
            merged_count = 0
            for sym, task_info in tasks.items():
                out_csv = task_info['out_csv']
                temp_csv = task_info.get('temp_csv')
                has_existing = task_info.get('has_existing', False)
                
                # 确定新数据文件（可能是临时文件或直接是输出文件）
                new_data_file = temp_csv if (temp_csv and temp_csv.exists()) else out_csv
                
                if not new_data_file.exists():
                    continue
                
                try:
                    # 读取新下载的数据
                    new_df = pd.read_csv(new_data_file, parse_dates=['datetime'], 
                                       on_bad_lines='skip', encoding='utf-8')
                    if new_df.empty:
                        # 如果新数据为空，删除临时文件
                        if temp_csv and temp_csv.exists():
                            temp_csv.unlink()
                        continue
                    
                    # 如果有已有数据，读取并合并
                    if has_existing:
                        backup_file = backup_dir / f"{sym.replace('.', '_')}_backup.csv"
                        if backup_file.exists():
                            existing_df = pd.read_csv(backup_file, parse_dates=['datetime'],
                                                     on_bad_lines='skip', encoding='utf-8')
                            # 合并数据
                            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                            # 去重（保留最新的）
                            combined_df = combined_df.drop_duplicates(subset=['datetime'], keep='last')
                            # 排序
                            combined_df = combined_df.sort_values('datetime')
                            # 保存
                            combined_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
                            logger.info(f"✅ 合并 {sym}: 已有 {len(existing_df)} 条 + 新增 {len(new_df)} 条 = 总计 {len(combined_df)} 条")
                        else:
                            # 备份文件不存在，直接使用新数据
                            new_df = new_df.drop_duplicates(subset=['datetime'], keep='last')
                            new_df = new_df.sort_values('datetime')
                            new_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
                            logger.info(f"✅ 保存 {sym}: {len(new_df)} 条（无备份文件）")
                    else:
                        # 没有已有数据，直接使用新数据
                        new_df = new_df.drop_duplicates(subset=['datetime'], keep='last')
                        new_df = new_df.sort_values('datetime')
                        new_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
                        logger.info(f"✅ 保存 {sym}: {len(new_df)} 条（新文件）")
                    
                    # 删除临时文件
                    if temp_csv and temp_csv.exists():
                        temp_csv.unlink()
                    
                    # 删除备份文件
                    if has_existing:
                        backup_file = backup_dir / f"{sym.replace('.', '_')}_backup.csv"
                        if backup_file.exists():
                            backup_file.unlink()
                    
                    merged_count += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️  合并 {sym} 数据失败: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            logger.info(f"✅ 数据合并完成: {merged_count}/{len(tasks)} 个文件")

        # --- 修正列名和 instrument 字段 ---
        self._normalize_csv(save_dir)

        # --- 生成日历与标的清单 ---
        self._ensure_calendar(save_dir, qlib_dir, interval)
        self._ensure_instruments(save_dir, qlib_dir)

        # --- 转换为 Qlib 数据 ---
        # Windows multiprocessing 修复：如果未指定，使用环境变量或默认值
        if max_workers is None:
            import os
            max_workers = int(os.environ.get("MAX_WORKERS", "2"))  # Windows 默认 2，减少进程池问题
        
        logger.info(f"使用并发数: {max_workers} (Windows 建议 1-2，避免 BrokenProcessPool 错误)")
        
        dump = DumpDataUpdate(
            data_path=str(save_dir),
            qlib_dir=str(qlib_dir),
            freq=interval,
            max_workers=max_workers,
            date_field_name="datetime",
            file_suffix=".csv",
            symbol_field_name="instrument",
        )
        dump.dump()
        logger.info("完成: ✅ 转换为 Qlib 格式")

    # ---------------------------------------------------------------------------------------
    def _normalize_csv(self, src_dir: Path):
        """将天勤导出的 CSV 文件标准化为 Qlib 格式"""
        for p in src_dir.glob("*.csv"):
            try:
                # 更容错的CSV读取
                df = pd.read_csv(p, on_bad_lines='skip', encoding='utf-8')
                if df.empty:
                    continue

                # 自动识别列名并改为 open/high/low/close/volume
                prefix = None
                for c in df.columns:
                    if ".open" in c:
                        prefix = c.split(".open")[0]
                        break

                if prefix:
                    rename_dict = {
                        f"{prefix}.open": "open",
                        f"{prefix}.high": "high",
                        f"{prefix}.low": "low",
                        f"{prefix}.close": "close",
                        f"{prefix}.volume": "volume",
                    }
                    df = df.rename(columns=rename_dict)

                # 规范时间列：TQSDK DataDownloader会自动生成datetime列
                # 如果列名不同，尝试自动识别
                datetime_col = None
                for col in df.columns:
                    col_lower = str(col).lower()
                    if col_lower in ['datetime', 'date', 'time', 'datetime']:
                        datetime_col = col
                        break
                
                if datetime_col is None:
                    logger.warning(f"⚠️  {p.name} 未找到datetime列，可用列: {list(df.columns)}")
                    continue
                
                # 如果列名不是datetime，重命名
                if datetime_col != "datetime":
                    df = df.rename(columns={datetime_col: "datetime"})
                
                # 规范时间格式（TQSDK生成的datetime格式可能不同，统一处理）
                df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce').dt.floor("S")

                # 从文件名推断标的
                fname = p.stem
                inst = fname.replace("_", ".").upper()
                if inst.startswith("SH."):
                    inst = f"{inst.split('.', 1)[1]}.SH"
                elif inst.startswith("SZ."):
                    inst = f"{inst.split('.', 1)[1]}.SZ"
                df["instrument"] = inst

                # 排序 + 导出
                df = df[["datetime", "open", "high", "low", "close", "volume", "instrument"]]
                df = df.dropna().sort_values("datetime")
                df.to_csv(p, index=False)
                logger.info(f"✅ 标准化完成: {p.name}")
            except Exception as e:
                logger.warning(f"⚠️ 无法处理 {p.name}: {e}")

    def _ensure_calendar(self, src_dir: Path, tgt_dir: Path, freq: str):
        cal_dir = tgt_dir / "calendars"
        cal_dir.mkdir(parents=True, exist_ok=True)
        freq_name = "day.txt" if freq.startswith("1d") else f"{freq}.txt"
        cal_file = cal_dir / freq_name

        if cal_file.exists():
            logger.info(f"已存在交易日历: {cal_file}")
            return

        all_dt = []
        for p in src_dir.glob("*.csv"):
            try:
                # 更容错的CSV读取
                df = pd.read_csv(p, usecols=["datetime"], parse_dates=["datetime"], 
                               on_bad_lines='skip', encoding='utf-8')
                if not df.empty:
                    all_dt.append(df["datetime"])
            except Exception as e:
                logger.warning(f"⚠️ 跳过文件 {p.name}: {e}")
                continue
        if not all_dt:
            return

        dt = pd.concat(all_dt).dropna().drop_duplicates().sort_values()
        fmt = "%Y-%m-%d %H:%M:%S" if "min" in freq else "%Y-%m-%d"
        with open(cal_file, "w", encoding="utf-8") as f:
            for ts in dt:
                # 确保ts是datetime对象
                if isinstance(ts, str):
                    ts = pd.to_datetime(ts)
                f.write(ts.strftime(fmt) + "\n")
        logger.info(f"✅ 生成交易日历: {cal_file}")

    def _ensure_instruments(self, src_dir: Path, tgt_dir: Path):
        inst_dir = tgt_dir / "instruments"
        inst_dir.mkdir(parents=True, exist_ok=True)
        inst_file = inst_dir / "all.txt"

        rows = []
        for p in src_dir.glob("*.csv"):
            try:
                df = pd.read_csv(p, usecols=["datetime"], parse_dates=["datetime"])
                if df.empty:
                    continue
                name = p.stem.replace("_", ".")
                if name.startswith("SH."):
                    name = f"{name.split('.',1)[1]}.SH"
                elif name.startswith("SZ."):
                    name = f"{name.split('.',1)[1]}.SZ"
                rows.append({
                    "instrument": name,
                    "start_time": df["datetime"].min(),
                    "end_time": df["datetime"].max(),
                })
            except Exception:
                continue

        if not rows:
            return
        out_df = pd.DataFrame(rows)
        out_df.to_csv(inst_file, header=False, index=False)
        logger.info(f"✅ 生成标的清单: {inst_file} ({len(out_df)} 条)")


if __name__ == "__main__":
    fire.Fire(KQDownloaderFixed)
