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
        benchmark_dir: str = None         # ⚡ 新增
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
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        logger.info(f"下载区间: {start_dt} ~ {end_dt}, 频率: {interval}")

        auth = TqAuth(username, password) if username and password else None
        api = TqApi(auth=auth)

        # --- 创建下载任务 ---
        tasks = {}
        for sym, kq in zip(codes, kq_symbols):
            out_csv = save_dir / f"{sym.replace('.', '_')}.csv"
            tasks[sym] = DataDownloader(
                api,
                symbol_list=kq,
                dur_sec=dur_sec,
                start_dt=start_dt,
                end_dt=end_dt,
                csv_file_name=str(out_csv),
            )
            logger.info(f"创建任务: {sym} -> {kq} -> {out_csv}")

        # --- 执行下载 ---
        with closing(api):
            while not all(t.is_finished() for t in tasks.values()):
                api.wait_update()
        logger.info("下载完成 ✅")

        # --- 修正列名和 instrument 字段 ---
        self._normalize_csv(save_dir)

        # --- 生成日历与标的清单 ---
        self._ensure_calendar(save_dir, qlib_dir, interval)
        self._ensure_instruments(save_dir, qlib_dir)

        # --- 转换为 Qlib 数据 ---
        dump = DumpDataUpdate(
            data_path=str(save_dir),
            qlib_dir=str(qlib_dir),
            freq=interval,
            max_workers=4,
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
                df = pd.read_csv(p)
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

                # 规范时间列
                df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("S")

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
                df = pd.read_csv(p, usecols=["datetime"], parse_dates=["datetime"])
                all_dt.append(df["datetime"])
            except Exception:
                continue
        if not all_dt:
            return

        dt = pd.concat(all_dt).dropna().drop_duplicates().sort_values()
        fmt = "%Y-%m-%d %H:%M:%S" if "min" in freq else "%Y-%m-%d"
        with open(cal_file, "w", encoding="utf-8") as f:
            for ts in dt:
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
