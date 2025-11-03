"""
KQ_tick_realtime.py

功能：
1. 实时采集指定股票的 tick 数据
2. 每隔 INTERVAL 秒更新一次
3. 实时追加模式，将 tick 写入 CSV 文件
4. 数据表头保持统一，方便后续分析

使用方法：
- 实时运行：python KQ_tick_realtime.py
- 写示例行：python KQ_tick_realtime.py --demo（无需连接即可预览输出格式）
- 终止：Ctrl+C（或可扩展定时退出参数）。

"""

import os
import pandas as pd
from tqsdk import TqApi, TqAuth
from datetime import datetime
import time
import sys
import argparse

# ---------------- 参数配置 ----------------
KQ_USER = "xclight"
KQ_PASSWORD = "xclight666"
INTERVAL = 1  # 秒
stocks = ["sh.688041", "sz.300750", "sh.603019","sh.688256", "sh.688183","sz.002837","sz.300803",
          "sh.688521", "sz.300757", "sh.601138"]
SAVE_PATH = "./realtime_data_collect"
os.makedirs(SAVE_PATH, exist_ok=True)

# ---------------- 定义固定表头 ----------------
COLUMNS = [
    "代码", "交易所代码", "自然日", "时间", "成交价", "成交量", "成交额", "成交笔数",
    "IOPV", "成交标志", "BS标志", "当日累计成交量", "当日成交额",
    "最高价", "最低价", "开盘价", "前收盘"
] + [f"申卖价{i}" for i in range(1, 11)] + [f"申卖量{i}" for i in range(1, 11)] \
  + [f"申买价{i}" for i in range(1, 11)] + [f"申买量{i}" for i in range(1, 11)] \
  + ["加权平均叫卖价", "加权平均叫买价", "叫卖总量", "叫买总量",
     "不加权指数", "品种总数", "上涨品种数", "下跌品种数", "持平品种数", "交易时段标识"]

# ---------------- 股票代码转换 ----------------
def _normalize_symbol(symbol: str) -> str:
    if symbol.lower().startswith("sh."):
        return "SSE." + symbol.split(".", 1)[1]
    if symbol.lower().startswith("sz."):
        return "SZSE." + symbol.split(".", 1)[1]
    return symbol

# ---------------- 主函数 ----------------
def _disable_proxy_env():
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

# ---------------- 交易时段识别 ----------------
def get_session_label(ts: pd.Timestamp) -> str:
    ts_sh = ts.tz_convert("Asia/Shanghai") if ts.tz is not None else ts.tz_localize("Asia/Shanghai")
    hms = int(ts_sh.strftime("%H%M%S"))
    if 91500 <= hms <= 92500:
        return "开盘集合竞价"
    elif 93000 <= hms <= 113000 or 130000 <= hms <= 145700:
        return "连续竞价"
    elif 145700 <= hms <= 150003:
        return "收盘集合竞价"
    else:
        return "休市"


def run_realtime():
    _disable_proxy_env()
    try:
        api = TqApi(auth=TqAuth(KQ_USER, KQ_PASSWORD))
    except Exception as e:
        print(f"初始化 TqApi 失败: {e}")
        return

    # 初始化 CSV 文件，如果不存在就创建表头
    today = datetime.now().strftime("%Y-%m-%d")

    
    for stock in stocks:
        stock_new = stock.replace(".", "")
        today_new_simple = today.replace("-", "")
        csv_file = os.path.join(SAVE_PATH, f"{stock_new}_{today_new_simple}.csv")
        if not os.path.exists(csv_file):
            pd.DataFrame(columns=COLUMNS).to_csv(csv_file, index=False, encoding="utf-8-sig")

    # 订阅行情（回调式：通过 wait_update 触发写入）
    quotes = {}
    for stock in stocks:
        try:
            quotes[stock] = api.get_quote(_normalize_symbol(stock))
        except Exception as e:
            print(f"订阅 {stock} 失败: {e}")
    
    # 初始化每只股票最后写入时间，防止重复写入
    last_tick_write_time = {stock: None for stock in stocks}
    

    try:
        while True:
            # 等待有更新，设置超时，超时则执行心跳写入
            now = pd.Timestamp.now(tz="Asia/Shanghai")
            #trading = _is_trading_time(now)
            session_label = get_session_label(now)  # 获取时段标识
            
            api.wait_update(deadline=time.time() + INTERVAL)

        
            for stock, tick in quotes.items():
                # 当该合约的行情时间发生变化时写入一行
                if api.is_changing(tick, "datetime"):
                    row = {
                        "代码": stock[3:],
                        "交易所代码": "SZSE" if stock.startswith("sz.") else "SSE",
                        "自然日": now.strftime("%Y-%m-%d"),
                        "时间": now.strftime("%H:%M:%S"),
                        "成交价": getattr(tick, "last_price", pd.NA),
                        "成交量": getattr(tick, "volume", pd.NA),
                        "成交额": getattr(tick, "amount", pd.NA),
                        "成交笔数": 0,
                        "IOPV": 0,
                        "成交标志": 0,
                        "BS标志": 0,
                        "当日累计成交量": getattr(tick, "volume", pd.NA),
                        "当日成交额": getattr(tick, "amount", pd.NA),
                        "最高价": getattr(tick, "highest", pd.NA),
                        "最低价": getattr(tick, "lowest", pd.NA),
                        "开盘价": getattr(tick, "open", pd.NA),
                        "前收盘": getattr(tick, "pre_close", pd.NA),
                    }

                    # 取四组数组
                    ask_prices = [getattr(tick, f"ask_price{i}", 0) for i in range(1, 11)]
                    ask_vols = [getattr(tick, f"ask_volume{i}", 0) for i in range(1, 11)]
                    bid_prices = [getattr(tick, f"bid_price{i}", 0) for i in range(1, 11)]
                    bid_vols = [getattr(tick, f"bid_volume{i}", 0) for i in range(1, 11)]
                    # 按表头顺序写入
                    for i in range(1, 11):
                        row[f"申卖价{i}"] = ask_prices[i-1]
                    for i in range(1, 11):
                        row[f"申卖量{i}"] = ask_vols[i-1]
                    for i in range(1, 11):
                        row[f"申买价{i}"] = bid_prices[i-1]
                    for i in range(1, 11):
                        row[f"申买量{i}"] = bid_vols[i-1]

                    row["加权平均叫卖价"] = sum(p*v for p,v in zip(ask_prices, ask_vols))/max(sum(ask_vols),1)
                    row["加权平均叫买价"] = sum(p*v for p,v in zip(bid_prices, bid_vols))/max(sum(bid_vols),1)
                    row["叫卖总量"] = sum(ask_vols)
                    row["叫买总量"] = sum(bid_vols)
                    row["不加权指数"] = 0
                    row["品种总数"] = 0
                    row["上涨品种数"] = 0
                    row["下跌品种数"] = 0
                    row["持平品种数"] = 0
                    row["交易时段标识"] = get_session_label(now)
                    
                    row_time = now.strftime("%H:%M:%S")

                    stock_new = stock.replace(".", "")
                    today_new_simple = today.replace("-", "")
                    csv_file = os.path.join(SAVE_PATH, f"{stock_new}_{today_new_simple}.csv")

            
                    if last_tick_write_time[stock] != row_time:
                        pd.DataFrame([row], columns=COLUMNS).to_csv(
                            csv_file, mode='a', index=False, header=False, encoding="utf-8-sig"
                        )
                        last_tick_write_time[stock] = row_time

                    print(f"{now.strftime('%H:%M:%S')} 写入 {stock} -> {csv_file}", flush=True)

    except KeyboardInterrupt:
        print("已停止采集")
    finally:
        api.close()
        print("采集结束")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="写入每个标的的一行示例数据后退出")
    args = parser.parse_args()

    if args.demo:
        # 写示例文件（不依赖实时行情）
        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now()
        for stock in stocks:
            stock_new = stock.replace(".", "")
            today_new_simple = today.replace("-", "")
            csv_file = os.path.join(SAVE_PATH, f"{stock_new}_{today_new_simple}.csv")
            if not os.path.exists(csv_file):
                pd.DataFrame(columns=COLUMNS).to_csv(csv_file, index=False, encoding="utf-8-sig")
            demo_row = {
                "代码": stock,
                "交易所代码": 0 if stock.startswith("sh.") else 1,
                "自然日": today,
                "时间": now.strftime("%H:%M:%S"),
                "成交价": 0, "成交量": 0, "成交额": 0, "成交笔数": 0,
                "IOPV": 0, "成交标志": -1, "BS标志": -1,
                "当日累计成交量": 0, "当日成交额": 0,
                "最高价": 0, "最低价": 0, "开盘价": 0, "前收盘": 0,
                "加权平均叫卖价": 0, "加权平均叫买价": 0,
                "叫卖总量": 0, "叫买总量": 0,
                "不加权指数": 0, "品种总数": 0, "上涨品种数": 0, "下跌品种数": 0, "持平品种数": 0,
                "交易时段标识": get_session_label(now)
            }
            for i in range(1, 11):
                demo_row[f"申卖价{i}"] = 0
                demo_row[f"申卖量{i}"] = 0
                demo_row[f"申买价{i}"] = 0
                demo_row[f"申买量{i}"] = 0
            pd.DataFrame([demo_row]).to_csv(csv_file, mode='a', index=False, header=False, encoding="utf-8-sig")
            print(f"示例已写入: {csv_file}")
    else:
        run_realtime()
