# ============================================
# 邮件发送脚本
# 功能：读取CSV信号，定时发送买入和卖出邮件
# ============================================

import pandas as pd
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import os
import json

class EmailSender:
    _qlib_initialized = False  # 类级别标志，避免重复初始化
    
    def __init__(self):
        # 邮件配置
        self.EMAIL_ADDRESS = 'chloechen@xcquant.com'
        self.EMAIL_PASSWORD = 'SX8DzzlyqbvwOk5V'
        
        # 收件人列表（主要收件人）
        self.RECIPIENT_LIST = ['chloechen@xcquant.com']
        
        # 抄送列表
        self.CC_LIST = ['chloechen@xcquant.com']
        
        # 文件路径
        self.CSV_FILE = Path("daily_predictions.csv")
        self.POSITIONS_FILE = Path("positions.json")  # 持仓记录文件
        
        print(f"[INIT] 邮件发送器初始化完成")
        print(f"[INFO] CSV文件: {self.CSV_FILE}")
        print(f"[INFO] 持仓文件: {self.POSITIONS_FILE}")
    
    def read_signal_csv(self):
        """读取信号CSV文件（带真实性校验：当日且非空）"""
        try:
            if not self.CSV_FILE.exists():
                print(f"[WARN] CSV文件不存在: {self.CSV_FILE}")
                return None
            # 校验文件修改时间为今日
            mtime = datetime.fromtimestamp(self.CSV_FILE.stat().st_mtime)
            today = datetime.now().date()
            if mtime.date() != today:
                print(f"[WARN] CSV非当日生成，mtime={mtime}")
                return None
            df = pd.read_csv(self.CSV_FILE)
            if df is None or len(df) == 0:
                print(f"[WARN] CSV为空: {self.CSV_FILE}")
                return None
            # 如果有timestamp列，校验为今日
            if 'timestamp' in df.columns:
                try:
                    ts0 = pd.to_datetime(df['timestamp'].iloc[0])
                    if ts0.date() != today:
                        print(f"[WARN] CSV时间戳非当日: {ts0}")
                        return None
                except Exception:
                    pass
            print(f"[OK] 读取信号成功: {len(df)} 条 (mtime={mtime})")
            return df
        except Exception as e:
            print(f"[ERROR] 读取CSV失败: {e}")
            return None
    
    def save_positions(self, df):
        """保存持仓信息（买入时调用）"""
        try:
            buy_date = datetime.now().strftime('%Y-%m-%d')
            
            positions = {}
            for _, row in df.iterrows():
                code = row['code']
                positions[code] = {
                    'buy_date': buy_date,
                    'buy_price': float(row['price']),
                    'shares': int(row['shares']),
                    'code': code,
                    'rank': int(row['rank']),
                    'score': float(row['score'])
                }
            
            # 保存为JSON
            with open(self.POSITIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(positions, f, indent=2, ensure_ascii=False)
            
            print(f"[OK] 持仓信息已保存: {len(positions)} 个")
            return True
            
        except Exception as e:
            print(f"[ERROR] 保存持仓失败: {e}")
            return False
    
    def load_positions(self):
        """读取持仓信息"""
        try:
            if not self.POSITIONS_FILE.exists():
                print(f"[WARN] 持仓文件不存在: {self.POSITIONS_FILE}")
                return {}
            
            with open(self.POSITIONS_FILE, 'r', encoding='utf-8') as f:
                positions = json.load(f)
            
            print(f"[OK] 读取持仓信息: {len(positions)} 个")
            return positions
            
        except Exception as e:
            print(f"[ERROR] 读取持仓失败: {e}")
            return {}
    
    def get_current_price(self, code, target_time=None):
        """
        直接从C:/Users/ASUS/qlib_data_recent/features/<code>/data.csv读取对应时间close价。
        入参code可能是 sh.603072 / sz.000001 / SH600519 / SZ000001 等，需统一为 SHxxxxxx / SZxxxxxx。
        """
        try:
            # 统一代码为文件夹格式 SHxxxxxx / SZxxxxxx
            raw = str(code).strip()
            up = raw.upper()
            folder_code = up
            if up.startswith('SH.') and len(up) > 3:
                folder_code = 'SH' + up.split('.', 1)[1]
            elif up.startswith('SZ.') and len(up) > 3:
                folder_code = 'SZ' + up.split('.', 1)[1]
            elif up.startswith('SH') and len(up) > 2:
                folder_code = 'SH' + up[2:]
            elif up.startswith('SZ') and len(up) > 2:
                folder_code = 'SZ' + up[2:]
            # 其余情况直接用大写

            folder = Path("C:/Users/ASUS/qlib_data_recent/features") / folder_code
            csv_file = folder / "data.csv"
            if not csv_file.exists():
                print(f"[WARN] {folder_code} data.csv不存在 ({csv_file})")
                return None
            df = pd.read_csv(csv_file)
            # 决定查价时间
            if target_time is not None:
                price_dt = datetime.strptime(target_time, "%Y-%m-%d %H:%M:%S").strftime("%Y/%m/%d 10:46")
            else:
                now = datetime.now()
                price_dt = now.strftime("%Y/%m/%d 10:46")
            # 回溯找有数据的最近10:46
            for d in range(0, 7):
                dt = (datetime.strptime(price_dt[:10], "%Y/%m/%d") - timedelta(days=d)).strftime("%Y/%m/%d 10:46")
                row = df[df['datetime'] == dt]
                if not row.empty:
                    price = float(row.iloc[0]['close'])
                    return price
            print(f"[WARN] {folder_code} 没有7天内10:46分数据")
            return None
        except Exception as e:
            print(f"[ERROR] 获取{code} 价格失败: {e}")
            return None
    
    def generate_buy_email_html(self, df):
        """生成买入邮件HTML"""
        datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        count = len(df)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset='utf-8'>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                h2 {{ color: #1e88e5; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #1e88e5; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .rank {{ font-weight: bold; color: #1976d2; }}
                .code {{ font-weight: bold; color: #d32f2f; }}
                .price {{ color: #388e3c; }}
                .score {{ color: #f57c00; }}
            </style>
        </head>
        <body>
            <h2>📈 买入信号 - {datetime_str}</h2>
            <p>共 {count} 个买入信号</p>
            <table>
                <tr>
                    <th>排名</th>
                    <th>代码</th>
                    <th>信号分数</th>
                    <th>目标价格</th>
                    <th>建议股数</th>
                </tr>
        """
        
        for _, row in df.iterrows():
            html += f"""
                <tr>
                    <td class="rank">{row['rank']}</td>
                    <td class="code">{row['code']}</td>
                    <td class="score">{row['score']:.4f}</td>
                    <td class="price">{row['price']:.2f}</td>
                    <td>{row['shares']}</td>
                </tr>
            """
        
        html += """
            </table>
            <p style="color: #666; font-size: 12px;">本邮件由量化交易系统自动发送</p>
        </body>
        </html>
        """
        
        return html
    
    def generate_sell_email_html(self, df_with_profit):
        """生成卖出邮件HTML（包含收益信息）"""
        # 计算汇总统计
        total_profit = df_with_profit['profit'].sum()
        total_profit_rate = df_with_profit['profit_rate'].mean() if len(df_with_profit) > 0 else 0
        win_count = len(df_with_profit[df_with_profit['profit'] > 0])
        loss_count = len(df_with_profit[df_with_profit['profit'] < 0])
        
        datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        count = len(df_with_profit)
        total_color = '#388e3c' if total_profit >= 0 else '#e53935'
        rate_color = '#388e3c' if total_profit_rate >= 0 else '#e53935'
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset='utf-8'>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                h2 {{ color: #e53935; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .summary-item {{ margin: 5px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #e53935; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .rank {{ font-weight: bold; color: #1976d2; }}
                .code {{ font-weight: bold; color: #d32f2f; }}
                .price {{ color: #388e3c; }}
                .profit-positive {{ color: #388e3c; font-weight: bold; }}
                .profit-negative {{ color: #e53935; font-weight: bold; }}
                .profit-rate-positive {{ color: #388e3c; }}
                .profit-rate-negative {{ color: #e53935; }}
            </style>
        </head>
        <body>
            <h2>📉 卖出信号 - {datetime_str}</h2>
            <div class="summary">
                <h3>收益汇总</h3>
                <div class="summary-item">总盈亏: <span style="color: {total_color}; font-weight: bold;">{total_profit:.2f} 元</span></div>
                <div class="summary-item">平均收益率: <span style="color: {rate_color}; font-weight: bold;">{total_profit_rate*100:.2f}%</span></div>
                <div class="summary-item">盈利数量: <span style="color: #388e3c; font-weight: bold;">{win_count}</span> | 亏损数量: <span style="color: #e53935; font-weight: bold;">{loss_count}</span></div>
            </div>
            <p>共 {count} 个卖出信号</p>
            <table>
                <tr>
                    <th>排名</th>
                    <th>代码</th>
                    <th>买入价格</th>
                    <th>卖出价格</th>
                    <th>股数</th>
                    <th>收益率</th>
                    <th>盈亏金额</th>
                </tr>
        """
        
        for _, row in df_with_profit.iterrows():
            profit_class = 'profit-positive' if row['profit'] >= 0 else 'profit-negative'
            rate_class = 'profit-rate-positive' if row['profit_rate'] >= 0 else 'profit-rate-negative'
            
            html += f"""
                <tr>
                    <td class="rank">{row['rank']}</td>
                    <td class="code">{row['code']}</td>
                    <td class="price">{row['buy_price']:.2f}</td>
                    <td class="price">{row['sell_price']:.2f}</td>
                    <td>{int(row['shares'])}</td>
                    <td class="{rate_class}">{row['profit_rate']*100:.2f}%</td>
                    <td class="{profit_class}">{row['profit']:.2f}</td>
                </tr>
            """
        
        html += """
            </table>
            <p style="color: #666; font-size: 12px;">本邮件由量化交易系统自动发送</p>
        </body>
        </html>
        """
        
        return html
    
    def send_email(self, subject, html_content, signal_type='buy'):
        """发送邮件"""
        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.EMAIL_ADDRESS
            msg['To'] = ', '.join(self.RECIPIENT_LIST)  # 主要收件人
            msg['Cc'] = ', '.join(self.CC_LIST) if self.CC_LIST else ''  # 抄送列表
            msg['Subject'] = Header(subject, 'utf-8')
            
            # 添加HTML内容
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
            # 构建所有收件人列表（包含收件人+抄送）
            all_recipients = self.RECIPIENT_LIST + self.CC_LIST
            # 去重
            all_recipients = list(dict.fromkeys(all_recipients))
            
            # 发送邮件（使用SSL，465端口）
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL('smtp.feishu.cn', 465, context=context) as server:
                server.login(self.EMAIL_ADDRESS, self.EMAIL_PASSWORD)
                server.sendmail(self.EMAIL_ADDRESS, all_recipients, msg.as_string())
            
            print(f"[OK] {signal_type}邮件发送成功: {subject}")
            print(f"[INFO] 收件人: {len(self.RECIPIENT_LIST)}人, 抄送: {len(self.CC_LIST)}人")
            return True
            
        except Exception as e:
            print(f"[ERROR] 邮件发送失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def send_buy_email(self):
        """发送买入邮件（14:45）"""
        df = self.read_signal_csv()
        if df is None:
            print("[ABORT] 未检测到当日真实预测CSV，取消发送买入邮件。")
            return
        if len(df) == 0:
            print("[WARN] CSV文件为空，取消发送")
            return
        buy_signals = df.sort_values('rank').head(50)
        html = self.generate_buy_email_html(buy_signals)
        subject = f"买入信号 - {datetime.now().strftime('%Y-%m-%d')}"
        success = self.send_email(subject, html, 'buy')
        if success:
            self.save_positions(buy_signals)
    
    def send_sell_email(self):
        """发送卖出邮件（10:46）"""
        # 读取持仓信息
        positions = self.load_positions()
        if len(positions) == 0:
            print("[WARN] 没有持仓信息，无法计算收益")
            return
        
        print(f"[INFO] 开始计算收益，持仓数量: {len(positions)}")
        
        # 准备收益数据
        profit_data = []
        
        for code, position in positions.items():
            buy_price = position['buy_price']
            shares = position['shares']
            
            # 获取当前价格
            sell_price = self.get_current_price(code)
            if sell_price is None:
                print(f"[WARN] {code} 无法获取当前价格，跳过")
                continue
            
            # 计算收益
            profit = (sell_price - buy_price) * shares
            profit_rate = (sell_price - buy_price) / buy_price if buy_price > 0 else 0
            
            profit_data.append({
                'code': code,
                'rank': position.get('rank', 0),
                'buy_price': buy_price,
                'sell_price': sell_price,
                'shares': shares,
                'profit': profit,
                'profit_rate': profit_rate,
                'buy_date': position.get('buy_date', '')
            })
        
        if len(profit_data) == 0:
            print("[WARN] 没有可用的收益数据")
            return
        
        # 转换为DataFrame
        df_with_profit = pd.DataFrame(profit_data)
        df_with_profit = df_with_profit.sort_values('rank')
        
        # 生成HTML
        html = self.generate_sell_email_html(df_with_profit)
        
        # 发送邮件
        subject = f"卖出信号 - {datetime.now().strftime('%Y-%m-%d')}"
        self.send_email(subject, html, 'sell')
        
        # 打印收益汇总
        total_profit = df_with_profit['profit'].sum()
        avg_rate = df_with_profit['profit_rate'].mean() * 100
        print(f"[SUMMARY] 总盈亏: {total_profit:.2f} 元, 平均收益率: {avg_rate:.2f}%")
    
    def start_scheduler(self):
        """启动定时任务"""
        print("\n" + "="*70)
        print("[SCHEDULER] 邮件发送定时任务")
        print("="*70)
        print("[INFO] 买入邮件: 每天 14:45")
        print("[INFO] 卖出邮件: 每天 10:46")
        print("="*70 + "\n")
        
        # 设置定时任务
        schedule.every().day.at("14:45").do(self.send_buy_email)
        schedule.every().day.at("10:46").do(self.send_sell_email)
        
        # 检查当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[TIME] 当前时间: {current_time}")
        print("[INFO] 等待定时任务触发...")
        
        # 主循环
        print("\n[START] 程序开始持续运行，等待定时任务...")
        print("[INFO] 按 Ctrl+C 停止程序")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
        except KeyboardInterrupt:
            print("\n[STOP] 程序已停止")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='邮件发送器')
    parser.add_argument('--buy-now', action='store_true', help='立即发送买入邮件')
    parser.add_argument('--sell-now', action='store_true', help='立即发送卖出邮件（10:46价格逻辑）')
    args = parser.parse_args()
    
    sender = EmailSender()
    
    if args.buy_now:
        print("[NOW] 立即发送买入邮件...")
        sender.send_buy_email()
        return
    if args.sell_now:
        print("[NOW] 立即发送卖出邮件...")
        sender.send_sell_email()
        return
    
    # 未指定立即发送时，进入定时任务
    sender.start_scheduler()

if __name__ == "__main__":
    main()
