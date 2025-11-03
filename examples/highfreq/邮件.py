import pandas as pd
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
from openpyxl import load_workbook
from datetime import datetime
import os

# =============== 飞书邮箱配置 =============== 

SMTP_SERVER = 'smtp.feishu.cn'
SMTP_PORT = 465  # 587或465（推荐使用587，兼容性更好）
EMAIL_ADDRESS = 'XXXXX@xcquant.com'  # 您的飞书邮箱
EMAIL_PASSWORD = 'XXXXXXXX'  # 飞书生成的SMTP授权码

# 收件人列表（可以是多个邮箱）

RECIPIENT_LIST = [
    'cloudwang@xcquant.com',
    'carrielu@xcquant.com',
    'jack@xcquant.com',
    'yudachihu@xcquant.com',
    'goldenguo@xcquant.com'

]

# 抄送列表（包含自己）
CC_LIST = [
    'xxxxxxn@xcquant.com'      # 可选备份邮箱
]

# =============== 程序核心功能 =============== 
# 安全增强：创建SSL上下文
context = ssl.create_default_context()

def create_email_body(text_data):
    """创建专业HTML邮件内容"""
    # 获取当前日期
    today_str = datetime.now().strftime('%Y年%m月%d日')
    
    # 样式优化（兼容主流邮件客户端）
    html = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        body {{ 
            font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif; 
            max-width: 650px; 
            margin: 0 auto; 
            background-color: #f6f8fa;
            color: #333;
        }}
        .header {{
            background: #3370FF; 
            color: white; 
            padding: 25px 20px; 
            text-align: center; 
            border-radius: 8px 8px 0 0;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
            font-weight: 500;
        }}
        .header .date {{
            font-size: 14px; 
            opacity: 0.9; 
            margin-top: 8px;
        }}
        .card {{
            background: white;
            border-radius: 8px; 
            box-shadow: 0 3px 8px rgba(0,0,0,0.05); 
            padding: 20px; 
            margin: 0 0 20px 0;
        }}
        .card-title {{
            color: #1F2329; 
            font-size: 18px;
            margin: 0 0 15px 0;
            padding-bottom: 12px;
            border-bottom: 1px solid #eaeef5;
        }}
        .stats-grid {{
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 15px; 
            margin: 15px 0;
        }}
        .stat-item {{
            background: #f0f5ff; 
            border-radius: 6px; 
            padding: 12px; 
            text-align: center;
        }}
        .stat-label {{
            font-size: 14px; 
            color: #646A73;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 18px; 
            font-weight: 600;
        }}
        .up {{ color: #00b578 !important; }}
        .down {{ color: #ff3141 !important; }}
        .ranking-grid {{
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 12px;
        }}
        .ranking-box {{
            background: #f2f7ff; 
            border-radius: 6px; 
            padding: 15px;
        }}
        .ranking-title {{
            font-weight: 600; 
            margin-bottom: 10px;
            font-size: 15px;
            color: #3370FF;
        }}
        .ranking-item {{
            padding: 8px 0; 
            font-size: 14px;
            border-bottom: 1px dashed #eaeef5;
        }}
        .ranking-item:last-child {{
            border-bottom: none;
        }}
        .table-container {{ 
            margin-top: 20px;
            overflow-x: auto;
        }}
        .sector-table {{
            width: 100%; 
            border-collapse: collapse;
        }}
        .sector-table th {{
            background: #3370FF;
            color: white;
            padding: 10px 12px;
            text-align: left;
            font-weight: 500;
        }}
        .sector-table td {{
            padding: 10px 12px; 
            border-bottom: 1px solid #eaeef5;
        }}
        .sector-table tr:nth-child(even) {{
            background: #f7faff;
        }}
        .sector-table .up {{ 
            color: #00b578;
            font-weight: 500;
        }}
        .sector-table .down {{ 
            color: #ff3141;
            font-weight: 500;
        }}
        .market-overview {{
            background: #f9fbfd;
            border-left: 3px solid #3370FF;
            padding: 12px 15px;
            margin: 15px 0;
            border-radius: 0 4px 4px 0;
        }}
        .footer {{
            text-align: center; 
            color: #8a949e; 
            font-size: 13px; 
            margin: 30px 0 20px;
        }}
        .recipient-note {{
            font-size: 12px; 
            color: #8a949e; 
            text-align: center;
            margin-top: 5px;
        }}
    </style>
    </head>
    <body>
        <div class="header">
            <h1>全天候策略动态简报</h1>
            <div class="date">{today_str}</div>
        </div>
    """
    
    # 1. 核心指标卡片
    html += '<div class="card"><h3 class="card-title">📊 关键指标</h3><div class="stats-grid">'
    
    # 提取核心数据

    rank_patterns_1 = {
        "总涨额": r"总涨额[：:]\s*(.*?)\s*总涨幅",
        "总涨幅": r"总涨幅[：:]\s*(.*?)\s*日涨额",
        "日涨额": r"日涨额[：:]\s*(.*?)\s*日涨幅",
        "日涨幅": r"日涨幅[：:]\s*(.*?)\s*，结束。"
    }
       
    rank_data_1 = {}
    for title, pattern in rank_patterns_1.items():
        match = re.search(pattern, text_data, re.DOTALL)
        rank_data_1[title] = [item.strip() for item in match.group(1).split(',')] if match else []
    
    # 图标映射
    rank_icons_1 = {
        "总涨额": "💰",
        "总涨幅": "📈",
        "日涨额": "💹",
        "日涨幅": "🚀"      
    }
    
    for title, items in rank_data_1.items():
        if items:
            # 为标题添加图标
            display_title = f"{rank_icons_1.get(title, '')} {title}"
            html += f'<div class="ranking-box"><div class="ranking-title">{display_title}</div>'
            for item in items:
                # 检查数值的正负，添加颜色
                colored_item = item
                # 查找包含负号的数字，优先考虑这些作为判断依据
                negative_numbers = re.findall(r'-\d*\.?\d+', item)
                positive_numbers = re.findall(r'\d+\.?\d*', item)
                
                if negative_numbers:
                    # 如果有负数，直接判断为负数
                    colored_item = f'<span style="color: #00b578;">{item}</span>'
                elif positive_numbers:
                    # 检查正数是否包含在百分比等格式中
                    number_str = positive_numbers[-1]
                    number = float(number_str)
                    if number > 0:
                        colored_item = f'<span style="color: #ff3141;">{item}</span>'
                html += f'<div class="ranking-item">{colored_item}</div>'
            html += '</div>'
    
    # 2. 排名卡片（左右布局）
    html += '</div></div><div class="card"><h3 class="card-title">🏆 市场表现排名</h3><div class="ranking-grid">'
    
    # 提取所有排名数据（增强容错）
    rank_patterns = {
        "总涨幅前三": r"总涨幅前三[：:]\s*(.*?)\s*总涨额前三",
        "总涨额前三": r"总涨额前三[：:]\s*(.*?)\s*总跌幅前三",
        "总跌幅前三": r"总跌幅前三[：:]\s*(.*?)\s*总跌额前三",
        "总跌额前三": r"总跌额前三[：:]\s*(.*?)\s*日涨幅前三",
        "日涨幅前三": r"日涨幅前三[：:]\s*(.*?)\s*日涨额前三",
        "日涨额前三": r"日涨额前三[：:]\s*(.*?)\s*日跌幅前三",
        "日跌幅前三": r"日跌幅前三[：:]\s*(.*?)\s*日跌额前三",
        "日跌额前三": r"日跌额前三[：:]\s*(.*?)\s*。结束"
    }

    rank_data = {}
    for title, pattern in rank_patterns.items():
        match = re.search(pattern, text_data, re.DOTALL)
        rank_data[title] = [item.strip() for item in match.group(1).split(',')] if match else []

    # 图标映射
    rank_icons = {
        "总涨幅前三": "📈",
        "总涨额前三": "💰",
        "总跌幅前三": "📉",
        "总跌额前三": "💸",
        "日涨幅前三": "🚀",
        "日涨额前三": "💹",
        "日跌幅前三": "🔻",
        "日跌额前三": "❗"
    }

    for title, items in rank_data.items():
        if items:
            # 为标题添加图标
            display_title = f"{rank_icons.get(title, '')} {title}"
            html += f'<div class="ranking-box"><div class="ranking-title">{display_title}</div>'
            for item in items:
                # 检查数值的正负，添加颜色
                colored_item = item
                # 查找包含负号的数字，优先考虑这些作为判断依据
                negative_numbers = re.findall(r'-\d*\.?\d+', item)
                positive_numbers = re.findall(r'\d+\.?\d*', item)
                
                if negative_numbers:
                    # 如果有负数，直接判断为负数
                    colored_item = f'<span style="color: #00b578;">{item}</span>'
                elif positive_numbers:
                    # 检查正数是否包含在百分比等格式中
                    number_str = positive_numbers[-1]
                    number = float(number_str)
                    if number > 0:
                        colored_item = f'<span style="color: #ff3141;">{item}</span>'
                html += f'<div class="ranking-item">{colored_item}</div>'
            html += '</div>'
       
    # 3. 市场概览卡片
    html += '</div></div><div class="card"><h3 class="card-title">🌐 市场全景</h3>'
    
    # 市场概览
    market_overview = re.search(r"市场概览[：:]\s*(.*?)\s*$", text_data, re.MULTILINE)
    if market_overview:
        html += f'<div class="market-overview">{market_overview.group(1).strip()}</div>'
    
    # A股行情
    a_shares = re.search(r"A股方面[，,]\s*(.*?)\s*$", text_data, re.MULTILINE)
    if a_shares:
        html += f'<p><strong>🇨🇳 A股市场</strong>: {a_shares.group(1).strip()}</p>'
    
    # 港股行情
    hk_shares = re.search(r"港股方面[，,]\s*(.*?)\s*$", text_data, re.MULTILINE)
    if hk_shares:
        html += f'<p><strong>🇭🇰 港股市场</strong>: {hk_shares.group(1).strip()}</p>'
    
    # 资金动向
    capital = re.search(r"资金方面[，,]\s*(.*?)\s*$", text_data, re.MULTILINE)
    if capital:
        html += f'<p><strong>💰 资金动向</strong>: {capital.group(1).strip()}</p>'
    
    # 板块行情表格
    sectors = re.search(r"板块方面[：:]\s*(.*?)\s*$", text_data, re.MULTILINE)
    if sectors:
        sector_content = sectors.group(1).strip()
        # 按分号分割板块表现内容
        sector_items = [item.strip() for item in sector_content.split('；') if item.strip()]
        # 过滤掉空项目
        sector_items = [item for item in sector_items if item]
        
        if sector_items:
            html += f'<p><strong>📊 板块表现</strong>:<br>'
            for item in sector_items:
                # 确保每项都以分号结尾
                if not item.endswith('；'):
                    item += '；'
                html += f'{item}<br>'
            html += '</p>'
        else:
            html += f'<p><strong>📊 板块表现</strong>: {sector_content}</p>'
    # 4. 风险机会卡片
    html += '</div></div><div class="card"><h3 class="card-title">🔮 风险机会</h3>'
    html += '<div class="ranking-grid">'
    
    # 日涨幅风险机会
    top_vcs = re.search(
        r"日涨幅风险机会[：:]\s*(.*?)(?:\s*日跌幅风险机会[：:]|\s*风险提示[：:])",
        text_data,
        re.DOTALL
    )
    if top_vcs and top_vcs.group(1).strip():
        # 分割每个TOP项目
        top_items = re.split(r'\s*[。\.]TOP\d+\.\s*', top_vcs.group(1).strip())
        # 过滤空项目并确保最多只取前3个
        top_items = [item.strip() for item in top_items if item.strip()][:3]
        if top_items:
            html += '<div class="ranking-box"><div class="ranking-title">📈🤑 日涨幅风险机会</div>'
            for i, item in enumerate(top_items):
                # 重新添加TOP编号，因为我们在分割时去掉了它
                lines = item.split('\n')
                first_line = lines[0] if lines else ""
                html += f'<div class="ranking-item"><strong>TOP{i+1}. {first_line}</strong>'
                # 添加剩余的详细信息
                for line in lines[1:]:
                    if line.strip():
                        html += f'<br>{line.strip()}'
                html += '</div>'
            html += '</div>'
    
    # 日跌幅风险机会
    low_vcs = re.search(
        r"日跌幅风险机会[：:]\s*(.*?)(?:\s*风险提示[：:]|$)",
        text_data,
        re.DOTALL
    )
    if low_vcs and low_vcs.group(1).strip():
        # 分割每个TOP项目
        low_items = re.split(r'\s*[。\.]TOP\d+\.\s*', low_vcs.group(1).strip())
        # 过滤空项目并确保最多只取前3个
        low_items = [item.strip() for item in low_items if item.strip()][:3]
        if low_items:
            html += '<div class="ranking-box"><div class="ranking-title">📉🤑 日跌幅风险机会</div>'
            for i, item in enumerate(low_items):
                # 重新添加TOP编号，因为我们在分割时去掉了它
                lines = item.split('\n')
                first_line = lines[0] if lines else ""
                html += f'<div class="ranking-item"><strong>TOP{i+1}. {first_line}</strong>'
                # 添加剩余的详细信息
                for line in lines[1:]:
                    if line.strip():
                        html += f'<br>{line.strip()}'
                html += '</div>'
            html += '</div>'
    
    html += '</div>'
    
    # 风险提示
    venture = re.search(r"风险提示[：:]\s*(.*?)(?:\s*结束|$)", text_data, re.DOTALL)
    if venture:
        risk_content = venture.group(1).strip()
        # 按分号分割风险提示内容
        risk_items = [item.strip() for item in risk_content.split('；') if item.strip()]
        # 过滤掉空项目
        risk_items = [item for item in risk_items if item]
        
        if risk_items:
            html += '<p><strong>⚠️⚠️ 风险提示</strong>:<br>'
            for item in risk_items:
                # 确保每项都以分号结尾
                if not item.endswith('；'):
                    item += '；'
                html += f'{item}<br>'
            html += '</p>'
        else:
            html += f'<p><strong>⚠️⚠️ 风险提示</strong>: {risk_content}</p>'
    
    # 页脚和收件人信息
    html += f"""
        </div>
        <div class="recipient-note">
            本邮件已发送至 {len(RECIPIENT_LIST)} 位收件人 | 抄送 {len(CC_LIST)} 位联系人
        </div>
        <div class="footer">
            数据来源：玄策光年 | 自动生成时间：{datetime.now().strftime('%m-%d %H:%M')}
        </div>
    </body>
    </html>
    """
    
    return html

def send_email_via_feishu(subject, html_content):
    """飞书邮箱发送函数（支持多收件人和抄送）"""
    # 创建邮件对象
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = ", ".join(RECIPIENT_LIST)  # 所有主要收件人
    msg['Cc'] = ", ".join(CC_LIST)         # 所有抄送收件人
    msg['Subject'] = subject
    
    # 添加HTML内容
    msg.attach(MIMEText(html_content, 'html'))
    
    # 构建收件人列表（包含收件人+抄送人）
    all_recipients = RECIPIENT_LIST + CC_LIST

    try:
        # 连接飞书SMTP服务器
        if SMTP_PORT == 465:
            # SSL加密方式
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.sendmail(EMAIL_ADDRESS, all_recipients, msg.as_string())
        else:
            # STARTTLS加密方式（推荐）
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls(context=context)
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.sendmail(EMAIL_ADDRESS, all_recipients, msg.as_string())

        print(f"邮件发送成功！收件人：{len(RECIPIENT_LIST)}人，抄送：{len(CC_LIST)}人")
        return True
    except Exception as e:
        print(f"邮件发送失败: {str(e)}")
        return False

def extract_briefing_data(file_path):
    """从Excel提取26-40行数据（增强合并单元格处理）"""
    try:
        wb = load_workbook(file_path, data_only=True)
        sheet = wb.active
        
        texts = []
        # 26-40行（索引25-41）
        for row in range(25, 41):
            row_text = ""
            for col in range(1, sheet.max_column + 1):
                cell = sheet.cell(row=row, column=col)
                
                # 处理合并单元格值
                cell_value = cell.value
                for merged_range in sheet.merged_cells.ranges:
                    if cell.coordinate in merged_range:
                        # 使用合并区域的第一个单元格的值
                        cell_value = sheet.cell(
                            row=merged_range.min_row, 
                            column=merged_range.min_col
                        ).value
                        break
                
                # 处理空值
                if cell_value is None:
                    cell_value = ""
                
                # 添加到行文本（用制表符分隔）
                row_text += str(cell_value) + "\t"
            
            texts.append(row_text.strip())
        
        return "\n".join(texts)

    except Exception as e:
        print(f"提取数据错误: {str(e)}")
        return ""

def main():
    """主函数 - 自动化处理流程"""
    # 1. 文件配置（根据实际情况修改）
    # 手动版
    #EXCEL_FILE = r'C:/Users/ASUS/Desktop/日报/玄策光年全天候估值表20250805.xlsx'
    # 自动获取当日日期并拼接文件名
    today_str = datetime.now().strftime('%Y%m%d')
    EXCEL_FILE = fr'C:/Users/ASUS/Desktop/日报/玄策光年全天候估值表{today_str}.xlsx'
    
    # 2. 检查文件是否存在
    if not os.path.exists(EXCEL_FILE):
        print(f"错误：文件未找到 - {EXCEL_FILE}")
        return
    
    # 3. 提取简报内容
    print("正在从Excel提取数据...")
    briefing_text = extract_briefing_data(EXCEL_FILE)
    
    if not briefing_text:
        print("错误：未能从Excel中提取有效数据")
        return
    
    # 4. 创建邮件内容
    print("正在创建邮件内容...")
    today_date = datetime.now().strftime("%Y-%m-%d")
    email_subject = f"【市场简报】{today_date}"
    html_content = create_email_body(briefing_text)
    
    # 5. 发送邮件
    print("正在通过飞书邮箱发送...")
    if send_email_via_feishu(email_subject, html_content):
        print("自动化邮件发送流程完成！")
    else:
        print("邮件发送失败，请检查配置后重试")

if __name__ == "__main__":
    # 添加日志记录
    start_time = datetime.now()
    print(f"=== 程序启动于 {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    main()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"=== 程序结束于 {end_time.strftime('%Y-%m-%d %H:%M:%S')} | 耗时: {duration:.2f}秒 ===")