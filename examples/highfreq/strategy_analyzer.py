"""
策略分析模块
包含：风险指标、归因分析、稳定性分析、交易分析
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class StrategyAnalyzer:
    """策略分析器"""
    
    def __init__(self, report: pd.DataFrame, benchmark_series: Optional[pd.Series] = None, output_suffix: str = ""):
        """
        report: 回测报告 DataFrame（包含 return, bench, cost 等列）
        benchmark_series: benchmark 序列（可选）
        output_suffix: 输出文件后缀
        """
        self.report = report.copy()
        self.benchmark_series = benchmark_series
        self.output_suffix = output_suffix
        
        # 确保必要的列存在
        required_cols = ['return']
        missing_cols = [col for col in required_cols if col not in self.report.columns]
        if missing_cols:
            raise ValueError(f"报告缺少必要的列: {missing_cols}")
    
    def calculate_risk_metrics(self) -> pd.DataFrame:
        """计算风险指标"""
        print("\n📊 计算风险指标...")
        
        metrics = {}
        
        # 计算收益序列
        if 'bench' in self.report.columns:
            excess_return = self.report['return'] - self.report['bench']
        else:
            excess_return = self.report['return']
        
        if 'cost' in self.report.columns:
            excess_return_w_cost = excess_return - self.report['cost']
        else:
            excess_return_w_cost = excess_return
        
        # 计算累计收益
        cum_return = (1 + self.report['return']).cumprod() - 1
        cum_excess = (1 + excess_return).cumprod() - 1
        cum_excess_w_cost = (1 + excess_return_w_cost).cumprod() - 1
        
        # 1. 年化收益率
        if len(cum_return) > 0:
            total_return = cum_return.iloc[-1]
            trading_days = len(self.report)
            if trading_days > 0:
                annual_return = (1 + total_return) ** (252 / trading_days) - 1
                metrics['年化收益率'] = annual_return
                
                annual_excess = (1 + cum_excess.iloc[-1]) ** (252 / trading_days) - 1
                metrics['年化超额收益（无成本）'] = annual_excess
                
                annual_excess_w_cost = (1 + cum_excess_w_cost.iloc[-1]) ** (252 / trading_days) - 1
                metrics['年化超额收益（含成本）'] = annual_excess_w_cost
        
        # 2. 波动率（年化）
        if len(excess_return) > 1:
            volatility = excess_return.std() * np.sqrt(252)
            metrics['波动率（年化）'] = volatility
            
            volatility_w_cost = excess_return_w_cost.std() * np.sqrt(252)
            metrics['波动率（年化，含成本）'] = volatility_w_cost
        
        # 3. 最大回撤
        if len(cum_return) > 0:
            running_max = cum_return.expanding().max()
            drawdown = cum_return - running_max
            max_drawdown = drawdown.min()
            metrics['最大回撤'] = max_drawdown
            
            # 超额收益的最大回撤
            running_max_excess = cum_excess.expanding().max()
            drawdown_excess = cum_excess - running_max_excess
            max_drawdown_excess = drawdown_excess.min()
            metrics['最大回撤（超额收益）'] = max_drawdown_excess
        
        # 4. 夏普比率
        if '年化超额收益（无成本）' in metrics and '波动率（年化）' in metrics:
            if metrics['波动率（年化）'] > 0:
                sharpe = metrics['年化超额收益（无成本）'] / metrics['波动率（年化）']
                metrics['夏普比率'] = sharpe
        
        # 5. 索提诺比率（只考虑下行波动）
        if len(excess_return) > 1:
            downside_returns = excess_return[excess_return < 0]
            if len(downside_returns) > 1:
                downside_std = downside_returns.std() * np.sqrt(252)
                if downside_std > 0 and '年化超额收益（无成本）' in metrics:
                    sortino = metrics['年化超额收益（无成本）'] / downside_std
                    metrics['索提诺比率'] = sortino
        
        # 6. Calmar比率（年化收益/最大回撤）
        if '年化收益率' in metrics and '最大回撤' in metrics:
            if abs(metrics['最大回撤']) > 1e-10:
                calmar = metrics['年化收益率'] / abs(metrics['最大回撤'])
                metrics['Calmar比率'] = calmar
        
        # 7. VaR (Value at Risk) - 95%置信度
        if len(excess_return) > 0:
            var_95 = excess_return.quantile(0.05)
            metrics['VaR (95%)'] = var_95
        
        # 8. CVaR (Conditional VaR) - 95%置信度
        if len(excess_return) > 0:
            cvar_95 = excess_return[excess_return <= var_95].mean()
            metrics['CVaR (95%)'] = cvar_95
        
        # 9. 胜率
        if len(excess_return) > 0:
            win_rate = (excess_return > 0).sum() / len(excess_return)
            metrics['胜率'] = win_rate
        
        # 10. 盈亏比
        if len(excess_return) > 0:
            positive_returns = excess_return[excess_return > 0]
            negative_returns = excess_return[excess_return < 0]
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                avg_win = positive_returns.mean()
                avg_loss = abs(negative_returns.mean())
                if avg_loss > 0:
                    profit_loss_ratio = avg_win / avg_loss
                    metrics['盈亏比'] = profit_loss_ratio
        
        # 转换为DataFrame
        metrics_df = pd.DataFrame({
            '指标': list(metrics.keys()),
            '数值': list(metrics.values())
        })
        
        return metrics_df
    
    def calculate_stability_metrics(self, window_days: int = 30) -> pd.DataFrame:
        """稳定性分析：滚动窗口分析"""
        print(f"\n📊 计算稳定性指标（滚动窗口: {window_days}天）...")
        
        if 'bench' in self.report.columns:
            excess_return = self.report['return'] - self.report['bench']
        else:
            excess_return = self.report['return']
        
        if len(excess_return) < window_days:
            print(f"   ⚠️ 数据不足，无法计算{window_days}天滚动窗口")
            return pd.DataFrame()
        
        # 滚动窗口统计
        rolling_stats = []
        
        for i in range(window_days, len(excess_return) + 1):
            window_returns = excess_return.iloc[i-window_days:i]
            
            if len(window_returns) > 0:
                # 计算窗口内的统计指标
                window_mean = window_returns.mean()
                window_std = window_returns.std()
                window_sharpe = window_mean / window_std * np.sqrt(252) if window_std > 0 else 0
                window_win_rate = (window_returns > 0).sum() / len(window_returns)
                
                # 累计收益
                window_cum = (1 + window_returns).cumprod().iloc[-1] - 1
                
                rolling_stats.append({
                    'date': excess_return.index[i-1],
                    '窗口收益': window_cum,
                    '窗口均值': window_mean,
                    '窗口波动率': window_std * np.sqrt(252),
                    '窗口夏普': window_sharpe,
                    '窗口胜率': window_win_rate,
                })
        
        if rolling_stats:
            stats_df = pd.DataFrame(rolling_stats)
            stats_df.set_index('date', inplace=True)
            
            # 计算整体统计
            summary = {
                '指标': [
                    '滚动收益均值',
                    '滚动收益标准差',
                    '滚动夏普均值',
                    '滚动夏普标准差',
                    '滚动胜率均值',
                ],
                '数值': [
                    stats_df['窗口收益'].mean(),
                    stats_df['窗口收益'].std(),
                    stats_df['窗口夏普'].mean(),
                    stats_df['窗口夏普'].std(),
                    stats_df['窗口胜率'].mean(),
                ]
            }
            
            return pd.DataFrame(summary)
        else:
            return pd.DataFrame()
    
    def calculate_attribution(self) -> pd.DataFrame:
        """归因分析：时间归因（月度/季度）"""
        print("\n📊 计算归因分析（时间归因）...")
        
        if 'bench' in self.report.columns:
            excess_return = self.report['return'] - self.report['bench']
        else:
            excess_return = self.report['return']
        
        # 提取日期
        if isinstance(excess_return.index, pd.DatetimeIndex):
            dates = excess_return.index
        else:
            dates = pd.to_datetime(excess_return.index)
        
        # 按月分组
        monthly_returns = excess_return.groupby([dates.year, dates.month]).sum()
        monthly_returns.index = pd.MultiIndex.from_tuples(monthly_returns.index, names=['year', 'month'])
        
        # 按季度分组
        quarterly_returns = excess_return.groupby([dates.year, dates.quarter]).sum()
        quarterly_returns.index = pd.MultiIndex.from_tuples(quarterly_returns.index, names=['year', 'quarter'])
        
        # 构建归因结果
        attribution_data = []
        
        # 月度归因
        for (year, month), ret in monthly_returns.items():
            attribution_data.append({
                '时间': f"{year}-{month:02d}",
                '类型': '月度',
                '收益': ret,
                '累计收益': (1 + monthly_returns.loc[year, :month].sum()) - 1 if month > 1 else ret,
            })
        
        # 季度归因
        for (year, quarter), ret in quarterly_returns.items():
            attribution_data.append({
                '时间': f"{year}Q{quarter}",
                '类型': '季度',
                '收益': ret,
                '累计收益': (1 + quarterly_returns.loc[year, :quarter].sum()) - 1 if quarter > 1 else ret,
            })
        
        if attribution_data:
            attribution_df = pd.DataFrame(attribution_data)
            return attribution_df
        else:
            return pd.DataFrame()
    
    def calculate_trade_metrics(self) -> pd.DataFrame:
        """交易分析（基础统计）"""
        print("\n📊 计算交易分析...")
        
        if 'cost' in self.report.columns:
            cost_series = self.report['cost']
            total_cost = cost_series.sum()
            avg_cost = cost_series.mean()
            cost_ratio = total_cost / abs(self.report['return'].sum()) if abs(self.report['return'].sum()) > 0 else 0
        else:
            total_cost = 0
            avg_cost = 0
            cost_ratio = 0
        
        if 'bench' in self.report.columns:
            excess_return = self.report['return'] - self.report['bench']
        else:
            excess_return = self.report['return']
        
        # 交易统计
        trade_metrics = {
            '总交易成本': total_cost,
            '平均交易成本': avg_cost,
            '成本占比': cost_ratio,
            '交易天数': len(self.report),
            '盈利天数': (excess_return > 0).sum(),
            '亏损天数': (excess_return < 0).sum(),
            '平均日收益': excess_return.mean(),
            '平均日收益（含成本）': (excess_return - self.report.get('cost', 0)).mean() if 'cost' in self.report.columns else excess_return.mean(),
        }
        
        trade_df = pd.DataFrame({
            '指标': list(trade_metrics.keys()),
            '数值': list(trade_metrics.values())
        })
        
        return trade_df
    
    def generate_full_report(self, output_dir: str = ".") -> None:
        """生成完整分析报告（文本格式）"""
        print("\n📄 生成完整分析报告...")
        
        output_path = Path(output_dir) / f"strategy_analysis_report{self.output_suffix}.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("策略分析报告\n")
            f.write("="*70 + "\n\n")
            
            # 风险指标
            f.write("一、风险指标\n")
            f.write("-"*70 + "\n")
            risk_metrics = self.calculate_risk_metrics()
            if not risk_metrics.empty:
                f.write(risk_metrics.to_string(index=False))
                f.write("\n\n")
            
            # 稳定性分析
            f.write("二、稳定性分析\n")
            f.write("-"*70 + "\n")
            stability_metrics = self.calculate_stability_metrics()
            if not stability_metrics.empty:
                f.write(stability_metrics.to_string(index=False))
                f.write("\n\n")
            
            # 归因分析
            f.write("三、归因分析（时间归因）\n")
            f.write("-"*70 + "\n")
            attribution = self.calculate_attribution()
            if not attribution.empty:
                f.write(attribution.to_string(index=False))
                f.write("\n\n")
            
            # 交易分析
            f.write("四、交易分析\n")
            f.write("-"*70 + "\n")
            trade_metrics = self.calculate_trade_metrics()
            if not trade_metrics.empty:
                f.write(trade_metrics.to_string(index=False))
                f.write("\n\n")
        
        print(f"✅ 完整分析报告已保存: {output_path}")

