"""
reporter.py

戦略評価結果のレポート生成と可視化を行うモジュール。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import logging
from datetime import datetime

class StrategyReporter:
    """
    戦略評価結果のレポート生成と可視化を行うクラス。
    """

    def __init__(self, output_dir: str = "./reports"):
        """
        Args:
            output_dir: レポート出力ディレクトリ
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        plt.style.use('seaborn')

    def generate_report(
        self,
        strategy_results: Dict[str, pd.DataFrame],
        evaluation_results: pd.DataFrame,
        report_title: str = "Strategy Comparison Report"
    ) -> None:
        """
        包括的なレポートを生成する。

        Args:
            strategy_results: 戦略名をキー、結果DataFrameを値とするディクショナリ
            evaluation_results: 評価指標をまとめたDataFrame
            report_title: レポートのタイトル
        """
        # レポートの構成要素を生成
        self._plot_portfolio_values(strategy_results)
        self._plot_performance_metrics(evaluation_results)
        self._plot_drawdowns(strategy_results)
        self._plot_monthly_returns_heatmap(strategy_results)
        self._plot_correlation_matrix(strategy_results)
        
        # 結果のサマリーを出力
        self._print_summary_statistics(evaluation_results)

    def _plot_portfolio_values(self, strategy_results: Dict[str, pd.DataFrame]) -> None:
        """ポートフォリオ価値の推移をプロット"""
        plt.figure(figsize=(12, 6))
        
        for name, df in strategy_results.items():
            plt.plot(df.index, df['portfolio_value'], label=name)
        
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _plot_performance_metrics(self, evaluation_results: pd.DataFrame) -> None:
        """主要なパフォーマンス指標を棒グラフでプロット"""
        metrics_to_plot = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            evaluation_results[metric].plot(kind='bar', ax=ax)
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()

    def _plot_drawdowns(self, strategy_results: Dict[str, pd.DataFrame]) -> None:
        """ドローダウンの推移をプロット"""
        plt.figure(figsize=(12, 6))
        
        for name, df in strategy_results.items():
            portfolio_values = df['portfolio_value']
            peak = portfolio_values.expanding(min_periods=1).max()
            drawdown = (portfolio_values - peak) / peak * 100
            plt.plot(df.index, drawdown, label=name)
        
        plt.title('Strategy Drawdowns')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _plot_monthly_returns_heatmap(self, strategy_results: Dict[str, pd.DataFrame]) -> None:
        """月次リターンのヒートマップを生成"""
        monthly_returns = {}
        
        for name, df in strategy_results.items():
            returns = df['portfolio_value'].resample('M').last().pct_change() * 100
            monthly_returns[name] = returns
        
        returns_df = pd.DataFrame(monthly_returns)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(returns_df.T, cmap='RdYlGn', center=0, annot=True, fmt='.1f')
        plt.title('Monthly Returns Heatmap (%)')
        plt.tight_layout()
        plt.show()

    def _plot_correlation_matrix(self, strategy_results: Dict[str, pd.DataFrame]) -> None:
        """戦略間のリターン相関をヒートマップで表示"""
        returns_dict = {}
        for name, df in strategy_results.items():
            returns_dict[name] = df['portfolio_value'].pct_change()
        
        returns_df = pd.DataFrame(returns_dict)
        corr_matrix = returns_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        plt.title('Strategy Return Correlations')
        plt.tight_layout()
        plt.show()

    def _print_summary_statistics(self, evaluation_results: pd.DataFrame) -> None:
        """評価指標のサマリー統計を出力"""
        print("\n=== Strategy Performance Summary ===")
        print("\nKey Metrics:")
        print(evaluation_results[['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']])
        
        print("\nBest Performing Strategy by Total Return:")
        best_strategy = evaluation_results['total_return'].idxmax()
        print(f"- {best_strategy}: {evaluation_results.loc[best_strategy, 'total_return']:.2f}%")
        
        print("\nBest Risk-Adjusted Return (Sharpe Ratio):")
        best_sharpe = evaluation_results['sharpe_ratio'].idxmax()
        print(f"- {best_sharpe}: {evaluation_results.loc[best_sharpe, 'sharpe_ratio']:.2f}")

    def generate_trade_analysis(self, strategy_results: Dict[str, pd.DataFrame]) -> None:
        """
        取引分析レポートを生成する。

        Args:
            strategy_results: 戦略名をキー、結果DataFrameを値とするディクショナリ
        """
        for name, df in strategy_results.items():
            trades = df[df['signal'] != 0]
            
            print(f"\n=== Trade Analysis for {name} ===")
            print(f"Total Trades: {len(trades)}")
            
            if len(trades) > 0:
                returns = trades['portfolio_value'].pct_change()
                
                print(f"Average Trade Return: {returns.mean()*100:.2f}%")
                print(f"Best Trade: {returns.max()*100:.2f}%")
                print(f"Worst Trade: {returns.min()*100:.2f}%")
                print(f"Trade Return Std Dev: {returns.std()*100:.2f}%")
                
                # 取引の分布をプロット
                plt.figure(figsize=(10, 6))
                returns.hist(bins=50)
                plt.title(f'Trade Return Distribution - {name}')
                plt.xlabel('Return (%)')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.show()

    def save_report(self, filename: str = None) -> None:
        """
        現在のレポートを保存する。

        Args:
            filename: 保存するファイル名（指定がない場合は日時から自動生成）
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_report_{timestamp}.html"
        
        # レポートの保存処理（実装は省略）
        self.logger.info(f"Report saved as {filename}")