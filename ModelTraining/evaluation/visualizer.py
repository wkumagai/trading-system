"""
visualizer.py

評価結果の可視化を行うモジュール。
パフォーマンス指標、取引結果、モデルの予測等を
グラフィカルに表示する。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PerformanceVisualizer:
    """パフォーマンス可視化クラス"""
    
    def __init__(self, style: str = 'seaborn'):
        """
        Args:
            style: プロットのスタイル
        """
        plt.style.use(style)
        self.colors = sns.color_palette()

    def plot_cumulative_returns(
        self,
        returns: Dict[str, pd.Series],
        title: str = "Cumulative Returns",
        save_path: Optional[str] = None
    ) -> None:
        """
        累積リターンのプロット

        Args:
            returns: 戦略ごとのリターン系列
            title: グラフのタイトル
            save_path: 保存先パス
        """
        plt.figure(figsize=(12, 6))
        
        for name, ret in returns.items():
            cum_returns = (1 + ret).cumprod()
            plt.plot(cum_returns.index, cum_returns.values, label=name)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_drawdown(
        self,
        returns: Dict[str, pd.Series],
        title: str = "Drawdown Analysis",
        save_path: Optional[str] = None
    ) -> None:
        """
        ドローダウンのプロット

        Args:
            returns: 戦略ごとのリターン系列
            title: グラフのタイトル
            save_path: 保存先パス
        """
        plt.figure(figsize=(12, 6))
        
        for name, ret in returns.items():
            cum_returns = (1 + ret).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / running_max
            plt.plot(drawdown.index, drawdown.values, label=name)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_monthly_returns_heatmap(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns Heatmap",
        save_path: Optional[str] = None
    ) -> None:
        """
        月次リターンのヒートマップ

        Args:
            returns: リターン系列
            title: グラフのタイトル
            save_path: 保存先パス
        """
        monthly_returns = returns.resample('M').agg(lambda x: (1 + x).prod() - 1)
        monthly_returns = monthly_returns.to_frame()
        monthly_returns.index = pd.MultiIndex.from_arrays([
            monthly_returns.index.year,
            monthly_returns.index.month
        ])
        monthly_returns = monthly_returns.unstack()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            monthly_returns,
            annot=True,
            fmt='.2%',
            center=0,
            cmap='RdYlGn',
            cbar_kws={'label': 'Monthly Return'}
        )
        
        plt.title(title)
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 252,
        metrics: List[str] = ['return', 'volatility', 'sharpe'],
        title: str = "Rolling Metrics",
        save_path: Optional[str] = None
    ) -> None:
        """
        移動平均ベースの指標プロット

        Args:
            returns: リターン系列
            window: 移動平均の窓サイズ
            metrics: 表示する指標
            title: グラフのタイトル
            save_path: 保存先パス
        """
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            if metric == 'return':
                rolling_metric = returns.rolling(window).mean() * 252
                label = f'Rolling Annual Return ({window}d)'
            elif metric == 'volatility':
                rolling_metric = returns.rolling(window).std() * np.sqrt(252)
                label = f'Rolling Annual Volatility ({window}d)'
            elif metric == 'sharpe':
                rolling_return = returns.rolling(window).mean() * 252
                rolling_vol = returns.rolling(window).std() * np.sqrt(252)
                rolling_metric = rolling_return / rolling_vol
                label = f'Rolling Sharpe Ratio ({window}d)'
            
            ax.plot(rolling_metric.index, rolling_metric.values)
            ax.set_title(label)
            ax.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def create_interactive_dashboard(
        self,
        returns: pd.Series,
        positions: pd.Series,
        prices: pd.Series,
        save_path: Optional[str] = None
    ) -> None:
        """
        インタラクティブなダッシュボードの作成

        Args:
            returns: リターン系列
            positions: ポジション系列
            prices: 価格系列
            save_path: 保存先パス
        """
        # サブプロットの作成
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                'Price and Positions',
                'Cumulative Returns',
                'Drawdown'
            )
        )
        
        # 価格とポジション
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices.values,
                name='Price',
                line=dict(color='blue')
            ),
            row=1,
            col=1
        )
        
        # ポジションの可視化
        long_pos = positions[positions == 1].index
        short_pos = positions[positions == -1].index
        
        fig.add_trace(
            go.Scatter(
                x=long_pos,
                y=prices[long_pos],
                name='Long',
                mode='markers',
                marker=dict(color='green', symbol='triangle-up')
            ),
            row=1,
            col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=short_pos,
                y=prices[short_pos],
                name='Short',
                mode='markers',
                marker=dict(color='red', symbol='triangle-down')
            ),
            row=1,
            col=1
        )
        
        # 累積リターン
        cum_returns = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=cum_returns.values,
                name='Cumulative Return',
                line=dict(color='green')
            ),
            row=2,
            col=1
        )
        
        # ドローダウン
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                name='Drawdown',
                line=dict(color='red')
            ),
            row=3,
            col=1
        )
        
        # レイアウトの調整
        fig.update_layout(
            height=900,
            title_text="Trading Performance Dashboard",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        fig.show()

if __name__ == "__main__":
    # 使用例
    # サンプルデータの生成
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    returns = pd.Series(
        np.random.normal(0.001, 0.02, len(dates)),
        index=dates
    )
    prices = 100 * (1 + returns).cumprod()
    positions = pd.Series(
        np.random.choice([-1, 0, 1], len(dates)),
        index=dates
    )
    
    # 可視化の実行
    visualizer = PerformanceVisualizer()
    
    # 累積リターンのプロット
    visualizer.plot_cumulative_returns(
        {'Strategy': returns},
        title="Strategy Cumulative Returns"
    )
    
    # 月次リターンヒートマップ
    visualizer.plot_monthly_returns_heatmap(
        returns,
        title="Monthly Returns Analysis"
    )
    
    # インタラクティブダッシュボード
    visualizer.create_interactive_dashboard(
        returns,
        positions,
        prices,
        save_path="dashboard.html"
    )