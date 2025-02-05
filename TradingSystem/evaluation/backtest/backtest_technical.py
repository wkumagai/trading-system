"""
backtest_technical.py

テクニカル分析戦略のバックテスト
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List

from Strategies.Technical.moving_average import SimpleMAStrategy, TripleMAStrategy
from Strategies.Technical.momentum import RSIStrategy, MACDStrategy

def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    株価データの取得
    Args:
        symbol: 銘柄コード
        start_date: 開始日
        end_date: 終了日
    Returns:
        株価データ
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    df.columns = [col.lower() for col in df.columns]
    return df

def calculate_returns(data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """
    リターンの計算
    Args:
        data: 株価データ
        signals: 取引シグナル
    Returns:
        リターン計算結果
    """
    df = signals.copy()
    df['returns'] = data['close'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    return df

def calculate_metrics(returns: pd.DataFrame) -> Dict:
    """
    パフォーマンス指標の計算
    Args:
        returns: リターン計算結果
    Returns:
        パフォーマンス指標
    """
    strategy_returns = returns['strategy_returns'].dropna()
    
    # 年率リターン
    annual_return = strategy_returns.mean() * 252
    
    # シャープレシオ
    sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    # 最大ドローダウン
    cumulative = returns['cumulative_returns']
    rolling_max = cumulative.expanding().max()
    drawdowns = cumulative / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    # 勝率
    winning_trades = (strategy_returns > 0).sum()
    total_trades = (returns['signal'] != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    return {
        'Annual Return': annual_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Total Trades': total_trades
    }

def run_backtest(symbol: str = 'NVDA', 
                start_date: str = '2023-01-01',
                end_date: str = '2024-12-31'):
    """
    バックテストの実行
    Args:
        symbol: 銘柄コード
        start_date: 開始日
        end_date: 終了日
    """
    # データの取得
    print(f"Fetching data for {symbol}...")
    data = fetch_data(symbol, start_date, end_date)
    
    # 戦略の初期化
    strategies = {
        'Simple MA': SimpleMAStrategy(short_window=5, long_window=20),
        'Triple MA': TripleMAStrategy(short_window=5, mid_window=20, long_window=50),
        'RSI': RSIStrategy(period=14, overbought=70, oversold=30),
        'MACD': MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
    }
    
    results = {}
    
    # 各戦略のバックテスト
    for name, strategy in strategies.items():
        print(f"\nBacktesting {name} strategy...")
        
        # シグナルの生成
        signals = strategy.generate_signals(data)[symbol if isinstance(data, dict) else 'default']
        
        # リターンの計算
        returns = calculate_returns(data, signals)
        
        # メトリクスの計算
        metrics = calculate_metrics(returns)
        
        results[name] = {
            'returns': returns,
            'metrics': metrics
        }
        
        # 結果の表示
        print(f"\n{name} Strategy Results:")
        print("-" * 50)
        print(f"Annual Return: {metrics['Annual Return']:.2%}")
        print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {metrics['Max Drawdown']:.2%}")
        print(f"Win Rate: {metrics['Win Rate']:.2%}")
        print(f"Total Trades: {metrics['Total Trades']}")
    
    return results

if __name__ == "__main__":
    # バックテストの実行
    results = run_backtest(
        symbol='NVDA',
        start_date='2023-01-01',
        end_date='2024-12-31'
    )