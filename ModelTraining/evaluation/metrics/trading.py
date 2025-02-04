"""
trading.py

取引パフォーマンスに関する評価指標を計算するモジュール。
リターン、リスク、その他の取引関連指標を計算。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from scipy import stats

class TradingMetrics:
    """取引パフォーマンスの評価指標を計算するクラス"""
    
    @staticmethod
    def calculate_returns_metrics(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        trading_days: int = 252
    ) -> Dict[str, float]:
        """
        リターンに関する指標を計算

        Args:
            returns: リターンの配列
            risk_free_rate: 無リスク金利（年率）
            trading_days: 年間取引日数

        Returns:
            評価指標の辞書
        """
        # 年率換算係数
        annualization_factor = np.sqrt(trading_days)
        
        # 累積リターン
        cumulative_return = (1 + returns).prod() - 1
        
        # 年率リターン
        annual_return = (1 + cumulative_return) ** (trading_days / len(returns)) - 1
        
        # ボラティリティ（年率）
        volatility = returns.std() * annualization_factor
        
        # シャープレシオ
        excess_returns = returns - risk_free_rate / trading_days
        sharpe_ratio = np.sqrt(trading_days) * np.mean(excess_returns) / np.std(excess_returns)
        
        # ソルティノレシオ
        downside_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(trading_days) * np.mean(excess_returns) / np.std(downside_returns)
        
        return {
            'cumulative_return': cumulative_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio
        }

    @staticmethod
    def calculate_risk_metrics(
        returns: np.ndarray,
        prices: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        リスクに関する指標を計算

        Args:
            returns: リターンの配列
            prices: 価格の配列（オプション）

        Returns:
            評価指標の辞書
        """
        # 最大ドローダウン
        if prices is not None:
            cummax = np.maximum.accumulate(prices)
            drawdown = (prices - cummax) / cummax
            max_drawdown = np.min(drawdown)
        else:
            cum_returns = (1 + returns).cumprod()
            cummax = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - cummax) / cummax
            max_drawdown = np.min(drawdown)
        
        # バリューアットリスク（VaR）
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # 条件付きバリューアットリスク（CVaR）
        cvar_95 = np.mean(returns[returns <= var_95])
        cvar_99 = np.mean(returns[returns <= var_99])
        
        # 歪度（分布の非対称性）
        skewness = stats.skew(returns)
        
        # 尖度（分布の裾の重さ）
        kurtosis = stats.kurtosis(returns)
        
        return {
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    @staticmethod
    def calculate_trading_efficiency(
        returns: np.ndarray,
        positions: np.ndarray
    ) -> Dict[str, float]:
        """
        取引効率に関する指標を計算

        Args:
            returns: リターンの配列
            positions: ポジションの配列（1: 買い、-1: 売り、0: なし）

        Returns:
            評価指標の辞書
        """
        # 取引回数
        trades = np.diff(positions) != 0
        trade_count = np.sum(trades)
        
        # 勝率
        trade_returns = returns[trades]
        win_rate = np.mean(trade_returns > 0)
        
        # 平均利益率と平均損失率
        profit_trades = trade_returns[trade_returns > 0]
        loss_trades = trade_returns[trade_returns < 0]
        
        avg_profit = np.mean(profit_trades) if len(profit_trades) > 0 else 0
        avg_loss = np.mean(loss_trades) if len(loss_trades) > 0 else 0
        
        # プロフィットファクター
        total_profit = np.sum(profit_trades) if len(profit_trades) > 0 else 0
        total_loss = abs(np.sum(loss_trades)) if len(loss_trades) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss != 0 else np.inf
        
        return {
            'trade_count': trade_count,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }

    @staticmethod
    def calculate_position_metrics(
        positions: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        ポジションに関する指標を計算

        Args:
            positions: ポジションの配列
            returns: リターンの配列

        Returns:
            評価指標の辞書
        """
        # ポジション保有期間の分析
        position_lengths = []
        current_length = 0
        
        for i in range(1, len(positions)):
            if positions[i] == positions[i-1] and positions[i] != 0:
                current_length += 1
            elif positions[i] != 0:
                if current_length > 0:
                    position_lengths.append(current_length)
                current_length = 1
            else:
                if current_length > 0:
                    position_lengths.append(current_length)
                current_length = 0
        
        position_lengths = np.array(position_lengths)
        
        # ポジションごとのリターン
        long_returns = returns[positions == 1]
        short_returns = returns[positions == -1]
        
        metrics = {
            'avg_position_length': np.mean(position_lengths),
            'max_position_length': np.max(position_lengths),
            'long_exposure': np.mean(positions == 1),
            'short_exposure': np.mean(positions == -1),
            'long_return': np.mean(long_returns) if len(long_returns) > 0 else 0,
            'short_return': np.mean(short_returns) if len(short_returns) > 0 else 0
        }
        
        return metrics

if __name__ == "__main__":
    # 使用例
    # サンプルデータの生成
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)
    prices = 100 * (1 + returns).cumprod()
    positions = np.random.choice([-1, 0, 1], 1000)
    
    # リターン指標の計算
    returns_metrics = TradingMetrics.calculate_returns_metrics(returns)
    print("\nReturns Metrics:")
    for metric, value in returns_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # リスク指標の計算
    risk_metrics = TradingMetrics.calculate_risk_metrics(returns, prices)
    print("\nRisk Metrics:")
    for metric, value in risk_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 取引効率の計算
    efficiency_metrics = TradingMetrics.calculate_trading_efficiency(returns, positions)
    print("\nEfficiency Metrics:")
    for metric, value in efficiency_metrics.items():
        print(f"{metric}: {value:.4f}")