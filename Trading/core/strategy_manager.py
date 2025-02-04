"""
strategy_manager.py

複数の取引戦略を管理し、実行・評価を行うモジュール。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

class StrategyManager:
    """
    複数の取引戦略を管理し、実行・評価を行うクラス。
    """

    def __init__(self, config):
        """
        Args:
            config: システム全体の設定値を持つモジュール
        """
        self.config = config
        self.strategies = {}
        self.results = {}
        self.logger = logging.getLogger(__name__)

    def register_strategy(self, strategy_instance):
        """
        新しい戦略を登録する。

        Args:
            strategy_instance: BaseStrategyを継承した戦略インスタンス
        """
        strategy_name = strategy_instance.strategy_name
        self.strategies[strategy_name] = strategy_instance
        self.logger.info(f"Registered strategy: {strategy_name}")

    def register_multiple_strategies(self, strategy_instances):
        """
        複数の戦略を一括登録する。

        Args:
            strategy_instances: 戦略インスタンスのリスト
        """
        for strategy in strategy_instances:
            self.register_strategy(strategy)

    def _run_single_strategy(self, strategy_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        単一の戦略を実行する。

        Args:
            strategy_name: 実行する戦略の名前
            df: 入力データ

        Returns:
            実行結果を含むディクショナリ
        """
        try:
            strategy = self.strategies[strategy_name]
            
            # 特徴量生成
            self.logger.info(f"Creating features for {strategy_name}")
            feat_df = strategy.create_features(df)
            
            # モデル学習
            self.logger.info(f"Training model for {strategy_name}")
            strategy.train_model(feat_df)
            
            # 予測実行
            self.logger.info(f"Making predictions for {strategy_name}")
            pred_df = strategy.predict(feat_df)
            
            return {
                'strategy_name': strategy_name,
                'predictions': pred_df,
                'parameters': strategy.get_parameters()
            }
            
        except Exception as e:
            self.logger.error(f"Error running strategy {strategy_name}: {str(e)}")
            return {
                'strategy_name': strategy_name,
                'error': str(e)
            }

    def run_all_strategies(self, df: pd.DataFrame, parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        登録された全戦略を実行する。

        Args:
            df: 入力データ
            parallel: 並列実行を行うかどうか

        Returns:
            各戦略の実行結果を含むディクショナリ
        """
        self.results = {}
        
        if parallel and len(self.strategies) > 1:
            # 並列実行
            with ThreadPoolExecutor() as executor:
                future_to_strategy = {
                    executor.submit(self._run_single_strategy, name, df.copy()): name
                    for name in self.strategies.keys()
                }
                
                for future in as_completed(future_to_strategy):
                    result = future.result()
                    if 'error' not in result:
                        self.results[result['strategy_name']] = result
        else:
            # 逐次実行
            for strategy_name in self.strategies.keys():
                result = self._run_single_strategy(strategy_name, df.copy())
                if 'error' not in result:
                    self.results[strategy_name] = result

        return self.results

    def get_ensemble_predictions(self, weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        全戦略の予測を組み合わせたアンサンブル予測を生成する。

        Args:
            weights: 各戦略の重み付けディクショナリ
                    指定がない場合は均等重み

        Returns:
            アンサンブル予測を含むDataFrame
        """
        if not self.results:
            raise ValueError("No strategy results available. Run strategies first.")

        # 重みの準備
        if weights is None:
            weights = {name: 1.0/len(self.results) for name in self.results.keys()}

        # 各戦略の予測を重み付けして合算
        ensemble_predictions = None
        for strategy_name, result in self.results.items():
            if 'predictions' not in result:
                continue
                
            pred_df = result['predictions']
            weighted_signal = pred_df['signal'] * weights[strategy_name]
            
            if ensemble_predictions is None:
                ensemble_predictions = pred_df.copy()
                ensemble_predictions['signal'] = weighted_signal
            else:
                ensemble_predictions['signal'] += weighted_signal

        if ensemble_predictions is not None:
            # シグナルの閾値処理
            ensemble_predictions['signal'] = np.sign(ensemble_predictions['signal'])

        return ensemble_predictions

    def get_strategy_performance(self) -> pd.DataFrame:
        """
        各戦略のパフォーマンス指標を計算する。

        Returns:
            パフォーマンス指標を含むDataFrame
        """
        performance_data = []
        
        for strategy_name, result in self.results.items():
            if 'predictions' not in result:
                continue
                
            pred_df = result['predictions']
            signals = pred_df['signal']
            returns = pred_df['Close'].pct_change()
            
            # 戦略リターンの計算
            strategy_returns = signals.shift(1) * returns
            
            performance = {
                'strategy_name': strategy_name,
                'total_return': strategy_returns.sum(),
                'sharpe_ratio': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252),
                'win_rate': (strategy_returns > 0).mean(),
                'max_drawdown': (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()
            }
            
            performance_data.append(performance)
        
        return pd.DataFrame(performance_data)

def run_strategy_comparison(symbol: str, start_date: str, end_date: str, config) -> None:
    """
    指定された期間で複数の戦略を実行し、比較する。

    Args:
        symbol: 対象銘柄
        start_date: 開始日
        end_date: 終了日
        config: 設定オブジェクト
    """
    from ..strategies.moving_average import MovingAverageStrategy
    from ..strategies.deep_learning import LSTMStrategy
    
    # データ管理モジュールを使用
    from .data_manager import DataManager
    dm = DataManager(config)
    df = dm.fetch_market_data(symbol, start_date, end_date, interval="1min")
    
    if df.empty:
        logging.error("No data fetched. Exiting comparison.")
        return
    
    # 戦略マネージャーの設定
    manager = StrategyManager(config)
    
    # 複数の戦略を登録
    strategies = [
        MovingAverageStrategy(config, short_window=5, long_window=20),
        MovingAverageStrategy(config, short_window=10, long_window=30),
        LSTMStrategy(config, sequence_length=10),
        LSTMStrategy(config, sequence_length=20)
    ]
    manager.register_multiple_strategies(strategies)
    
    # 全戦略を実行
    manager.run_all_strategies(df)
    
    # パフォーマンス評価
    performance_df = manager.get_strategy_performance()
    print("\nStrategy Performance:")
    print(performance_df)
    
    # アンサンブル予測の生成
    ensemble_df = manager.get_ensemble_predictions()
    if ensemble_df is not None:
        print("\nEnsemble Strategy Generated")