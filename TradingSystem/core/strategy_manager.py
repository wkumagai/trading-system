"""
strategy_manager.py

戦略の管理と実行を行うクラス
"""

from typing import Dict, Type, Optional
from Strategies.Base.base_strategy import BaseStrategy
from Strategies.Technical.moving_average import SimpleMAStrategy, TripleMAStrategy
from Strategies.Technical.momentum import RSIStrategy, MACDStrategy
from Strategies.ML.deep_learning import DeepLearningStrategy

class StrategyManager:
    """戦略管理クラス"""
    
    def __init__(self):
        """初期化"""
        self.strategies: Dict[str, BaseStrategy] = {}
        self._available_strategies = {
            # テクニカル分析戦略
            'simple_ma': SimpleMAStrategy,
            'triple_ma': TripleMAStrategy,
            'rsi': RSIStrategy,
            'macd': MACDStrategy,
            # 機械学習戦略
            'deep_learning': DeepLearningStrategy
        }
    
    def register_strategy(self, 
                        strategy_id: str, 
                        strategy_class: Optional[Type[BaseStrategy]] = None, 
                        **params) -> None:
        """
        戦略の登録
        Args:
            strategy_id: 戦略ID（利用可能な戦略名またはカスタム戦略のID）
            strategy_class: カスタム戦略クラス（オプション）
            **params: 戦略初期化パラメータ
        """
        if strategy_class is None:
            if strategy_id not in self._available_strategies:
                raise ValueError(f"Unknown strategy: {strategy_id}")
            strategy_class = self._available_strategies[strategy_id]
        
        self.strategies[strategy_id] = strategy_class(**params)
    
    def get_strategy(self, strategy_id: str) -> BaseStrategy:
        """
        戦略の取得
        Args:
            strategy_id: 戦略ID
        Returns:
            戦略インスタンス
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy not registered: {strategy_id}")
        return self.strategies[strategy_id]
    
    def list_available_strategies(self) -> Dict[str, Type[BaseStrategy]]:
        """
        利用可能な戦略一覧の取得
        Returns:
            戦略名と戦略クラスの辞書
        """
        return self._available_strategies.copy()
    
    def remove_strategy(self, strategy_id: str) -> None:
        """
        戦略の削除
        Args:
            strategy_id: 戦略ID
        """
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]