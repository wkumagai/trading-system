"""
base_strategy.py

取引戦略の基底クラス
全ての戦略クラスはこのクラスを継承する
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Union
import pandas as pd

class BaseStrategy(ABC):
    """取引戦略の基底クラス"""

    @abstractmethod
    def predict(self, 
               data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
               symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        予測の実行
        Args:
            data: 入力データ（単一銘柄のDataFrameまたは複数銘柄のDataFrameの辞書）
            symbols: 予測対象の銘柄リスト（複数銘柄の場合）
        Returns:
            予測結果
        """
        pass

    @abstractmethod
    def generate_signals(self, 
                        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                        symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        取引シグナルの生成
        Args:
            data: 入力データ
            symbols: 対象銘柄リスト
        Returns:
            取引シグナル
        """
        pass

    @abstractmethod
    def validate_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> bool:
        """
        入力データの検証
        Args:
            data: 検証するデータ
        Returns:
            検証結果（True/False）
        """
        pass

    def execute(self, 
                data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        戦略の実行
        Args:
            data: 入力データ
            symbols: 対象銘柄リスト
        Returns:
            取引シグナル
        """
        # データの検証
        if not self.validate_data(data):
            raise ValueError("Invalid input data")

        # シグナルの生成
        return self.generate_signals(data, symbols)

    def get_strategy_info(self) -> dict:
        """
        戦略情報の取得
        Returns:
            戦略情報の辞書
        """
        return {
            "strategy_type": self.__class__.__name__,
            "description": self.__doc__,
            "parameters": self._get_parameters()
        }

    def _get_parameters(self) -> dict:
        """
        戦略パラメータの取得
        Returns:
            パラメータの辞書
        """
        # 継承クラスで必要に応じてオーバーライド
        return {}