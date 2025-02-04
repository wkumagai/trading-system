"""
base.py

トレーディング戦略の基底クラスを定義するモジュール。
すべての戦略はこのBaseStrategyを継承して実装する。
"""

from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """
    トレーディング戦略の基底クラス。
    すべての具体的な戦略クラスはこのクラスを継承する。
    """

    def __init__(self, config):
        """
        Args:
            config: システム全体の設定値を持つモジュール
        """
        self.config = config

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """
        戦略の一意な識別子を返す。
        Returns:
            str: 戦略の名前（例: "moving_average_5_20"）
        """
        pass

    @abstractmethod
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        価格データから特徴量を生成する。

        Args:
            df: 価格データを含むDataFrame
                必須列: ["Date", "Open", "High", "Low", "Close", "Volume"]

        Returns:
            特徴量が追加されたDataFrame
        """
        pass

    @abstractmethod
    def train_model(self, df: pd.DataFrame) -> None:
        """
        特徴量を使ってモデルを学習する。

        Args:
            df: 特徴量を含むDataFrame
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        モデルを使って予測を行い、売買シグナルを生成する。

        Args:
            df: 特徴量を含むDataFrame

        Returns:
            シグナル列が追加されたDataFrame
            - signal: 1（買い）/ -1（売り）/ 0（ホールド）
        """
        pass

    def get_parameters(self) -> dict:
        """
        戦略のパラメータを取得する。
        オーバーライド可能。

        Returns:
            dict: パラメータ名と値のディクショナリ
        """
        return {}

    def set_parameters(self, parameters: dict) -> None:
        """
        戦略のパラメータを設定する。
        オーバーライド可能。

        Args:
            parameters: パラメータ名と値のディクショナリ
        """
        pass