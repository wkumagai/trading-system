"""
moving_average.py

移動平均を使用した基本的なトレーディング戦略の実装。
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy

class MovingAverageStrategy(BaseStrategy):
    """
    シンプルな移動平均クロス戦略。
    短期移動平均が長期移動平均を上回ったら買い、
    下回ったら売りのシグナルを生成する。
    """

    def __init__(self, config, short_window=5, long_window=20):
        """
        Args:
            config: システム全体の設定値を持つモジュール
            short_window (int): 短期移動平均の期間
            long_window (int): 長期移動平均の期間
        """
        super().__init__(config)
        self.short_window = short_window
        self.long_window = long_window

    @property
    def strategy_name(self) -> str:
        return f"moving_average_{self.short_window}_{self.long_window}"

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        価格データから移動平均を計算する。

        Args:
            df: 価格データを含むDataFrame

        Returns:
            移動平均が追加されたDataFrame
        """
        df = df.copy()
        
        # 短期・長期の移動平均を計算
        df[f'MA_{self.short_window}'] = df['Close'].rolling(window=self.short_window).mean()
        df[f'MA_{self.long_window}'] = df['Close'].rolling(window=self.long_window).mean()
        
        # オプションの特徴量
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # 欠損値を除去
        df.dropna(inplace=True)
        
        return df

    def train_model(self, df: pd.DataFrame) -> None:
        """
        移動平均戦略は学習不要なので何もしない。
        """
        pass

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        移動平均のクロスオーバーに基づいてシグナルを生成。

        Args:
            df: 特徴量を含むDataFrame

        Returns:
            シグナル列が追加されたDataFrame
        """
        df = df.copy()
        
        # クロスオーバーの検出
        df['signal'] = np.where(
            df[f'MA_{self.short_window}'] > df[f'MA_{self.long_window}'], 
            1,  # 短期が長期を上回る → 買い
            -1  # 短期が長期を下回る → 売り
        )
        
        # シグナルの変化点のみを抽出（ポジション変更時のみ）
        df['signal_change'] = df['signal'].diff()
        df['signal'] = np.where(df['signal_change'] == 0, 0, df['signal'])
        
        return df

    def get_parameters(self) -> dict:
        """
        戦略のパラメータを取得。

        Returns:
            dict: パラメータ名と値のディクショナリ
        """
        return {
            'short_window': self.short_window,
            'long_window': self.long_window
        }

    def set_parameters(self, parameters: dict) -> None:
        """
        戦略のパラメータを設定。

        Args:
            parameters: パラメータ名と値のディクショナリ
        """
        if 'short_window' in parameters:
            self.short_window = parameters['short_window']
        if 'long_window' in parameters:
            self.long_window = parameters['long_window']