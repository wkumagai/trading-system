"""
moving_average.py

移動平均を使用した取引戦略
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union
from ..Base.base_strategy import BaseStrategy

class SimpleMAStrategy(BaseStrategy):
    """シンプルな移動平均クロス戦略"""
    
    def __init__(self, short_window: int = 5, long_window: int = 20):
        """
        初期化
        Args:
            short_window: 短期移動平均の期間
            long_window: 長期移動平均の期間
        """
        self.short_window = short_window
        self.long_window = long_window

    def predict(self, 
               data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
               symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        移動平均の計算と予測
        Args:
            data: 価格データ
            symbols: 対象銘柄リスト
        Returns:
            移動平均とクロスシグナル
        """
        if isinstance(data, pd.DataFrame):
            return {symbols[0] if symbols else "default": self._calculate_ma(data)}
        else:
            return {symbol: self._calculate_ma(df) for symbol, df in data.items()}

    def _calculate_ma(self, data: pd.DataFrame) -> pd.DataFrame:
        """移動平均の計算"""
        df = data.copy()
        df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
        return df

    def generate_signals(self, 
                        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                        symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        取引シグナルの生成
        Args:
            data: 価格データ
            symbols: 対象銘柄リスト
        Returns:
            取引シグナル
        """
        predictions = self.predict(data, symbols)
        signals = {}
        
        for symbol, pred_df in predictions.items():
            df = pred_df.copy()
            # クロスオーバーの検出
            df['signal'] = 0
            df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1  # 買いシグナル
            df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1  # 売りシグナル
            
            # シグナルの変化点のみを抽出
            df['signal_change'] = df['signal'].diff()
            df.loc[df['signal_change'] == 0, 'signal'] = 0
            
            signals[symbol] = df[['signal', 'short_ma', 'long_ma']]
            
        return signals

    def validate_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> bool:
        """
        入力データの検証
        Args:
            data: 検証するデータ
        Returns:
            検証結果（True/False）
        """
        try:
            if isinstance(data, pd.DataFrame):
                return 'close' in data.columns
            else:
                return all('close' in df.columns for df in data.values())
        except Exception as e:
            print(f"Data validation failed: {str(e)}")
            return False

class TripleMAStrategy(BaseStrategy):
    """3本の移動平均を使用した戦略"""
    
    def __init__(self, short_window: int = 5, mid_window: int = 20, long_window: int = 50):
        """
        初期化
        Args:
            short_window: 短期移動平均の期間
            mid_window: 中期移動平均の期間
            long_window: 長期移動平均の期間
        """
        self.short_window = short_window
        self.mid_window = mid_window
        self.long_window = long_window

    def predict(self, 
               data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
               symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """移動平均の計算と予測"""
        if isinstance(data, pd.DataFrame):
            return {symbols[0] if symbols else "default": self._calculate_ma(data)}
        else:
            return {symbol: self._calculate_ma(df) for symbol, df in data.items()}

    def _calculate_ma(self, data: pd.DataFrame) -> pd.DataFrame:
        """3本の移動平均を計算"""
        df = data.copy()
        df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df['mid_ma'] = df['close'].rolling(window=self.mid_window).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
        return df

    def generate_signals(self, 
                        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                        symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        取引シグナルの生成
        - 全ての移動平均が順列に並んでいる場合にシグナルを生成
        """
        predictions = self.predict(data, symbols)
        signals = {}
        
        for symbol, pred_df in predictions.items():
            df = pred_df.copy()
            df['signal'] = 0
            
            # 強気シグナル: 短期 > 中期 > 長期
            bull_condition = (df['short_ma'] > df['mid_ma']) & (df['mid_ma'] > df['long_ma'])
            df.loc[bull_condition, 'signal'] = 1
            
            # 弱気シグナル: 短期 < 中期 < 長期
            bear_condition = (df['short_ma'] < df['mid_ma']) & (df['mid_ma'] < df['long_ma'])
            df.loc[bear_condition, 'signal'] = -1
            
            # シグナルの変化点のみを抽出
            df['signal_change'] = df['signal'].diff()
            df.loc[df['signal_change'] == 0, 'signal'] = 0
            
            signals[symbol] = df[['signal', 'short_ma', 'mid_ma', 'long_ma']]
            
        return signals

    def validate_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> bool:
        """入力データの検証"""
        try:
            if isinstance(data, pd.DataFrame):
                return 'close' in data.columns
            else:
                return all('close' in df.columns for df in data.values())
        except Exception as e:
            print(f"Data validation failed: {str(e)}")
            return False