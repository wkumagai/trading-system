"""
momentum.py

モメンタム指標を使用した取引戦略
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union
from ..Base.base_strategy import BaseStrategy

class RSIStrategy(BaseStrategy):
    """RSI（Relative Strength Index）を使用した戦略"""
    
    def __init__(self, 
                period: int = 14, 
                overbought: float = 70, 
                oversold: float = 30):
        """
        初期化
        Args:
            period: RSIの計算期間
            overbought: 売られすぎと判断する閾値
            oversold: 買われすぎと判断する閾値
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def _calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """RSIの計算"""
        df = data.copy()
        df['price_diff'] = df['close'].diff()
        
        df['gain'] = df['price_diff'].clip(lower=0)
        df['loss'] = -df['price_diff'].clip(upper=0)
        
        avg_gain = df['gain'].rolling(window=self.period).mean()
        avg_loss = df['loss'].rolling(window=self.period).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df

    def predict(self, 
               data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
               symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """RSIの計算と予測"""
        if isinstance(data, pd.DataFrame):
            return {symbols[0] if symbols else "default": self._calculate_rsi(data)}
        else:
            return {symbol: self._calculate_rsi(df) for symbol, df in data.items()}

    def generate_signals(self, 
                        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                        symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        取引シグナルの生成
        - RSIが閾値を超えた場合にシグナルを生成
        """
        predictions = self.predict(data, symbols)
        signals = {}
        
        for symbol, pred_df in predictions.items():
            df = pred_df.copy()
            df['signal'] = 0
            
            # 売られすぎ（買いシグナル）
            df.loc[df['rsi'] < self.oversold, 'signal'] = 1
            
            # 買われすぎ（売りシグナル）
            df.loc[df['rsi'] > self.overbought, 'signal'] = -1
            
            signals[symbol] = df[['signal', 'rsi']]
            
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

class MACDStrategy(BaseStrategy):
    """MACD（Moving Average Convergence Divergence）を使用した戦略"""
    
    def __init__(self, 
                fast_period: int = 12, 
                slow_period: int = 26, 
                signal_period: int = 9):
        """
        初期化
        Args:
            fast_period: 短期EMAの期間
            slow_period: 長期EMAの期間
            signal_period: シグナルラインの期間
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def _calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """MACDの計算"""
        df = data.copy()
        
        # 指数移動平均の計算
        fast_ema = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # MACD線の計算
        df['macd'] = fast_ema - slow_ema
        
        # シグナル線の計算
        df['signal_line'] = df['macd'].ewm(span=self.signal_period, adjust=False).mean()
        
        # ヒストグラムの計算
        df['histogram'] = df['macd'] - df['signal_line']
        
        return df

    def predict(self, 
               data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
               symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """MACDの計算と予測"""
        if isinstance(data, pd.DataFrame):
            return {symbols[0] if symbols else "default": self._calculate_macd(data)}
        else:
            return {symbol: self._calculate_macd(df) for symbol, df in data.items()}

    def generate_signals(self, 
                        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                        symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        取引シグナルの生成
        - MACDがシグナル線をクロスした場合にシグナルを生成
        """
        predictions = self.predict(data, symbols)
        signals = {}
        
        for symbol, pred_df in predictions.items():
            df = pred_df.copy()
            df['signal'] = 0
            
            # MACDがシグナル線を上向きにクロス（買いシグナル）
            df.loc[(df['macd'] > df['signal_line']) & 
                  (df['macd'].shift(1) <= df['signal_line'].shift(1)), 'signal'] = 1
            
            # MACDがシグナル線を下向きにクロス（売りシグナル）
            df.loc[(df['macd'] < df['signal_line']) & 
                  (df['macd'].shift(1) >= df['signal_line'].shift(1)), 'signal'] = -1
            
            signals[symbol] = df[['signal', 'macd', 'signal_line', 'histogram']]
            
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