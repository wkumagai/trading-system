"""
features.py

株価データから特徴量を生成するモジュール。
テクニカル指標の計算と特徴量エンジニアリングを行う。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from sklearn.preprocessing import MinMaxScaler

from .config import FEATURE_CONFIG, MODEL_CONFIG

class FeatureGenerator:
    """特徴量生成クラス"""
    
    def __init__(self):
        """特徴量生成器の初期化"""
        self.scalers = {}
        self.logger = logging.getLogger(__name__)

    def create_sequence_features(
        self,
        df: pd.DataFrame,
        sequence_length: int = MODEL_CONFIG['sequence_length']
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        時系列データのシーケンス特徴量を生成

        Args:
            df: 入力データ
            sequence_length: シーケンスの長さ

        Returns:
            (X, y): 特徴量とターゲット
        """
        features = FEATURE_CONFIG['sequence_features']
        
        # 特徴量のスケーリング
        scaled_data = {}
        for feature in features:
            if feature not in df.columns:
                continue
            
            scaler = MinMaxScaler()
            scaled_data[feature] = scaler.fit_transform(
                df[feature].values.reshape(-1, 1)
            ).flatten()
            self.scalers[feature] = scaler
        
        # シーケンスデータの作成
        X, y = [], []
        for i in range(len(df) - sequence_length - MODEL_CONFIG['prediction_target']):
            # 特徴量シーケンス
            sequence = []
            for feature in features:
                if feature in scaled_data:
                    sequence.append(
                        scaled_data[feature][i:(i + sequence_length)]
                    )
            X.append(np.array(sequence).T)
            
            # ターゲット（n期後の価格）
            target_idx = i + sequence_length + MODEL_CONFIG['prediction_target']
            y.append(scaled_data['close'][target_idx])
        
        return np.array(X), np.array(y)

    def calculate_technical_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        テクニカル指標を計算

        Args:
            df: 入力データ

        Returns:
            テクニカル指標が追加されたデータ
        """
        df = df.copy()
        
        # 基本的な価格指標
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']).diff()
        
        # ボラティリティ
        df['volatility'] = df['returns'].rolling(
            window=20
        ).std()
        
        # 価格変化率
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change()
        
        # 出来高関連
        df['volume_change'] = df['volume'].diff()
        df['volume_change_pct'] = df['volume'].pct_change()
        
        # 価格帯
        df['price_position'] = (
            (df['close'] - df['low']) / (df['high'] - df['low'])
        )
        
        # モメンタム指標
        df['momentum'] = df['close'].diff(periods=10)
        df['rate_of_change'] = df['close'].pct_change(periods=10)
        
        # 移動平均との関係
        if 'sma' in df.columns:
            df['ma_ratio'] = df['close'] / df['sma']
            df['ma_diff'] = df['close'] - df['sma']
        
        # RSIとの関係
        if 'rsi' in df.columns:
            df['rsi_diff'] = df['rsi'].diff()
            df['rsi_ma'] = df['rsi'].rolling(window=10).mean()
        
        # ボリンジャーバンドとの関係
        if all(col in df.columns for col in ['bbands_upper', 'bbands_lower']):
            df['bb_position'] = (
                (df['close'] - df['bbands_lower']) /
                (df['bbands_upper'] - df['bbands_lower'])
            )
        
        # MACDとの関係
        if 'macd' in df.columns:
            df['macd_diff'] = df['macd'].diff()
            df['macd_ma'] = df['macd'].rolling(window=10).mean()
        
        return df

    def prepare_training_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        学習データの準備

        Args:
            df: 入力データ

        Returns:
            (X, y): 学習用データセット
        """
        # テクニカル指標の計算
        df = self.calculate_technical_features(df)
        
        # 欠損値の除去
        df.dropna(inplace=True)
        
        # シーケンス特徴量の生成
        X, y = self.create_sequence_features(df)
        
        self.logger.info(f"Prepared {len(X)} sequences with shape {X.shape}")
        return X, y

    def inverse_transform_predictions(
        self,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        予測値を元のスケールに戻す

        Args:
            y_pred: スケーリングされた予測値

        Returns:
            元のスケールの予測値
        """
        if 'close' in self.scalers:
            return self.scalers['close'].inverse_transform(
                y_pred.reshape(-1, 1)
            ).flatten()
        return y_pred

if __name__ == "__main__":
    # ロギングの設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # サンプルデータの生成
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1min')
    df = pd.DataFrame({
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates)),
        'sma': np.random.randn(len(dates)).cumsum() + 100,
        'rsi': np.random.uniform(0, 100, len(dates)),
        'macd': np.random.randn(len(dates)),
        'bbands_upper': np.random.randn(len(dates)).cumsum() + 110,
        'bbands_lower': np.random.randn(len(dates)).cumsum() + 90
    }, index=dates)
    
    # 特徴量生成の実行
    generator = FeatureGenerator()
    X, y = generator.prepare_training_data(df)
    
    print("\nFeature generation results:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")