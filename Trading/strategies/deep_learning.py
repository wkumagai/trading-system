"""
deep_learning.py

LSTMを使用した深層学習ベースのトレーディング戦略の実装。
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from .base import BaseStrategy

class LSTMStrategy(BaseStrategy):
    """
    LSTM（Long Short-Term Memory）を使用した戦略。
    過去の価格パターンから将来の価格動向を予測する。
    """

    def __init__(self, config, sequence_length=10, epochs=50, batch_size=32):
        """
        Args:
            config: システム全体の設定値を持つモジュール
            sequence_length (int): 予測に使用する過去のデータポイント数
            epochs (int): 学習エポック数
            batch_size (int): バッチサイズ
        """
        super().__init__(config)
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()

    @property
    def strategy_name(self) -> str:
        return f"lstm_seq{self.sequence_length}"

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        価格データから特徴量を生成する。

        Args:
            df: 価格データを含むDataFrame

        Returns:
            特徴量が追加されたDataFrame
        """
        df = df.copy()

        # テクニカル指標の計算
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # ボリンジャーバンド
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['MA_20'] + (std_20 * 2)
        df['BB_Lower'] = df['MA_20'] - (std_20 * 2)

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # モメンタム
        df['Momentum'] = df['Close'].pct_change(periods=10)

        # 出来高の変化率
        df['Volume_Change'] = df['Volume'].pct_change()

        # 欠損値を除去
        df.dropna(inplace=True)

        return df

    def _prepare_sequences(self, data):
        """
        LSTMのための入力シーケンスを準備する。

        Args:
            data: スケーリング済みの特徴量データ

        Returns:
            X: 入力シーケンス
            y: 教師データ（次の期間の価格変化）
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            # 次の期間の価格が上がれば1、下がれば0
            price_change = data[i + self.sequence_length, 0] > data[i + self.sequence_length - 1, 0]
            y.append(1 if price_change else 0)
        return np.array(X), np.array(y)

    def _build_model(self, input_shape):
        """
        LSTMモデルを構築する。

        Args:
            input_shape: 入力データの形状

        Returns:
            構築されたKerasモデル
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_model(self, df: pd.DataFrame) -> None:
        """
        LSTMモデルを学習する。

        Args:
            df: 特徴量を含むDataFrame
        """
        # 特徴量の準備
        feature_columns = ['Close', 'RSI', 'MACD', 'Momentum', 'Volume_Change']
        features = df[feature_columns].values
        
        # スケーリング
        scaled_features = self.feature_scaler.fit_transform(features)
        
        # シーケンスデータの準備
        X, y = self._prepare_sequences(scaled_features)
        
        # モデルの構築と学習
        if self.model is None:
            self.model = self._build_model(input_shape=(self.sequence_length, len(feature_columns)))
        
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=0
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        学習したモデルで予測を行う。

        Args:
            df: 特徴量を含むDataFrame

        Returns:
            シグナル列が追加されたDataFrame
        """
        df = df.copy()
        
        # 特徴量の準備
        feature_columns = ['Close', 'RSI', 'MACD', 'Momentum', 'Volume_Change']
        features = df[feature_columns].values
        
        # スケーリング
        scaled_features = self.feature_scaler.transform(features)
        
        # 予測用シーケンスの準備
        sequences = []
        for i in range(len(scaled_features) - self.sequence_length + 1):
            sequences.append(scaled_features[i:(i + self.sequence_length)])
        
        if not sequences:
            df['signal'] = 0
            return df
            
        # 予測
        predictions = self.model.predict(np.array(sequences), verbose=0)
        
        # シグナルの生成
        df['signal'] = 0
        df.loc[self.sequence_length-1:, 'signal'] = np.where(
            predictions > 0.5,
            1,  # 上昇予測
            -1  # 下降予測
        )
        
        return df

    def get_parameters(self) -> dict:
        """
        戦略のパラメータを取得。

        Returns:
            dict: パラメータ名と値のディクショナリ
        """
        return {
            'sequence_length': self.sequence_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }

    def set_parameters(self, parameters: dict) -> None:
        """
        戦略のパラメータを設定。

        Args:
            parameters: パラメータ名と値のディクショナリ
        """
        if 'sequence_length' in parameters:
            self.sequence_length = parameters['sequence_length']
        if 'epochs' in parameters:
            self.epochs = parameters['epochs']
        if 'batch_size' in parameters:
            self.batch_size = parameters['batch_size']