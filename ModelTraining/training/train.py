"""
train.py

LSTMモデルの学習を行うモジュール。
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import logging
import os
from datetime import datetime
from typing import Dict, Any, Tuple

from .config import MODEL_CONFIG
from .features import FeatureGenerator
from ..data_collection.fetcher import MarketDataFetcher

class LSTMModel:
    """LSTMモデルクラス"""
    
    def __init__(self, config: Dict[str, Any] = MODEL_CONFIG):
        """
        Args:
            config: モデルの設定
        """
        self.config = config
        self.model = None
        self.feature_generator = FeatureGenerator()
        self.logger = logging.getLogger(__name__)

    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        LSTMモデルの構築

        Args:
            input_shape: 入力データの形状
        """
        model = Sequential([
            # 入力層を明示的に定義
            Input(shape=input_shape)
        ])
        
        # LSTM層の追加
        for i, units in enumerate(self.config['architecture']['lstm_layers']):
            return_sequences = i < len(self.config['architecture']['lstm_layers']) - 1
            model.add(LSTM(
                units,
                return_sequences=return_sequences
            ))
            model.add(Dropout(self.config['architecture']['dropout_rate']))
        
        # Dense層の追加
        for units in self.config['architecture']['dense_layers']:
            model.add(Dense(
                units,
                activation=self.config['architecture']['activation']
            ))
            model.add(Dropout(self.config['architecture']['dropout_rate']))
        
        # 出力層
        model.add(Dense(
            1,
            activation=self.config['architecture']['output_activation']
        ))
        
        # モデルのコンパイル
        model.compile(
            optimizer=Adam(
                learning_rate=self.config['training']['learning_rate']
            ),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
        # モデル構造の表示
        self.logger.info("Model Architecture:")
        self.model.summary(print_fn=self.logger.info)
        self.logger.info(f"Input Shape: {input_shape}")
        self.logger.info(f"Learning Rate: {self.config['training']['learning_rate']}")
        self.logger.info(f"Batch Size: {self.config['training']['batch_size']}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> tf.keras.callbacks.History:
        """
        モデルの学習を実行

        Args:
            X_train: 訓練データ
            y_train: 訓練データのラベル
            X_val: 検証データ
            y_val: 検証データのラベル

        Returns:
            学習履歴
        """
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Early Stoppingの設定
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True
        )
        
        # モデルの学習
        history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.config['training']['batch_size'],
            epochs=self.config['training']['epochs'],
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        予測の実行

        Args:
            X: 入力データ

        Returns:
            予測値
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)

    def save(self, model_dir: str) -> None:
        """
        モデルの保存

        Args:
            model_dir: 保存先ディレクトリ
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # モデルの保存（新しい.kerasフォーマット）
        model_path = os.path.join(model_dir, 'model.keras')
        tf.keras.models.save_model(
            self.model,
            model_path,
            save_format='keras'
        )
        
        # スケーラーの保存
        if hasattr(self.feature_generator, 'scalers'):
            import joblib
            joblib.dump(
                self.feature_generator.scalers,
                os.path.join(model_dir, 'scalers.joblib')
            )
        
        self.logger.info(f"Model saved to {model_dir}")

    def load(self, model_dir: str) -> None:
        """
        モデルの読み込み

        Args:
            model_dir: モデルの保存ディレクトリ
        """
        model_path = os.path.join(model_dir, 'model.keras')
        scaler_path = os.path.join(model_dir, 'scalers.joblib')
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model not found at {model_path}")
        
        # モデルの読み込み
        self.model = tf.keras.models.load_model(model_path)
        
        # スケーラーの読み込み
        if os.path.exists(scaler_path):
            import joblib
            self.feature_generator.scalers = joblib.load(scaler_path)
        
        self.logger.info(f"Model loaded from {model_dir}")

def train_model(
    data_fetcher: MarketDataFetcher,
    model_id: str,
    test_size: float = 0.2,
    validation_size: float = 0.2
) -> Dict[str, Any]:
    """
    モデルの学習を実行

    Args:
        data_fetcher: データ取得クラス
        model_id: モデルの識別子
        test_size: テストデータの割合
        validation_size: 検証データの割合

    Returns:
        学習結果
    """
    # データの取得
    df = data_fetcher.fetch_data()
    if df.empty:
        raise ValueError("No data fetched")
    
    # 特徴量の生成
    feature_generator = FeatureGenerator()
    X, y = feature_generator.prepare_training_data(df)
    
    # データの分割
    train_size = int(len(X) * (1 - test_size - validation_size))
    val_size = int(len(X) * validation_size)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # モデルの学習
    model = LSTMModel()
    history = model.train(X_train, y_train, X_val, y_val)
    
    # テストデータでの評価
    test_loss, test_mae = model.model.evaluate(X_test, y_test, verbose=0)
    
    # モデルの保存
    model_dir = f"../models/{model_id}"
    model.save(model_dir)
    
    return {
        'model_id': model_id,
        'test_loss': test_loss,
        'test_mae': test_mae,
        'history': history.history,
        'model_dir': model_dir
    }

if __name__ == "__main__":
    # ロギングの設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # データ取得の準備
    data_fetcher = MarketDataFetcher()
    
    # モデルの学習実行
    model_id = f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        results = train_model(data_fetcher, model_id)
        print("\nTraining completed successfully!")
        print(f"Model saved to: {results['model_dir']}")
        print(f"Test Loss: {results['test_loss']:.4f}")
        print(f"Test MAE: {results['test_mae']:.4f}")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")