"""
model_loader.py

学習済みモデルを読み込み、予測に使用するモジュール。
学習プロセスは別フォルダ（ModelTraining）で管理。
"""

import os
import json
import logging
from typing import Dict, Any
import tensorflow as tf
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class ModelLoader:
    """学習済みモデルのローダー"""
    
    def __init__(self, model_dir: str):
        """
        Args:
            model_dir: 学習済みモデルが格納されているディレクトリ
        """
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.model_configs = {}
        self.logger = logging.getLogger(__name__)

    def load_model(self, model_id: str) -> None:
        """
        学習済みモデルを読み込む

        Args:
            model_id: モデルの識別子
        """
        model_path = os.path.join(self.model_dir, model_id)
        
        # モデル情報の読み込み
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model_type = config['model_type']
        self.model_configs[model_id] = config

        # モデルの読み込み
        if model_type == 'lstm':
            model_file = os.path.join(model_path, 'model.h5')
            self.models[model_id] = tf.keras.models.load_model(model_file)
        elif model_type == 'xgboost':
            model_file = os.path.join(model_path, 'model.joblib')
            self.models[model_id] = joblib.load(model_file)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # スケーラーの読み込み
        scaler_file = os.path.join(model_path, 'scaler.joblib')
        self.scalers[model_id] = joblib.load(scaler_file)
        
        self.logger.info(f"Loaded model {model_id} of type {model_type}")

    def predict(self, model_id: str, X: np.ndarray) -> np.ndarray:
        """
        モデルを使用して予測を実行

        Args:
            model_id: モデルの識別子
            X: 入力データ

        Returns:
            予測結果
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        config = self.model_configs[model_id]
        
        # データの前処理
        if config['model_type'] == 'lstm':
            # シーケンスデータの場合
            sequence_length = config['sequence_length']
            if len(X) < sequence_length:
                raise ValueError(f"Input data length {len(X)} is less than sequence_length {sequence_length}")
            
            # シーケンスの準備
            sequences = []
            for i in range(len(X) - sequence_length + 1):
                sequence = X[i:(i + sequence_length)]
                sequences.append(sequence)
            
            X = np.array(sequences)
            
            # スケーリング
            original_shape = X.shape
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scalers[model_id].transform(X_reshaped)
            X_scaled = X_scaled.reshape(original_shape)
        else:
            # 通常のデータの場合
            X_scaled = self.scalers[model_id].transform(X)

        # 予測の実行
        return self.models[model_id].predict(X_scaled)

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        モデルの情報を取得

        Args:
            model_id: モデルの識別子

        Returns:
            モデルの設定情報
        """
        if model_id not in self.model_configs:
            raise ValueError(f"Model {model_id} not loaded")
        
        return self.model_configs[model_id]

    def list_available_models(self) -> list:
        """
        利用可能なモデルの一覧を取得

        Returns:
            利用可能なモデルIDのリスト
        """
        models = []
        for item in os.listdir(self.model_dir):
            config_path = os.path.join(self.model_dir, item, 'config.json')
            if os.path.isfile(config_path):
                models.append(item)
        return models