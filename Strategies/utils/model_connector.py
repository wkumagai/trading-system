"""
model_connector.py

ModelTrainingとの連携を行うクラス
"""

import os
import sys
from typing import Optional, Dict, Any, Union
import tensorflow as tf
import pandas as pd

# ModelTrainingのパスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from ModelTraining.training.model_registry import ModelRegistry
from ModelTraining.training.dataset_manager import DatasetManager

class ModelConnector:
    """ModelTrainingとの連携クラス"""
    
    def __init__(self):
        """初期化"""
        self.model_registry = ModelRegistry()
        self.dataset_manager = DatasetManager()
        
    def load_model(self, 
                  dataset_id: str, 
                  model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        モデルの読み込み
        Args:
            dataset_id: データセットID
            model_id: モデルID（指定しない場合は最新のモデルを使用）
        Returns:
            モデル情報（モデル本体とメタデータ）
        """
        # モデル情報の取得
        if model_id:
            model_info = self.model_registry.get_model_info(model_id)
            if not model_info:
                raise ValueError(f"Model {model_id} not found")
        else:
            model_info = self.model_registry.get_latest_model(dataset_id)
            if not model_info:
                raise ValueError(f"No models found for dataset {dataset_id}")

        # モデルファイルの読み込み
        model_path = os.path.join("ModelTraining/models", model_info["path"])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = tf.keras.models.load_model(model_path)
        
        return {
            "model": model,
            "metadata": model_info
        }

    def prepare_input_data(self, 
                         data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                         dataset_id: str) -> tf.Tensor:
        """
        入力データの前処理
        Args:
            data: 入力データ（単一銘柄のDataFrameまたは複数銘柄のDataFrameの辞書）
            dataset_id: データセットID
        Returns:
            モデルへの入力用テンソル
        """
        config = self.dataset_manager.get_dataset_config(dataset_id)
        features = config["features"]
        sequence_length = config["preprocessing"]["sequence_length"]
        
        if isinstance(data, pd.DataFrame):
            # 単一銘柄の場合
            return self._prepare_single_stock_data(data, features, sequence_length)
        else:
            # 複数銘柄の場合
            return self._prepare_multi_stock_data(data, features, sequence_length)

    def _prepare_single_stock_data(self, 
                                data: pd.DataFrame, 
                                features: list, 
                                sequence_length: int) -> tf.Tensor:
        """単一銘柄データの前処理"""
        # 特徴量の選択
        feature_data = data[features].values
        
        # シーケンスデータの作成
        sequences = []
        for i in range(len(feature_data) - sequence_length + 1):
            sequences.append(feature_data[i:i + sequence_length])
            
        return tf.convert_to_tensor(sequences, dtype=tf.float32)

    def _prepare_multi_stock_data(self, 
                               data: Dict[str, pd.DataFrame], 
                               features: list, 
                               sequence_length: int) -> tf.Tensor:
        """複数銘柄データの前処理"""
        all_sequences = []
        
        for symbol, df in data.items():
            # 各銘柄ごとにシーケンスデータを作成
            feature_data = df[features].values
            
            sequences = []
            for i in range(len(feature_data) - sequence_length + 1):
                sequences.append(feature_data[i:i + sequence_length])
                
            all_sequences.extend(sequences)
            
        return tf.convert_to_tensor(all_sequences, dtype=tf.float32)

    def get_model_parameters(self, model_id: str) -> dict:
        """
        モデルパラメータの取得
        Args:
            model_id: モデルID
        Returns:
            モデルパラメータの辞書
        """
        model_info = self.model_registry.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found")
            
        return model_info["parameters"]

    def get_model_metrics(self, model_id: str) -> dict:
        """
        モデル評価指標の取得
        Args:
            model_id: モデルID
        Returns:
            評価指標の辞書
        """
        model_info = self.model_registry.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found")
            
        return model_info["metrics"]