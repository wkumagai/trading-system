"""
deep_learning.py

深層学習モデルを使用した取引戦略
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Union
from ..utils.model_connector import ModelConnector
from ..Base.base_strategy import BaseStrategy

class DeepLearningStrategy(BaseStrategy):
    """深層学習モデルを使用した取引戦略クラス"""
    
    def __init__(self, 
                dataset_id: str,
                model_id: Optional[str] = None,
                threshold: float = 0.01,
                position_size: float = 0.1):
        """
        初期化
        Args:
            dataset_id: データセットID
            model_id: モデルID（指定しない場合は最新のモデルを使用）
            threshold: 取引実行の閾値（予測変化率がこの値を超えた場合に取引）
            position_size: ポジションサイズ（0.0-1.0）
        """
        self.dataset_id = dataset_id
        self.connector = ModelConnector()
        self.threshold = threshold
        self.position_size = position_size
        
        # モデルの読み込み
        model_data = self.connector.load_model(dataset_id, model_id)
        self.model = model_data["model"]
        self.model_info = model_data["metadata"]
        
        # モデルのパラメータを取得
        self.parameters = self.model_info["parameters"]
        self.sequence_length = self.parameters["sequence_length"]
        self.features = self.parameters["features"]

    def predict(self, 
               data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
               symbols: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        価格変動の予測
        Args:
            data: 入力データ（単一銘柄のDataFrameまたは複数銘柄のDataFrameの辞書）
            symbols: 予測対象の銘柄リスト（複数銘柄の場合）
        Returns:
            銘柄ごとの予測値の辞書
        """
        # 入力データの準備
        input_tensor = self.connector.prepare_input_data(data, self.dataset_id)
        
        # 予測の実行
        predictions = self.model.predict(input_tensor)
        
        # 予測結果の整形
        if isinstance(data, pd.DataFrame):
            # 単一銘柄の場合
            return {symbols[0] if symbols else "default": predictions}
        else:
            # 複数銘柄の場合
            target_symbols = symbols if symbols else list(data.keys())
            results = {}
            pred_index = 0
            for symbol in target_symbols:
                symbol_data = data[symbol]
                num_sequences = len(symbol_data) - self.sequence_length + 1
                results[symbol] = predictions[pred_index:pred_index + num_sequences]
                pred_index += num_sequences
            return results

    def generate_signals(self, 
                        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                        symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        取引シグナルの生成
        Args:
            data: 入力データ
            symbols: 対象銘柄リスト
        Returns:
            銘柄ごとの取引シグナルのDataFrame
        """
        predictions = self.predict(data, symbols)
        signals = {}
        
        for symbol, preds in predictions.items():
            # 予測値から取引シグナルを生成
            signal_data = []
            for pred in preds:
                if pred > self.threshold:
                    signal = 1  # 買いシグナル
                elif pred < -self.threshold:
                    signal = -1  # 売りシグナル
                else:
                    signal = 0  # ホールド
                    
                signal_data.append({
                    'signal': signal,
                    'prediction': float(pred),
                    'position_size': self.position_size if signal != 0 else 0
                })
            
            signals[symbol] = pd.DataFrame(signal_data)
            
        return signals

    def get_model_info(self) -> dict:
        """
        モデル情報の取得
        Returns:
            モデル情報の辞書
        """
        return {
            "model_id": self.model_info.get("model_id"),
            "dataset_id": self.dataset_id,
            "metrics": self.model_info.get("metrics", {}),
            "parameters": self.parameters,
            "created_at": self.model_info.get("created_at")
        }

    def validate_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> bool:
        """
        入力データの検証
        Args:
            data: 検証するデータ
        Returns:
            検証結果（True/False）
        """
        try:
            # 必要な特徴量が存在するか確認
            if isinstance(data, pd.DataFrame):
                return all(feature in data.columns for feature in self.features)
            else:
                return all(
                    all(feature in df.columns for feature in self.features)
                    for df in data.values()
                )
        except Exception as e:
            print(f"Data validation failed: {str(e)}")
            return False