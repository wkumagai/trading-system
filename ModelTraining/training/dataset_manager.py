"""
dataset_manager.py

データセットの管理と読み込みを行うクラス
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime

class DatasetManager:
    """データセット管理クラス"""
    
    def __init__(self, base_path: str = "ModelTraining/data/datasets"):
        """
        初期化
        Args:
            base_path: データセットのベースパス
        """
        self.base_path = base_path
        self.metadata_path = os.path.join(base_path, "metadata")
        self.processed_path = os.path.join(base_path, "processed")
        
    def get_dataset_config(self, dataset_id: str) -> dict:
        """
        データセット設定の取得
        Args:
            dataset_id: データセットID
        Returns:
            設定情報の辞書
        """
        # メタデータファイルを探索
        for filename in os.listdir(self.metadata_path):
            if filename.endswith('.json'):
                file_path = os.path.join(self.metadata_path, filename)
                with open(file_path, 'r') as f:
                    config = json.load(f)
                    if config.get('dataset_id') == dataset_id:
                        return config
        raise ValueError(f"Dataset {dataset_id} not found")

    def load_dataset(self, dataset_id: str, symbols: Optional[List[str]] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        データセットの読み込み
        Args:
            dataset_id: データセットID
            symbols: 特定の銘柄のみ読み込む場合に指定
        Returns:
            単一銘柄の場合はDataFrame、複数銘柄の場合は銘柄をキーとするDataFrameの辞書
        """
        config = self.get_dataset_config(dataset_id)
        target_symbols = symbols if symbols else config['symbols']
        
        if config['type'] == 'single_stock':
            # 単一銘柄の場合
            file_path = os.path.join(self.processed_path, f"{dataset_id}.csv")
            return pd.read_csv(file_path, parse_dates=['date'])
        else:
            # 複数銘柄の場合
            data = {}
            for symbol in target_symbols:
                file_path = os.path.join(self.processed_path, f"{dataset_id}_{symbol}.csv")
                data[symbol] = pd.read_csv(file_path, parse_dates=['date'])
            return data

    def preprocess_dataset(self, dataset_id: str) -> None:
        """
        データセットの前処理
        Args:
            dataset_id: データセットID
        """
        config = self.get_dataset_config(dataset_id)
        preprocessing = config['preprocessing']
        
        # 前処理のロジックを実装
        # - データの正規化
        # - シーケンスデータの作成
        # - 訓練/検証データの分割
        pass

    def get_feature_list(self, dataset_id: str) -> List[str]:
        """
        特徴量リストの取得
        Args:
            dataset_id: データセットID
        Returns:
            特徴量のリスト
        """
        config = self.get_dataset_config(dataset_id)
        return config['features']

    def get_sequence_length(self, dataset_id: str) -> int:
        """
        シーケンス長の取得
        Args:
            dataset_id: データセットID
        Returns:
            シーケンス長
        """
        config = self.get_dataset_config(dataset_id)
        return config['preprocessing']['sequence_length']

    def validate_dataset(self, dataset_id: str) -> bool:
        """
        データセットの検証
        Args:
            dataset_id: データセットID
        Returns:
            検証結果（True/False）
        """
        try:
            config = self.get_dataset_config(dataset_id)
            # 必要なファイルの存在確認
            # データの整合性チェック
            return True
        except Exception as e:
            print(f"Dataset validation failed: {str(e)}")
            return False