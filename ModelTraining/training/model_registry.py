"""
model_registry.py

モデルの登録と管理を行うクラス
"""

import os
import json
from typing import Dict, Optional, List
from datetime import datetime

class ModelRegistry:
    """モデル登録管理クラス"""
    
    def __init__(self, base_path: str = "ModelTraining/models"):
        """
        初期化
        Args:
            base_path: モデルのベースパス
        """
        self.base_path = base_path
        self.registry_path = os.path.join(base_path, "registry", "models.json")
        self.trained_path = os.path.join(base_path, "trained")
        
    def _load_registry(self) -> dict:
        """レジストリファイルの読み込み"""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {"models": {}}
    
    def _save_registry(self, registry: dict) -> None:
        """レジストリファイルの保存"""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

    def register_model(self, 
                      model_id: str,
                      dataset_id: str,
                      model_path: str,
                      metrics: dict,
                      parameters: dict,
                      training_config: dict) -> None:
        """
        モデルの登録
        Args:
            model_id: モデルID
            dataset_id: データセットID
            model_path: モデルファイルのパス
            metrics: 評価指標
            parameters: モデルパラメータ
            training_config: 学習設定
        """
        registry = self._load_registry()
        
        registry["models"][model_id] = {
            "dataset_id": dataset_id,
            "model_type": parameters.get("architecture", {}).get("type", "unknown"),
            "path": model_path,
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "metrics": metrics,
            "parameters": parameters,
            "training_config": training_config
        }
        
        self._save_registry(registry)

    def get_model_info(self, model_id: str) -> Optional[dict]:
        """
        モデル情報の取得
        Args:
            model_id: モデルID
        Returns:
            モデル情報の辞書
        """
        registry = self._load_registry()
        return registry["models"].get(model_id)

    def get_latest_model(self, dataset_id: str) -> Optional[dict]:
        """
        指定されたデータセットの最新モデルを取得
        Args:
            dataset_id: データセットID
        Returns:
            最新のモデル情報
        """
        registry = self._load_registry()
        models = [
            model for model in registry["models"].values()
            if model["dataset_id"] == dataset_id
        ]
        
        if not models:
            return None
            
        return max(models, key=lambda x: x["created_at"])

    def list_models(self, dataset_id: Optional[str] = None) -> List[dict]:
        """
        モデル一覧の取得
        Args:
            dataset_id: 特定のデータセットのモデルのみ取得する場合に指定
        Returns:
            モデル情報のリスト
        """
        registry = self._load_registry()
        models = registry["models"].values()
        
        if dataset_id:
            models = [m for m in models if m["dataset_id"] == dataset_id]
            
        return list(models)

    def delete_model(self, model_id: str) -> bool:
        """
        モデルの削除
        Args:
            model_id: モデルID
        Returns:
            削除成功の場合True
        """
        registry = self._load_registry()
        
        if model_id in registry["models"]:
            model_info = registry["models"][model_id]
            model_path = os.path.join(self.base_path, model_info["path"])
            
            # モデルファイルの削除
            if os.path.exists(model_path):
                os.remove(model_path)
            
            # レジストリから削除
            del registry["models"][model_id]
            self._save_registry(registry)
            return True
            
        return False

    def validate_model(self, model_id: str) -> bool:
        """
        モデルの検証
        Args:
            model_id: モデルID
        Returns:
            検証結果（True/False）
        """
        model_info = self.get_model_info(model_id)
        if not model_info:
            return False
            
        model_path = os.path.join(self.base_path, model_info["path"])
        return os.path.exists(model_path)