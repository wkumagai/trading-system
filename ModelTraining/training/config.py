"""
config.py

モデル学習の設定を管理するモジュール。
"""

import os
from typing import Dict, Any

# Alpha Vantage設定
ALPHA_VANTAGE_CONFIG = {
    'api_key': "HIB19MS8JAUS2ABC",
    'symbol': "NVDA",
    'interval': "1min",
    'start_date': "2024-12-01",
    'requests_per_minute': 5
}

# データ設定
DATA_CONFIG = {
    'raw_dir': '../data/raw',
    'processed_dir': '../data/processed',
    'features_dir': '../data/features',
    'test_size': 0.2,
    'validation_size': 0.2,
    'sequence_length': 10
}

# 特徴量設定
FEATURE_CONFIG = {
    'sequence_features': [
        'close',
        'volume',
        'sma',
        'rsi',
        'macd',
        'macd_signal',
        'bb_upper',
        'bb_lower'
    ]
}

# LSTMモデル設定
MODEL_CONFIG = {
    'architecture': {
        'lstm_layers': [50, 25],  # 2層のLSTM
        'dense_layers': [10],     # 1層のDense
        'dropout_rate': 0.2,
        'activation': 'relu',
        'output_activation': 'linear'
    },
    'training': {
        'batch_size': 16,         # バッチサイズを小さく
        'epochs': 100,            # エポック数を増やす
        'learning_rate': 0.0005,  # 学習率を小さく
        'early_stopping_patience': 10  # より長い待機
    }
}

# 評価設定
EVALUATION_CONFIG = {
    'metrics': [
        'mse',
        'mae',
        'rmse',
        'mape',
        'directional_accuracy'
    ],
    'visualization': {
        'show_predictions': True,
        'show_feature_importance': True,
        'save_plots': True
    }
}

def get_model_path(model_id: str) -> str:
    """
    モデルの保存パスを取得

    Args:
        model_id: モデルの識別子

    Returns:
        モデルの保存ディレクトリパス
    """
    base_dir = '../models'
    return os.path.join(base_dir, model_id)

def create_model_config(
    model_id: str,
    features: list,
    training_period: str,
    performance: Dict[str, float]
) -> Dict[str, Any]:
    """
    モデル設定ファイルの内容を生成

    Args:
        model_id: モデルの識別子
        features: 使用した特徴量
        training_period: 学習期間
        performance: パフォーマンス指標

    Returns:
        設定ファイルの内容
    """
    return {
        'model_id': model_id,
        'model_type': 'lstm',
        'features': features,
        'training_period': training_period,
        'performance_metrics': performance,
        'model_params': MODEL_CONFIG
    }