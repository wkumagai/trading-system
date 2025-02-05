"""
Tradingシステムの設定ファイル
"""

import os
from datetime import datetime, timedelta

# Environment Variables (from .env)
STOCK_API_KEY = os.getenv('STOCK_API_KEY')
ENV_MODE = os.getenv('ENV_MODE', 'development')
LLM_API_KEY = os.getenv('LLM_API_KEY')

# Directory Settings
REPORT_DIR = 'reports'
LOG_DIR = 'logs'
MODEL_SAVE_DIR = 'models'

# Logging Settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

# Default Settings
DEFAULT_SETTINGS = {
    # Data Settings
    'TARGET_SYMBOL': 'NVDA',
    'DATA_INTERVAL': '1min',
    'START_DATE': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),  # 1年前から

    # Model Settings
    'MODEL_TYPE': 'lstm',
    'SEQUENCE_LENGTH': 10,
    'PREDICTION_TARGET': 1
}

# Application Settings
INITIAL_CAPITAL = 1000000  # 実際の運用資金に応じて調整
COMMISSION_RATE = 0.0015   # 取引手数料（0.15%）

# Strategy Parameters
STRATEGY_PARAMS = {
    "moving_average": {
        "short_windows": [5, 10, 15],
        "long_windows": [20, 30, 50]
    },
    "lstm": {
        "sequence_lengths": [10, 20, 30],
        "epochs": 50,
        "batch_size": 32
    }
}

# Risk Management
MAX_POSITION_SIZE = 0.1    # 最大ポジションサイズ（資金の10%）
STOP_LOSS_PCT = 0.03      # ストップロス（3%）
TAKE_PROFIT_PCT = 0.06    # 利益確定（6%）

# Performance Thresholds
MIN_SHARPE_RATIO = 1.0
MAX_DRAWDOWN_LIMIT = -0.10  # 最大ドローダウン制限（-10%）

# Feature Settings
FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "RSI",
    "MACD",
    "Signal_Line",
    "BB_Upper",
    "BB_Lower",
    "Momentum"
]

# Ensemble Settings
ENSEMBLE_WEIGHTS = {
    "moving_average_5_20": 0.25,
    "moving_average_10_30": 0.25,
    "lstm_seq10": 0.25,
    "lstm_seq20": 0.25
}

# Alert Settings
ALERT_THRESHOLDS = {
    "drawdown": -0.03,      # 3%のドローダウンでアラート
    "profit": 0.05,         # 5%の利益でアラート
    "volume_spike": 2.5      # 平均出来高の2.5倍でアラート
}

def create_directories():
    """必要なディレクトリを作成"""
    directories = [
        REPORT_DIR,
        LOG_DIR,
        MODEL_SAVE_DIR
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

# 起動時にディレクトリを作成
create_directories()