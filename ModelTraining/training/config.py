"""
config.py

モデル学習の設定パラメータ
"""

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
    'START_DATE': '2025-01-01',
    
    # Model Settings
    'MODEL_TYPE': 'lstm',
    'SEQUENCE_LENGTH': 10,
    'PREDICTION_TARGET': 1
}

# Model Parameters
MODEL_PARAMS = {
    'lstm': {
        'sequence_length': 10,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001
    }
}

# Data Processing
FEATURE_COLUMNS = [
    'Close',
    'Volume',
    'RSI',
    'MACD',
    'Signal_Line',
    'BB_Upper',
    'BB_Lower',
    'Momentum'
]

# Training Settings
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Early Stopping
EARLY_STOPPING = {
    'patience': 10,
    'min_delta': 0.001
}

def create_directories():
    """必要なディレクトリを作成"""
    import os
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