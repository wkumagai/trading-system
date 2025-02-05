"""
Tradingシステムの設定ファイル
"""

import os
from datetime import datetime, timedelta

# Environment Variables
US_STOCK_API_KEY = os.getenv('US_STOCK_API_KEY')
JP_STOCK_API_KEY = os.getenv('JP_STOCK_API_KEY')
LLM_API_KEY = os.getenv('LLM_API_KEY')
IB_ACCOUNT = os.getenv('IB_ACCOUNT')
IB_PORT = int(os.getenv('IB_PORT', '7497'))
IB_HOST = os.getenv('IB_HOST', '127.0.0.1')
ENV_MODE = os.getenv('ENV_MODE', 'development')

# Trading Settings
STOCK_MARKET = 'US'  # 'US' or 'JP'
PAPER_TRADING = True  # True: ペーパートレード, False: 実取引
INITIAL_CAPITAL = 1000000  # 初期資金（ドル）

# Target Symbols
US_SYMBOLS = ['NVDA', 'AAPL', 'MSFT', 'GOOGL']
JP_SYMBOLS = ['7203.T', '9984.T', '6758.T', '7974.T']  # トヨタ,ソフトバンクG,ソニー,任天堂

# Risk Management
RISK_PER_TRADE = 0.02      # 1取引あたりのリスク（資金に対する割合）
MAX_POSITION_SIZE = 0.1    # 最大ポジションサイズ（資金に対する割合）
STOP_LOSS_PCT = 0.03      # ストップロス（3%）
TAKE_PROFIT_PCT = 0.06    # 利益確定（6%）

# Data Settings
DATA_INTERVAL = '1min'
START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1年前から

# Alert Thresholds
ALERT_THRESHOLDS = {
    'drawdown': -0.03,     # 3%のドローダウンでアラート
    'profit': 0.05,        # 5%の利益でアラート
    'volume_spike': 2.5    # 平均出来高の2.5倍でアラート
}

# Strategy Parameters
STRATEGY_PARAMS = {
    'moving_average': {
        'short_windows': [5, 10, 15],
        'long_windows': [20, 30, 50]
    },
    'lstm': {
        'sequence_lengths': [10, 20, 30],
        'epochs': 50,
        'batch_size': 32
    }
}

# Feature Settings
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

# Directory Settings
REPORT_DIR = 'reports'
LOG_DIR = 'logs'
MODEL_SAVE_DIR = 'models'

# Logging Settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'

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