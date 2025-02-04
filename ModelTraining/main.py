"""
main.py

モデル学習の実行スクリプト。
データ取得から学習までの一連の処理を実行。
"""

import os
import logging
from datetime import datetime
import argparse
from dotenv import load_dotenv

from data_collection.fetcher import MarketDataFetcher
from training.train import train_model
from training.config import DEFAULT_SETTINGS

def setup_logging(log_dir: str) -> None:
    """
    ロギングの設定

    Args:
        log_dir: ログ出力ディレクトリ
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(
                    log_dir,
                    f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                )
            )
        ]
    )

def main():
    """メイン実行関数"""
    # 環境変数の読み込み
    load_dotenv()
    
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='LSTM Model Training')
    parser.add_argument(
        '--symbol',
        default=DEFAULT_SETTINGS['TARGET_SYMBOL'],
        help='Target symbol (e.g., NVDA)'
    )
    parser.add_argument(
        '--interval',
        default=DEFAULT_SETTINGS['DATA_INTERVAL'],
        help='Data interval (e.g., 1min)'
    )
    parser.add_argument(
        '--model-type',
        default=DEFAULT_SETTINGS['MODEL_TYPE'],
        help='Model type (e.g., lstm)'
    )
    
    args = parser.parse_args()
    
    # ロギングの設定
    setup_logging(os.getenv('LOG_DIR', 'logs'))
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting model training process...")
        logger.info(f"Target Symbol: {args.symbol}")
        logger.info(f"Data Interval: {args.interval}")
        logger.info(f"Model Type: {args.model_type}")
        
        # データ取得の準備
        data_fetcher = MarketDataFetcher()
        
        # モデルIDの生成
        model_id = f"{args.model_type}_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # モデルの学習実行
        results = train_model(
            data_fetcher=data_fetcher,
            model_id=model_id
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {results['model_dir']}")
        logger.info(f"Test Loss: {results['test_loss']:.4f}")
        logger.info(f"Test MAE: {results['test_mae']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

def run_example():
    """使用例の実行"""
    main()

if __name__ == "__main__":
    main()