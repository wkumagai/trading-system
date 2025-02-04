"""
main.py

Tradingシステムのメインエントリーポイント。
"""

import os
import argparse
import pandas as pd
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta

def setup_logging():
    """ロギングの設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/trading_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )
    return logging.getLogger(__name__)

def run_trading(mode: str, **kwargs):
    """
    トレーディングシステムの実行

    Args:
        mode: 実行モード ('daily', 'backtest', or 'compare')
        kwargs: 追加パラメータ
    """
    from core.strategy_manager import StrategyManager
    from strategies.moving_average import MovingAverageStrategy
    from strategies.deep_learning import LSTMStrategy
    from evaluation.evaluator import StrategyEvaluator
    from reporting.reporter import StrategyReporter
    from core.data_manager import DataManager
    import config.config as config

    # データ取得
    symbol = kwargs.get('symbol', 'AAPL')
    start_date = kwargs.get('start_date', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    end_date = kwargs.get('end_date', datetime.now().strftime('%Y-%m-%d'))

    dm = DataManager(config)
    df = dm.fetch_market_data(symbol, start_date, end_date, interval="1min")

    if df.empty:
        logging.error("No data fetched. Exiting.")
        return

    # 戦略マネージャーの設定
    manager = StrategyManager(config)

    # 戦略の登録
    strategies = [
        MovingAverageStrategy(config, short_window=5, long_window=20),
        MovingAverageStrategy(config, short_window=10, long_window=30),
        LSTMStrategy(config, sequence_length=10),
        LSTMStrategy(config, sequence_length=20)
    ]
    manager.register_multiple_strategies(strategies)

    # モード別の処理
    if mode == 'daily':
        # 日次処理（最適な戦略で実取引）
        results = manager.run_all_strategies(df)
        evaluator = StrategyEvaluator()
        performance = evaluator.evaluate_multiple_strategies(results)
        
        # 最良の戦略を選択
        best_strategy = performance['sharpe_ratio'].idxmax()
        logging.info(f"Selected best strategy: {best_strategy}")
        
        # 選択した戦略の最新の予測に基づいて取引実行
        latest_signals = results[best_strategy]['predictions'].iloc[-1:]
        if not latest_signals.empty:
            signal = latest_signals['signal'].iloc[0]
            if signal != 0:
                logging.info(f"Executing trade with signal: {signal}")
                # TODO: 実取引の実装
    
    elif mode == 'backtest':
        # 単一戦略のバックテスト
        strategy_name = kwargs.get('strategy', 'moving_average_5_20')
        if strategy_name in manager.strategies:
            strategy = manager.strategies[strategy_name]
            result = manager._run_single_strategy(strategy_name, df)
            
            evaluator = StrategyEvaluator()
            performance = evaluator.evaluate_strategy(result['predictions'])
            
            reporter = StrategyReporter()
            reporter.generate_report(
                {strategy_name: result['predictions']},
                pd.DataFrame([performance], index=[strategy_name])
            )
    
    else:  # compare mode
        # 全戦略の比較
        results = manager.run_all_strategies(df)
        
        evaluator = StrategyEvaluator()
        performance = evaluator.evaluate_multiple_strategies(results)
        
        reporter = StrategyReporter()
        reporter.generate_report(
            {name: result['predictions'] for name, result in results.items()},
            performance,
            f"Strategy Comparison Report ({symbol})"
        )

def main():
    """メイン実行関数"""
    # 環境変数の読み込み
    load_dotenv()

    # ロギングの設定
    logger = setup_logging()

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='Trading System')
    parser.add_argument('--mode', choices=['daily', 'backtest', 'compare'],
                       default='daily', help='Operation mode')
    parser.add_argument('--symbol', default='AAPL',
                       help='Stock symbol for backtest/compare mode')
    parser.add_argument('--start-date', help='Start date for backtest/compare mode')
    parser.add_argument('--end-date', help='End date for backtest/compare mode')
    parser.add_argument('--strategy', help='Strategy name for backtest mode')

    args = parser.parse_args()

    try:
        logger.info(f"Starting Trading System (Mode: {args.mode})")
        
        kwargs = {
            'symbol': args.symbol,
            'start_date': args.start_date,
            'end_date': args.end_date,
            'strategy': args.strategy
        }

        run_trading(args.mode, **kwargs)
        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()