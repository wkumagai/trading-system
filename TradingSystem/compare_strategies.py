"""
複数の取引戦略を同時に実行し、パフォーマンスを比較するスクリプト
"""

import asyncio
import logging
from datetime import datetime
import pandas as pd
from typing import Dict, List
import os
from dotenv import load_dotenv
import json
from concurrent.futures import ThreadPoolExecutor

from core.strategy_manager import StrategyManager
from execution.trade_executor import TradeExecutor
from execution.ib_executor import IBExecutor
from Strategies.Technical.moving_average import MovingAverageStrategy
from Strategies.Technical.momentum import MomentumStrategy

# 環境変数の読み込み
load_dotenv()

# ロギングの設定
os.makedirs('logs/strategy_comparison', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'logs/strategy_comparison/comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StrategyComparison:
    """戦略比較クラス"""
    
    def __init__(self):
        """初期化"""
        # 設定の読み込み
        self.initial_capital = float(os.getenv('INITIAL_CAPITAL', '1000000'))
        self.risk_per_trade = float(os.getenv('RISK_PER_TRADE', '0.02'))
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0.1'))
        self.symbols = os.getenv('TARGET_SYMBOLS', 'NVDA').split(',')
        
        # 戦略の定義
        self.strategies = {
            'MA_5_20': MovingAverageStrategy(short_window=5, long_window=20),
            'MA_10_50': MovingAverageStrategy(short_window=10, long_window=50),
            'MA_20_100': MovingAverageStrategy(short_window=20, long_window=100),
            'Momentum_12': MomentumStrategy(period=12),
            'Momentum_26': MomentumStrategy(period=26)
        }
        
        # 各戦略用のExecutorを初期化
        self.executors = {
            name: TradeExecutor(
                initial_capital=self.initial_capital,
                risk_per_trade=self.risk_per_trade,
                max_position_size=self.max_position_size,
                paper_trading=True
            )
            for name in self.strategies.keys()
        }
        
        self.running = False
        self.performance_data = {
            name: {
                'trade_history': [],
                'metrics': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_profit': 0.0,
                    'total_loss': 0.0,
                    'max_drawdown': 0.0,
                    'current_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'returns': []
                }
            }
            for name in self.strategies.keys()
        }

    async def start(self):
        """戦略比較の開始"""
        try:
            logger.info("Starting strategy comparison...")
            logger.info(f"Initial capital per strategy: ${self.initial_capital:,.2f}")
            logger.info(f"Target symbols: {self.symbols}")
            logger.info(f"Comparing strategies: {list(self.strategies.keys())}")
            
            # 各戦略のExecutorを開始
            for name, executor in self.executors.items():
                if not await executor.start():
                    logger.error(f"Failed to start executor for {name}")
                    return
            
            self.running = True
            logger.info("Strategy comparison started successfully")
            
            # メインループの開始
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Error starting strategy comparison: {str(e)}")
            await self.stop()

    async def stop(self):
        """戦略比較の停止"""
        try:
            self.running = False
            
            # 各Executorの停止
            for name, executor in self.executors.items():
                await executor.stop()
            
            # 最終パフォーマンス比較レポートの生成
            self._generate_comparison_report()
            
            logger.info("Strategy comparison stopped")
        except Exception as e:
            logger.error(f"Error stopping strategy comparison: {str(e)}")

    async def _main_loop(self):
        """メインの比較ループ"""
        while self.running:
            try:
                # 各戦略の実行
                for strategy_name, strategy in self.strategies.items():
                    executor = self.executors[strategy_name]
                    
                    # 口座情報の取得と記録
                    account = await executor.executor.get_account_summary()
                    self._update_performance_metrics(strategy_name, account)
                    
                    # 各銘柄の処理
                    for symbol in self.symbols:
                        # 市場データの取得
                        market_data = await self._get_market_data(executor, symbol)
                        if market_data is None:
                            continue
                        
                        # 戦略シグナルの生成
                        signals = strategy.generate_signals(market_data)
                        
                        # 現在価格の取得
                        current_prices = {
                            symbol: market_data['Close'].iloc[-1]
                        }
                        
                        # 取引シグナルの実行
                        results = await executor.execute_signals(
                            {symbol: signals},
                            current_prices
                        )
                        
                        # 取引結果の記録
                        self._record_trades(strategy_name, symbol, results)
                
                # パフォーマンスの定期的な保存
                self._save_performance_data()
                
                # パフォーマンスの比較と表示
                self._display_performance_comparison()
                
                # 1分待機
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(60)

    def _update_performance_metrics(self, strategy_name: str, account: Dict):
        """
        パフォーマンス指標の更新
        Args:
            strategy_name: 戦略名
            account: 口座情報
        """
        current_equity = float(account.get('NetLiquidation', {}).get('value', self.initial_capital))
        profit_loss = current_equity - self.initial_capital
        
        metrics = self.performance_data[strategy_name]['metrics']
        
        # リターンの記録
        metrics['returns'].append(profit_loss / self.initial_capital)
        
        # ドローダウンの計算
        if current_equity > self.initial_capital:
            metrics['current_drawdown'] = 0
        else:
            current_drawdown = (self.initial_capital - current_equity) / self.initial_capital
            metrics['current_drawdown'] = current_drawdown
            metrics['max_drawdown'] = max(
                metrics['max_drawdown'],
                current_drawdown
            )
        
        # Sharpe Ratioの計算（年率）
        if len(metrics['returns']) > 1:
            returns = pd.Series(metrics['returns'])
            metrics['sharpe_ratio'] = (
                returns.mean() * 252 / (returns.std() * np.sqrt(252))
                if returns.std() > 0 else 0
            )
        
        logger.info(f"{strategy_name} - Current equity: ${current_equity:,.2f} (P/L: ${profit_loss:,.2f})")

    def _record_trades(self, strategy_name: str, symbol: str, results: Dict):
        """
        取引結果の記録
        Args:
            strategy_name: 戦略名
            symbol: 銘柄コード
            results: 取引結果
        """
        metrics = self.performance_data[strategy_name]['metrics']
        
        for result in results.get(symbol, []):
            if result['status'] == 'executed':
                self.performance_data[strategy_name]['trade_history'].append({
                    'timestamp': result['timestamp'].isoformat(),
                    'symbol': symbol,
                    'action': result['action'],
                    'quantity': result['quantity'],
                    'price': result['price']
                })
                
                metrics['total_trades'] += 1
                
                # 損益の記録
                if result['action'] == 'SELL':
                    profit = result['quantity'] * result['price']
                    if profit > 0:
                        metrics['winning_trades'] += 1
                        metrics['total_profit'] += profit
                    else:
                        metrics['losing_trades'] += 1
                        metrics['total_loss'] += abs(profit)

    def _save_performance_data(self):
        """パフォーマンスデータの保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 各戦略のデータを保存
            for strategy_name, data in self.performance_data.items():
                filename = f'logs/strategy_comparison/{strategy_name}_{timestamp}.json'
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")

    def _display_performance_comparison(self):
        """パフォーマンス比較の表示"""
        try:
            comparison = {}
            
            for strategy_name, data in self.performance_data.items():
                metrics = data['metrics']
                win_rate = (metrics['winning_trades'] / 
                           metrics['total_trades']) if metrics['total_trades'] > 0 else 0
                
                profit_factor = (metrics['total_profit'] / 
                               metrics['total_loss']) if metrics['total_loss'] > 0 else float('inf')
                
                comparison[strategy_name] = {
                    'total_trades': metrics['total_trades'],
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'max_drawdown': metrics['max_drawdown'],
                    'sharpe_ratio': metrics['sharpe_ratio']
                }
            
            # 比較結果の表示
            logger.info("\nStrategy Comparison:")
            for strategy_name, metrics in comparison.items():
                logger.info(f"\n{strategy_name}:")
                logger.info(f"Total trades: {metrics['total_trades']}")
                logger.info(f"Win rate: {metrics['win_rate']:.2%}")
                logger.info(f"Profit factor: {metrics['profit_factor']:.2f}")
                logger.info(f"Max drawdown: {metrics['max_drawdown']:.2%}")
                logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
            
        except Exception as e:
            logger.error(f"Error displaying performance comparison: {str(e)}")

    def _generate_comparison_report(self):
        """最終比較レポートの生成"""
        try:
            report = {
                'comparison_summary': {},
                'detailed_metrics': self.performance_data
            }
            
            # 各戦略の要約を作成
            for strategy_name, data in self.performance_data.items():
                metrics = data['metrics']
                win_rate = (metrics['winning_trades'] / 
                           metrics['total_trades']) if metrics['total_trades'] > 0 else 0
                
                profit_factor = (metrics['total_profit'] / 
                               metrics['total_loss']) if metrics['total_loss'] > 0 else float('inf')
                
                report['comparison_summary'][strategy_name] = {
                    'total_trades': metrics['total_trades'],
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'max_drawdown': metrics['max_drawdown'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'total_profit': metrics['total_profit'],
                    'total_loss': metrics['total_loss']
                }
            
            # レポートの保存
            filename = f'logs/strategy_comparison/final_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Final comparison report generated: {filename}")
            
        except Exception as e:
            logger.error(f"Error generating comparison report: {str(e)}")

    async def _get_market_data(self, executor: TradeExecutor, symbol: str) -> pd.DataFrame:
        """
        市場データの取得
        Args:
            executor: 取引実行システム
            symbol: 銘柄コード
        Returns:
            市場データ
        """
        try:
            contract = executor.executor.create_contract(symbol)
            bars = executor.executor.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='2 D',
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=True
            )
            
            if not bars:
                logger.warning(f"No market data available for {symbol}")
                return None
            
            df = pd.DataFrame({
                'Date': [bar.date for bar in bars],
                'Open': [bar.open for bar in bars],
                'High': [bar.high for bar in bars],
                'Low': [bar.low for bar in bars],
                'Close': [bar.close for bar in bars],
                'Volume': [bar.volume for bar in bars]
            })
            
            df.set_index('Date', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return None

async def main():
    """メイン関数"""
    comparison = StrategyComparison()
    
    try:
        await comparison.start()
        
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down strategy comparison...")
        await comparison.stop()

if __name__ == "__main__":
    asyncio.run(main())