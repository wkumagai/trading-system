"""
ペーパートレード実行スクリプト
"""

import asyncio
import logging
from datetime import datetime
import pandas as pd
from typing import Dict, List
import os
from dotenv import load_dotenv
import json

from core.strategy_manager import StrategyManager
from execution.trade_executor import TradeExecutor
from execution.ib_executor import IBExecutor

# 環境変数の読み込み
load_dotenv()

# ロギングの設定
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'logs/paper_trade_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PaperTradingSystem:
    """ペーパートレードシステム"""
    
    def __init__(self):
        """初期化"""
        # 設定の読み込み
        self.initial_capital = float(os.getenv('INITIAL_CAPITAL', '1000000'))
        self.risk_per_trade = float(os.getenv('RISK_PER_TRADE', '0.02'))
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0.1'))
        self.symbols = os.getenv('TARGET_SYMBOLS', 'NVDA').split(',')
        
        # 戦略マネージャーの初期化
        self.strategy_manager = StrategyManager()
        
        # 取引実行システムの初期化
        self.executor = TradeExecutor(
            initial_capital=self.initial_capital,
            risk_per_trade=self.risk_per_trade,
            max_position_size=self.max_position_size,
            paper_trading=True
        )
        
        self.running = False
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0
        }

    async def start(self):
        """ペーパートレードの開始"""
        try:
            logger.info("Starting paper trading system...")
            logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
            logger.info(f"Target symbols: {self.symbols}")
            
            # 取引実行システムの開始
            if not await self.executor.start():
                logger.error("Failed to start trade executor")
                return
            
            self.running = True
            logger.info("Paper trading system started successfully")
            
            # メインループの開始
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Error starting paper trading system: {str(e)}")
            await self.stop()

    async def stop(self):
        """ペーパートレードの停止"""
        try:
            self.running = False
            await self.executor.stop()
            
            # 最終パフォーマンスレポートの生成
            self._generate_performance_report()
            
            logger.info("Paper trading system stopped")
        except Exception as e:
            logger.error(f"Error stopping paper trading system: {str(e)}")

    async def _main_loop(self):
        """メインの取引ループ"""
        while self.running:
            try:
                # 口座情報の取得と記録
                account = await self.executor.executor.get_account_summary()
                self._update_performance_metrics(account)
                
                # 現在のポジションを取得
                positions = await self.executor.get_current_positions()
                logger.info(f"Current positions: {positions}")
                
                # 各銘柄の処理
                for symbol in self.symbols:
                    # 市場データの取得
                    market_data = await self._get_market_data(symbol)
                    if market_data is None:
                        continue
                    
                    # 戦略の実行
                    signals = self.strategy_manager.execute(symbol, market_data)
                    
                    # 現在価格の取得
                    current_prices = {
                        symbol: market_data['Close'].iloc[-1]
                    }
                    
                    # 取引シグナルの実行
                    results = await self.executor.execute_signals(
                        signals,
                        current_prices
                    )
                    
                    # 取引結果の記録
                    self._record_trades(symbol, results)
                
                # パフォーマンスの定期的な保存
                self._save_performance_data()
                
                # 1分待機
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(60)

    def _update_performance_metrics(self, account: Dict):
        """
        パフォーマンス指標の更新
        Args:
            account: 口座情報
        """
        current_equity = float(account.get('NetLiquidation', {}).get('value', self.initial_capital))
        profit_loss = current_equity - self.initial_capital
        
        # ドローダウンの計算
        if current_equity > self.initial_capital:
            self.performance_metrics['current_drawdown'] = 0
        else:
            current_drawdown = (self.initial_capital - current_equity) / self.initial_capital
            self.performance_metrics['current_drawdown'] = current_drawdown
            self.performance_metrics['max_drawdown'] = max(
                self.performance_metrics['max_drawdown'],
                current_drawdown
            )
        
        logger.info(f"Current equity: ${current_equity:,.2f} (P/L: ${profit_loss:,.2f})")
        logger.info(f"Current drawdown: {self.performance_metrics['current_drawdown']:.2%}")

    def _record_trades(self, symbol: str, results: Dict):
        """
        取引結果の記録
        Args:
            symbol: 銘柄コード
            results: 取引結果
        """
        for result in results.get(symbol, []):
            if result['status'] == 'executed':
                self.trade_history.append({
                    'timestamp': result['timestamp'].isoformat(),
                    'symbol': symbol,
                    'action': result['action'],
                    'quantity': result['quantity'],
                    'price': result['price']
                })
                
                self.performance_metrics['total_trades'] += 1
                
                # 損益の記録（簡易計算）
                if result['action'] == 'SELL':
                    profit = result['quantity'] * result['price']
                    if profit > 0:
                        self.performance_metrics['winning_trades'] += 1
                        self.performance_metrics['total_profit'] += profit
                    else:
                        self.performance_metrics['losing_trades'] += 1
                        self.performance_metrics['total_loss'] += abs(profit)

    def _save_performance_data(self):
        """パフォーマンスデータの保存"""
        try:
            # トレード履歴の保存
            with open(f'logs/trade_history_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
                json.dump(self.trade_history, f, indent=2)
            
            # パフォーマンス指標の保存
            with open(f'logs/performance_metrics_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")

    def _generate_performance_report(self):
        """最終パフォーマンスレポートの生成"""
        try:
            win_rate = (self.performance_metrics['winning_trades'] / 
                       self.performance_metrics['total_trades']) if self.performance_metrics['total_trades'] > 0 else 0
            
            profit_factor = (self.performance_metrics['total_profit'] / 
                           self.performance_metrics['total_loss']) if self.performance_metrics['total_loss'] > 0 else float('inf')
            
            report = {
                'summary': {
                    'total_trades': self.performance_metrics['total_trades'],
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'max_drawdown': self.performance_metrics['max_drawdown']
                },
                'trade_history': self.trade_history,
                'final_metrics': self.performance_metrics
            }
            
            # レポートの保存
            with open(f'logs/final_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info("Final performance report generated")
            logger.info(f"Win rate: {win_rate:.2%}")
            logger.info(f"Profit factor: {profit_factor:.2f}")
            logger.info(f"Max drawdown: {self.performance_metrics['max_drawdown']:.2%}")
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")

    async def _get_market_data(self, symbol: str) -> pd.DataFrame:
        """
        市場データの取得
        Args:
            symbol: 銘柄コード
        Returns:
            市場データ
        """
        try:
            # IBKRから市場データを取得
            contract = self.executor.executor.create_contract(symbol)
            bars = self.executor.executor.ib.reqHistoricalData(
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
            
            # DataFrameに変換
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
    # ペーパートレードシステムの初期化
    system = PaperTradingSystem()
    
    try:
        # システムの開始
        await system.start()
        
        # Ctrl+Cで停止するまで実行
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down paper trading system...")
        await system.stop()

if __name__ == "__main__":
    # イベントループの実行
    asyncio.run(main())