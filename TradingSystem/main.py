"""
メイン実行スクリプト
戦略の実行とIBKRでの取引を管理
"""

import asyncio
import logging
from datetime import datetime
import pandas as pd
from typing import Dict, List
import os
from dotenv import load_dotenv

from core.strategy_manager import StrategyManager
from execution.trade_executor import TradeExecutor
from execution.ib_executor import IBExecutor

# 環境変数の読み込み
load_dotenv()

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSystem:
    """取引システムのメインクラス"""
    
    def __init__(
        self,
        symbols: List[str],
        initial_capital: float = 1000000,
        paper_trading: bool = True
    ):
        """
        初期化
        Args:
            symbols: 取引対象の銘柄リスト
            initial_capital: 初期資金
            paper_trading: ペーパートレードモードかどうか
        """
        self.symbols = symbols
        self.paper_trading = paper_trading
        
        # 戦略マネージャーの初期化
        self.strategy_manager = StrategyManager()
        
        # 取引実行システムの初期化
        self.executor = TradeExecutor(
            initial_capital=initial_capital,
            paper_trading=paper_trading
        )
        
        self.running = False

    async def start(self):
        """取引システムの開始"""
        try:
            logger.info("Starting trading system...")
            
            # 取引実行システムの開始
            if not await self.executor.start():
                logger.error("Failed to start trade executor")
                return
            
            self.running = True
            logger.info("Trading system started successfully")
            
            # メインループの開始
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Error starting trading system: {str(e)}")
            await self.stop()

    async def stop(self):
        """取引システムの停止"""
        try:
            self.running = False
            await self.executor.stop()
            logger.info("Trading system stopped")
        except Exception as e:
            logger.error(f"Error stopping trading system: {str(e)}")

    async def _main_loop(self):
        """メインの取引ループ"""
        while self.running:
            try:
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
                    
                    # 結果のログ出力
                    for symbol, trades in results.items():
                        for trade in trades:
                            logger.info(f"Trade executed: {trade}")
                
                # 1分待機
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(60)  # エラー時も1分待機

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
    # 取引対象の銘柄
    symbols = ['NVDA']
    
    # 取引システムの初期化
    system = TradingSystem(
        symbols=symbols,
        initial_capital=1000000,  # 100万ドル
        paper_trading=True  # ペーパートレードモード
    )
    
    try:
        # システムの開始
        await system.start()
        
        # Ctrl+Cで停止するまで実行
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await system.stop()

if __name__ == "__main__":
    # イベントループの実行
    asyncio.run(main())