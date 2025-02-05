"""
取引実行を管理するモジュール
"""

import logging
from typing import Dict, Optional, List, Union
from datetime import datetime
import asyncio
import pandas as pd
from .ib_executor import IBExecutor

class TradeExecutor:
    """取引実行管理クラス"""
    
    def __init__(
        self,
        initial_capital: float,
        risk_per_trade: float = 0.02,  # 1取引あたりのリスク（資金の2%）
        max_position_size: float = 0.1,  # 最大ポジションサイズ（資金の10%）
        paper_trading: bool = True
    ):
        """
        初期化
        Args:
            initial_capital: 初期資金
            risk_per_trade: 1取引あたりのリスク
            max_position_size: 最大ポジションサイズ
            paper_trading: ペーパートレードモードかどうか
        """
        self.logger = logging.getLogger(__name__)
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.paper_trading = paper_trading
        
        # IBKRエグゼキューターの初期化
        self.executor = IBExecutor(paper_trading=paper_trading)
        self.positions = {}
        self.orders = {}

    async def start(self) -> bool:
        """
        取引実行システムを開始
        Returns:
            開始成功したかどうか
        """
        try:
            # IBKRに接続
            connected = await self.executor.connect()
            if not connected:
                return False

            # 現在のポジションを取得
            positions = await self.executor.get_positions()
            self.positions = {
                pos['symbol']: pos
                for pos in positions
            }

            self.logger.info("Trade executor started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start trade executor: {str(e)}")
            return False

    async def stop(self):
        """取引実行システムを停止"""
        try:
            await self.executor.disconnect()
            self.logger.info("Trade executor stopped")
        except Exception as e:
            self.logger.error(f"Error stopping trade executor: {str(e)}")

    async def execute_signals(
        self,
        signals: Dict[str, pd.DataFrame],
        prices: Dict[str, float]
    ) -> Dict[str, List[Dict]]:
        """
        取引シグナルを実行
        Args:
            signals: 銘柄ごとの取引シグナル
            prices: 銘柄ごとの現在価格
        Returns:
            実行結果
        """
        results = {}
        
        for symbol, signal_df in signals.items():
            if symbol not in prices:
                continue

            current_price = prices[symbol]
            latest_signal = signal_df.iloc[-1]
            
            # ポジションサイズの計算
            size = self._calculate_position_size(
                symbol,
                current_price,
                latest_signal['signal']
            )
            
            if size == 0:
                continue

            # 注文の実行
            try:
                if size > 0:
                    trade = await self.executor.place_market_order(
                        symbol=symbol,
                        quantity=size,
                        action='BUY'
                    )
                else:
                    trade = await self.executor.place_market_order(
                        symbol=symbol,
                        quantity=abs(size),
                        action='SELL'
                    )
                
                if trade:
                    results[symbol] = [{
                        'timestamp': datetime.now(),
                        'action': 'BUY' if size > 0 else 'SELL',
                        'quantity': abs(size),
                        'price': current_price,
                        'status': 'executed'
                    }]
                    
            except Exception as e:
                self.logger.error(f"Error executing trade for {symbol}: {str(e)}")
                results[symbol] = [{
                    'timestamp': datetime.now(),
                    'error': str(e),
                    'status': 'failed'
                }]

        return results

    def _calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        signal: float
    ) -> int:
        """
        ポジションサイズの計算
        Args:
            symbol: 銘柄コード
            current_price: 現在価格
            signal: 取引シグナル
        Returns:
            取引数量（正：買い、負：売り）
        """
        try:
            # 口座情報の取得
            account = asyncio.run(self.executor.get_account_summary())
            current_capital = float(account.get('NetLiquidation', {}).get('value', self.initial_capital))
            
            # リスクに基づく取引数量の計算
            risk_amount = current_capital * self.risk_per_trade
            max_amount = current_capital * self.max_position_size
            
            # 現在のポジションを考慮
            current_position = self.positions.get(symbol, {}).get('position', 0)
            
            # シグナルに基づく取引数量の決定
            if signal > 0:  # 買いシグナル
                if current_position >= 0:
                    # 新規または追加の買い
                    max_additional = int(max_amount / current_price) - current_position
                    size = min(
                        int(risk_amount / current_price),
                        max_additional
                    )
                else:
                    # 売りポジションの解消
                    size = abs(current_position)
            elif signal < 0:  # 売りシグナル
                if current_position <= 0:
                    # 新規または追加の売り
                    max_additional = int(max_amount / current_price) + current_position
                    size = -min(
                        int(risk_amount / current_price),
                        max_additional
                    )
                else:
                    # 買いポジションの解消
                    size = -current_position
            else:
                size = 0
            
            return size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0

    async def get_current_positions(self) -> Dict[str, Dict]:
        """
        現在のポジション情報を取得
        Returns:
            ポジション情報
        """
        try:
            positions = await self.executor.get_positions()
            self.positions = {
                pos['symbol']: pos
                for pos in positions
            }
            return self.positions
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return {}