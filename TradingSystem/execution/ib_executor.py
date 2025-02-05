"""
Interactive Brokersを使用した注文執行を管理するモジュール
"""

import logging
from typing import Dict, Optional, List, Union
from datetime import datetime
import asyncio
from ib_insync import *
import pandas as pd

class IBExecutor:
    """Interactive Brokers執行クラス"""
    
    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 7497,  # TWS: 7497, IB Gateway: 4001
        client_id: int = 1,
        paper_trading: bool = True
    ):
        """
        初期化
        Args:
            host: IBKRサーバーのホスト
            port: IBKRサーバーのポート
            client_id: クライアントID
            paper_trading: ペーパートレードモードかどうか
        """
        self.logger = logging.getLogger(__name__)
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.paper_trading = paper_trading
        self.connected = False

    async def connect(self) -> bool:
        """
        IBKRサーバーに接続
        Returns:
            接続成功したかどうか
        """
        try:
            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id
            )
            self.connected = True
            self.logger.info("Successfully connected to IBKR")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to IBKR: {str(e)}")
            return False

    async def disconnect(self):
        """IBKRサーバーから切断"""
        if self.connected:
            await self.ib.disconnectAsync()
            self.connected = False
            self.logger.info("Disconnected from IBKR")

    def create_contract(
        self,
        symbol: str,
        sec_type: str = 'STK',
        exchange: str = 'SMART',
        currency: str = 'USD'
    ) -> Contract:
        """
        取引対象の契約を作成
        Args:
            symbol: 銘柄コード
            sec_type: 証券タイプ
            exchange: 取引所
            currency: 通貨
        Returns:
            契約オブジェクト
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        return contract

    async def place_market_order(
        self,
        symbol: str,
        quantity: int,
        action: str = 'BUY'
    ) -> Optional[Trade]:
        """
        成行注文を発注
        Args:
            symbol: 銘柄コード
            quantity: 数量（正：買い、負：売り）
            action: 'BUY' or 'SELL'
        Returns:
            注文オブジェクト
        """
        if not self.connected:
            self.logger.error("Not connected to IBKR")
            return None

        try:
            contract = self.create_contract(symbol)
            order = MarketOrder(action, abs(quantity))
            trade = self.ib.placeOrder(contract, order)
            self.logger.info(f"Placed market order: {symbol} {action} {quantity}")
            return trade
        except Exception as e:
            self.logger.error(f"Failed to place market order: {str(e)}")
            return None

    async def place_limit_order(
        self,
        symbol: str,
        quantity: int,
        limit_price: float,
        action: str = 'BUY'
    ) -> Optional[Trade]:
        """
        指値注文を発注
        Args:
            symbol: 銘柄コード
            quantity: 数量
            limit_price: 指値価格
            action: 'BUY' or 'SELL'
        Returns:
            注文オブジェクト
        """
        if not self.connected:
            self.logger.error("Not connected to IBKR")
            return None

        try:
            contract = self.create_contract(symbol)
            order = LimitOrder(action, abs(quantity), limit_price)
            trade = self.ib.placeOrder(contract, order)
            self.logger.info(f"Placed limit order: {symbol} {action} {quantity} @ {limit_price}")
            return trade
        except Exception as e:
            self.logger.error(f"Failed to place limit order: {str(e)}")
            return None

    async def get_positions(self) -> List[Dict]:
        """
        現在のポジションを取得
        Returns:
            ポジション情報のリスト
        """
        if not self.connected:
            self.logger.error("Not connected to IBKR")
            return []

        try:
            positions = self.ib.positions()
            return [
                {
                    'symbol': pos.contract.symbol,
                    'position': pos.position,
                    'avg_cost': pos.avgCost
                }
                for pos in positions
            ]
        except Exception as e:
            self.logger.error(f"Failed to get positions: {str(e)}")
            return []

    async def get_account_summary(self) -> Dict:
        """
        口座情報を取得
        Returns:
            口座情報の辞書
        """
        if not self.connected:
            self.logger.error("Not connected to IBKR")
            return {}

        try:
            account_values = self.ib.accountSummary()
            summary = {}
            for av in account_values:
                summary[av.tag] = {
                    'value': av.value,
                    'currency': av.currency
                }
            return summary
        except Exception as e:
            self.logger.error(f"Failed to get account summary: {str(e)}")
            return {}

    async def cancel_order(self, order_id: int) -> bool:
        """
        注文をキャンセル
        Args:
            order_id: 注文ID
        Returns:
            キャンセル成功したかどうか
        """
        if not self.connected:
            self.logger.error("Not connected to IBKR")
            return False

        try:
            trade = self.ib.trades().get(order_id)
            if trade:
                self.ib.cancelOrder(trade.order)
                self.logger.info(f"Cancelled order: {order_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {str(e)}")
            return False

    def run(self):
        """IBKRイベントループを実行"""
        self.ib.run()