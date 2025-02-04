"""
real_time.py

リアルタイムの株価データを取得・管理するモジュール。
ストリーミングAPIを使用してリアルタイムデータを取得し、
モデルの検証に使用する。
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import websocket
import json
import threading
from queue import Queue

class RealTimeDataCollector:
    """リアルタイムデータ収集クラス"""
    
    def __init__(self, symbols: List[str], buffer_size: int = 1000):
        """
        Args:
            symbols: 監視する銘柄のリスト
            buffer_size: データバッファのサイズ
        """
        self.symbols = symbols
        self.buffer_size = buffer_size
        self.data_buffer = {symbol: Queue(maxsize=buffer_size) for symbol in symbols}
        self.latest_data = {}
        self.logger = logging.getLogger(__name__)
        self.ws = None
        self.is_connected = False

    def start_streaming(self, api_key: str):
        """
        データストリーミングを開始

        Args:
            api_key: API認証キー
        """
        def on_message(ws, message):
            """メッセージ受信時のコールバック"""
            try:
                data = json.loads(message)
                self._process_market_data(data)
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")

        def on_error(ws, error):
            """エラー発生時のコールバック"""
            self.logger.error(f"WebSocket error: {str(error)}")

        def on_close(ws, close_status_code, close_msg):
            """接続切断時のコールバック"""
            self.is_connected = False
            self.logger.info("WebSocket connection closed")

        def on_open(ws):
            """接続確立時のコールバック"""
            self.is_connected = True
            self.logger.info("WebSocket connection established")
            # サブスクリプションメッセージの送信
            subscribe_message = {
                "type": "subscribe",
                "symbols": self.symbols
            }
            ws.send(json.dumps(subscribe_message))

        # WebSocket接続の設定
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            f"wss://your.websocket.url?token={api_key}",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )

        # 別スレッドでWebSocket接続を開始
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def stop_streaming(self):
        """データストリーミングを停止"""
        if self.ws is not None:
            self.ws.close()
        self.is_connected = False

    def _process_market_data(self, data: Dict[str, Any]):
        """
        受信したマーケットデータの処理

        Args:
            data: 受信したデータ
        """
        try:
            symbol = data.get('symbol')
            if symbol not in self.symbols:
                return

            # データの整形
            processed_data = {
                'timestamp': datetime.fromtimestamp(data['timestamp']),
                'price': float(data['price']),
                'volume': int(data['volume'])
            }

            # バッファにデータを追加
            if self.data_buffer[symbol].full():
                self.data_buffer[symbol].get()  # 古いデータを削除
            self.data_buffer[symbol].put(processed_data)

            # 最新データの更新
            self.latest_data[symbol] = processed_data

        except Exception as e:
            self.logger.error(f"Error processing market data: {str(e)}")

    def get_latest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        指定された銘柄の最新データを取得

        Args:
            symbol: 銘柄コード

        Returns:
            最新のマーケットデータ
        """
        return self.latest_data.get(symbol)

    def get_buffer_data(self, symbol: str) -> pd.DataFrame:
        """
        指定された銘柄のバッファデータを取得

        Args:
            symbol: 銘柄コード

        Returns:
            バッファ内のデータをDataFrameとして返す
        """
        buffer_data = list(self.data_buffer[symbol].queue)
        if not buffer_data:
            return pd.DataFrame()

        df = pd.DataFrame(buffer_data)
        df.set_index('timestamp', inplace=True)
        return df

if __name__ == "__main__":
    # ロギングの設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # 使用例
    symbols = ["AAPL", "GOOGL"]
    collector = RealTimeDataCollector(symbols)
    
    # ストリーミング開始（実際のAPIキーを使用）
    collector.start_streaming("your_api_key")
    
    # 一定時間後に停止
    import time
    time.sleep(60)  # 60秒間データを収集
    
    collector.stop_streaming()
    
    # 収集したデータの確認
    for symbol in symbols:
        df = collector.get_buffer_data(symbol)
        print(f"\nData for {symbol}:")
        print(df.head())