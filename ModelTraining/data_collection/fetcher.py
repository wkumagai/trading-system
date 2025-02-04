"""
fetcher.py

Alpha Vantage APIを使用して株価データを取得するモジュール。
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import os
from typing import Optional, Dict, Any

from ..training.config import ALPHA_VANTAGE_CONFIG

class AlphaVantageClient:
    """Alpha Vantage APIクライアント"""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: str, requests_per_minute: int = 5):
        """
        Args:
            api_key: Alpha Vantage APIキー
            requests_per_minute: 1分あたりのリクエスト制限
        """
        self.api_key = api_key
        self.requests_per_minute = requests_per_minute
        self.last_request_time = None
        self.logger = logging.getLogger(__name__)

    def _wait_for_rate_limit(self):
        """レート制限に基づいて待機"""
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            wait_time = (60.0 / self.requests_per_minute) - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
        self.last_request_time = time.time()

    def get_intraday_data(
        self,
        symbol: str,
        interval: str = "1min",
        output_size: str = "compact"
    ) -> pd.DataFrame:
        """
        株価の時系列データを取得

        Args:
            symbol: 銘柄コード
            interval: データ間隔
            output_size: データサイズ（compact/full）

        Returns:
            取得したデータ
        """
        self._wait_for_rate_limit()
        
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": output_size,
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            # データの抽出
            time_series_key = f"Time Series ({interval})"
            if time_series_key not in data:
                self.logger.error(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # JSONデータをDataFrameに変換
            df = pd.DataFrame.from_dict(
                data[time_series_key],
                orient='index'
            )
            
            # カラム名の整理
            df.columns = [col.split(". ")[1] for col in df.columns]
            df.columns = [col.lower() for col in df.columns]
            
            # データ型の変換
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # インデックスを日時型に変換
            df.index = pd.to_datetime(df.index)
            
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()

class MarketDataFetcher:
    """株価データ取得クラス"""
    
    def __init__(self, cache_dir: str = "../data/raw"):
        """
        Args:
            cache_dir: キャッシュディレクトリ
        """
        self.client = AlphaVantageClient(
            api_key=ALPHA_VANTAGE_CONFIG['api_key'],
            requests_per_minute=ALPHA_VANTAGE_CONFIG['requests_per_minute']
        )
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
        # キャッシュディレクトリの作成
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_data(
        self,
        symbol: str = ALPHA_VANTAGE_CONFIG['symbol'],
        interval: str = ALPHA_VANTAGE_CONFIG['interval']
    ) -> pd.DataFrame:
        """
        株価データを取得

        Args:
            symbol: 銘柄コード
            interval: データ間隔

        Returns:
            取得したデータ
        """
        # キャッシュファイルのパス
        cache_file = os.path.join(
            self.cache_dir,
            f"{symbol}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        
        # キャッシュの確認
        if os.path.exists(cache_file):
            self.logger.info(f"Loading cached data from {cache_file}")
            return pd.read_csv(
                cache_file,
                index_col=0,
                parse_dates=True
            )
        
        # データの取得
        df = self.client.get_intraday_data(symbol, interval)
        if df.empty:
            return df

        # テクニカル指標の計算（ローカルで計算）
        # SMA
        df['sma'] = df['close'].rolling(window=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # データの保存
        df.to_csv(cache_file)
        self.logger.info(f"Data saved to {cache_file}")
        
        return df

if __name__ == "__main__":
    # ロギングの設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # データ取得の実行
    fetcher = MarketDataFetcher()
    df = fetcher.fetch_data()
    
    if not df.empty:
        print("\nFetched data sample:")
        print(df.head())
        print("\nColumns:", df.columns.tolist())
        print("\nShape:", df.shape)