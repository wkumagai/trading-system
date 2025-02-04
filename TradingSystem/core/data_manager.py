"""
data_manager.py

市場データの取得と管理を行うモジュール。
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional
import logging

class DataManager:
    """
    市場データの取得と管理を行うクラス。
    """

    def __init__(self, config):
        """
        Args:
            config: システム全体の設定値を持つモジュール
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache = {}

    def fetch_market_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        指定された銘柄の市場データを取得する。

        Args:
            symbol: 銘柄コード
            start_date: 開始日
            end_date: 終了日
            interval: データ間隔（1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo）

        Returns:
            市場データを含むDataFrame
        """
        cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
        
        if cache_key in self.cache:
            self.logger.info(f"Returning cached data for {symbol}")
            return self.cache[cache_key]

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )

            if df.empty:
                self.logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            # カラム名を標準化
            df.index.name = 'Date'
            df.columns = [col.title() for col in df.columns]

            # キャッシュに保存
            self.cache[cache_key] = df
            
            self.logger.info(f"Successfully fetched data for {symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_multiple_symbols(
        self,
        symbols: list,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> dict:
        """
        複数銘柄の市場データを取得する。

        Args:
            symbols: 銘柄コードのリスト
            start_date: 開始日
            end_date: 終了日
            interval: データ間隔

        Returns:
            銘柄コードをキー、DataFrameを値とするディクショナリ
        """
        data = {}
        for symbol in symbols:
            df = self.fetch_market_data(symbol, start_date, end_date, interval)
            if not df.empty:
                data[symbol] = df
        return data

    def clear_cache(self):
        """キャッシュをクリアする"""
        self.cache.clear()
        self.logger.info("Cache cleared")