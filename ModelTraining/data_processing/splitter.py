"""
splitter.py

時系列データの分割を管理するモジュール。
訓練データと評価データの分割を時系列の特性を考慮して行う。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging

class TimeSeriesSplitter:
    """時系列データ分割クラス"""

    def __init__(
        self,
        train_period: str = '2Y',
        validation_period: str = '6M',
        step: str = '3M',
        min_samples: int = 252  # 1年分の取引日数
    ):
        """
        Args:
            train_period: 訓練期間 (例: '2Y' = 2年)
            validation_period: 検証期間 (例: '6M' = 6ヶ月)
            step: スライド幅 (例: '3M' = 3ヶ月)
            min_samples: 最小サンプル数
        """
        self.train_period = pd.Timedelta(train_period)
        self.validation_period = pd.Timedelta(validation_period)
        self.step = pd.Timedelta(step)
        self.min_samples = min_samples
        self.logger = logging.getLogger(__name__)

    def create_windows(
        self,
        data: pd.DataFrame,
        date_column: str = 'Date'
    ) -> List[Dict[str, Tuple[datetime, datetime]]]:
        """
        時系列ウィンドウを生成

        Args:
            data: 入力データ
            date_column: 日付カラムの名前

        Returns:
            各ウィンドウの開始日と終了日を含む辞書のリスト
        """
        if date_column in data.columns:
            dates = pd.to_datetime(data[date_column])
        else:
            dates = pd.to_datetime(data.index)

        start_date = dates.min()
        end_date = dates.max()

        windows = []
        current_start = start_date

        while current_start + self.train_period + self.validation_period <= end_date:
            train_end = current_start + self.train_period
            validation_end = train_end + self.validation_period

            # 訓練データと検証データのサンプル数をチェック
            train_mask = (dates >= current_start) & (dates < train_end)
            valid_mask = (dates >= train_end) & (dates < validation_end)

            if (train_mask.sum() >= self.min_samples and
                valid_mask.sum() >= self.min_samples // 4):  # 検証期間は訓練期間の1/4以上

                windows.append({
                    'train': (current_start, train_end),
                    'validation': (train_end, validation_end)
                })

            current_start += self.step

        self.logger.info(f"Created {len(windows)} time windows")
        return windows

    def split_data(
        self,
        data: pd.DataFrame,
        window: Dict[str, Tuple[datetime, datetime]],
        date_column: str = 'Date'
    ) -> Dict[str, pd.DataFrame]:
        """
        指定されたウィンドウでデータを分割

        Args:
            data: 入力データ
            window: 分割ウィンドウ情報
            date_column: 日付カラムの名前

        Returns:
            訓練データと検証データを含む辞書
        """
        if date_column in data.columns:
            dates = pd.to_datetime(data[date_column])
        else:
            dates = pd.to_datetime(data.index)

        train_start, train_end = window['train']
        valid_start, valid_end = window['validation']

        train_mask = (dates >= train_start) & (dates < train_end)
        valid_mask = (dates >= valid_start) & (dates < valid_end)

        return {
            'train': data[train_mask].copy(),
            'validation': data[valid_mask].copy()
        }

    def get_market_phase(
        self,
        data: pd.DataFrame,
        window_size: int = 20,
        threshold: float = 0.02
    ) -> str:
        """
        市場フェーズを判定

        Args:
            data: 価格データ
            window_size: トレンド判定の窓サイズ
            threshold: トレンド判定の閾値

        Returns:
            市場フェーズ ('trend' or 'range')
        """
        if 'Close' not in data.columns:
            raise ValueError("Price data must contain 'Close' column")

        # 移動平均の計算
        ma = data['Close'].rolling(window=window_size).mean()

        # トレンドの強さを計算
        trend_strength = abs(data['Close'] - ma) / ma

        # 平均トレンド強度が閾値を超えていればトレンド相場
        if trend_strength.mean() > threshold:
            return 'trend'
        else:
            return 'range'