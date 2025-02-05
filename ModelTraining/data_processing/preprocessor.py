"""
株価データの前処理を行うモジュール。
欠損値処理、異常値検出、正規化などを実施。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

class DataPreprocessor:
    """データ前処理クラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.fill_methods = {
            'ffill': lambda x: x.fillna(method='ffill'),
            'bfill': lambda x: x.fillna(method='bfill'),
            'linear': lambda x: x.interpolate(method='linear'),
            'zero': lambda x: x.fillna(0)
        }

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        methods: Dict[str, str]
    ) -> pd.DataFrame:
        """
        欠損値の処理

        Args:
            df: 入力データ
            methods: カラムごとの欠損値処理方法
                    例: {'Close': 'ffill', 'Volume': 'zero'}

        Returns:
            欠損値処理済みのデータ
        """
        df = df.copy()

        for column, method in methods.items():
            if column not in df.columns:
                continue

            if method not in self.fill_methods:
                self.logger.warning(f"Unknown fill method: {method}")
                continue

            df[column] = self.fill_methods[method](df[column])

        return df

    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        n_std: float = 3.0
    ) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
        """
        異常値の検出

        Args:
            df: 入力データ
            columns: 検査対象のカラム
            n_std: 標準偏差の倍数（この値を超えるものを異常値とする）

        Returns:
            (処理済みデータ, 異常値のインデックス)
        """
        df = df.copy()
        outliers = {}

        for column in columns:
            if column not in df.columns:
                continue

            mean = df[column].mean()
            std = df[column].std()

            lower_bound = mean - n_std * std
            upper_bound = mean + n_std * std

            outliers[column] = df[
                (df[column] < lower_bound) |
                (df[column] > upper_bound)
            ].index.tolist()

            # 異常値を境界値で置換
            df.loc[df[column] < lower_bound, column] = lower_bound
            df.loc[df[column] > upper_bound, column] = upper_bound

        return df, outliers

    def normalize_data(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'minmax',
        fit: bool = True
    ) -> pd.DataFrame:
        """
        データの正規化

        Args:
            df: 入力データ
            columns: 正規化対象のカラム
            method: 正規化方法 ('minmax' or 'standard')
            fit: スケーラーを新規作成するかどうか

        Returns:
            正規化されたデータ
        """
        df = df.copy()

        for column in columns:
            if column not in df.columns:
                continue

            scaler_key = f"{method}_{column}"

            if fit or scaler_key not in self.scalers:
                if method == 'minmax':
                    self.scalers[scaler_key] = MinMaxScaler()
                elif method == 'standard':
                    self.scalers[scaler_key] = StandardScaler()
                else:
                    raise ValueError(f"Unknown normalization method: {method}")

                df[column] = self.scalers[scaler_key].fit_transform(
                    df[column].values.reshape(-1, 1)
                ).flatten()
            else:
                df[column] = self.scalers[scaler_key].transform(
                    df[column].values.reshape(-1, 1)
                ).flatten()

        return df

    def add_time_features(
        self,
        df: pd.DataFrame,
        date_column: str = 'Date'
    ) -> pd.DataFrame:
        """
        時間関連の特徴量を追加

        Args:
            df: 入力データ
            date_column: 日付カラムの名前

        Returns:
            時間特徴量が追加されたデータ
        """
        df = df.copy()

        if date_column in df.columns:
            dates = pd.to_datetime(df[date_column])
        else:
            dates = pd.to_datetime(df.index)

        # 曜日（0=月曜、6=日曜）
        df['day_of_week'] = dates.dt.dayofweek

        # 月（1-12）
        df['month'] = dates.dt.month

        # 四半期（1-4）
        df['quarter'] = dates.dt.quarter

        # 年
        df['year'] = dates.dt.year

        # 月初からの経過日数
        df['day_of_month'] = dates.dt.day

        # 年初からの経過日数
        df['day_of_year'] = dates.dt.dayofyear

        return df

    def process_data(
        self,
        df: pd.DataFrame,
        missing_methods: Dict[str, str],
        outlier_columns: List[str],
        normalize_columns: List[str],
        add_time: bool = True
    ) -> pd.DataFrame:
        """
        一連の前処理を実行

        Args:
            df: 入力データ
            missing_methods: 欠損値処理方法
            outlier_columns: 異常値検出対象カラム
            normalize_columns: 正規化対象カラム
            add_time: 時間特徴量を追加するかどうか

        Returns:
            前処理済みのデータ
        """
        # 欠損値処理
        df = self.handle_missing_values(df, missing_methods)

        # 異常値検出と処理
        df, outliers = self.detect_outliers(df, outlier_columns)
        if outliers:
            self.logger.info(f"Detected outliers: {outliers}")

        # 時間特徴量の追加
        if add_time:
            df = self.add_time_features(df)

        # データの正規化
        df = self.normalize_data(df, normalize_columns)

        return df