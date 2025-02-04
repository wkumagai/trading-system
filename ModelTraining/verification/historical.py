"""
historical.py

過去データを使用したモデル検証を行うモジュール。
学習済みモデルの性能を時系列データで検証する。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import json
import os

from ..data_processing.splitter import TimeSeriesSplitter
from ..evaluation.metrics.accuracy import PredictionMetrics
from ..evaluation.metrics.trading import TradingMetrics

class HistoricalVerification:
    """過去データによる検証クラス"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 検証設定ファイルのパス
        """
        self.logger = logging.getLogger(__name__)
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.splitter = TimeSeriesSplitter(
            train_period=self.config.get('train_period', '2Y'),
            validation_period=self.config.get('validation_period', '6M'),
            step=self.config.get('step', '3M')
        )

    def verify_predictions(
        self,
        model,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """
        予測性能の検証

        Args:
            model: 検証対象のモデル
            data: 検証用データ
            target_column: 目的変数のカラム名

        Returns:
            検証結果
        """
        windows = self.splitter.create_windows(data)
        results = []
        
        for window in windows:
            # データの分割
            split_data = self.splitter.split_data(data, window)
            validation_data = split_data['validation']
            
            # 予測の実行
            try:
                predictions = model.predict(validation_data)
                
                # 予測精度の評価
                y_true = validation_data[target_column]
                metrics = PredictionMetrics.calculate_classification_metrics(
                    y_true,
                    predictions
                )
                
                # 市場フェーズの判定
                market_phase = self.splitter.get_market_phase(validation_data)
                
                results.append({
                    'period': f"{window['validation'][0]} to {window['validation'][1]}",
                    'market_phase': market_phase,
                    'metrics': metrics
                })
                
            except Exception as e:
                self.logger.error(f"Error in prediction: {str(e)}")
                continue
        
        return {
            'window_results': results,
            'average_metrics': self._calculate_average_metrics(results)
        }

    def verify_trading(
        self,
        model,
        data: pd.DataFrame,
        price_column: str = 'Close'
    ) -> Dict[str, Any]:
        """
        取引性能の検証

        Args:
            model: 検証対象のモデル
            data: 検証用データ
            price_column: 価格のカラム名

        Returns:
            検証結果
        """
        windows = self.splitter.create_windows(data)
        results = []
        
        for window in windows:
            # データの分割
            split_data = self.splitter.split_data(data, window)
            validation_data = split_data['validation']
            
            try:
                # シグナルの生成
                signals = model.predict(validation_data)
                
                # リターンの計算
                prices = validation_data[price_column]
                returns = prices.pct_change()
                
                # トレーディング指標の計算
                trading_metrics = TradingMetrics.calculate_returns_metrics(
                    returns.values,
                    risk_free_rate=self.config.get('risk_free_rate', 0.0)
                )
                
                risk_metrics = TradingMetrics.calculate_risk_metrics(
                    returns.values,
                    prices.values
                )
                
                efficiency_metrics = TradingMetrics.calculate_trading_efficiency(
                    returns.values,
                    signals
                )
                
                results.append({
                    'period': f"{window['validation'][0]} to {window['validation'][1]}",
                    'trading_metrics': trading_metrics,
                    'risk_metrics': risk_metrics,
                    'efficiency_metrics': efficiency_metrics
                })
                
            except Exception as e:
                self.logger.error(f"Error in trading verification: {str(e)}")
                continue
        
        return {
            'window_results': results,
            'average_metrics': self._calculate_average_trading_metrics(results)
        }

    def _calculate_average_metrics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        予測指標の平均値を計算

        Args:
            results: 各期間の結果

        Returns:
            平均指標
        """
        if not results:
            return {}
            
        metrics_sum = {}
        for result in results:
            for metric, value in result['metrics'].items():
                metrics_sum[metric] = metrics_sum.get(metric, 0) + value
        
        return {
            metric: value / len(results)
            for metric, value in metrics_sum.items()
        }

    def _calculate_average_trading_metrics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        取引指標の平均値を計算

        Args:
            results: 各期間の結果

        Returns:
            平均指標
        """
        if not results:
            return {}
            
        metrics_sum = {
            'trading': {},
            'risk': {},
            'efficiency': {}
        }
        
        for result in results:
            for metric, value in result['trading_metrics'].items():
                metrics_sum['trading'][metric] = \
                    metrics_sum['trading'].get(metric, 0) + value
            
            for metric, value in result['risk_metrics'].items():
                metrics_sum['risk'][metric] = \
                    metrics_sum['risk'].get(metric, 0) + value
            
            for metric, value in result['efficiency_metrics'].items():
                metrics_sum['efficiency'][metric] = \
                    metrics_sum['efficiency'].get(metric, 0) + value
        
        return {
            category: {
                metric: value / len(results)
                for metric, value in metrics.items()
            }
            for category, metrics in metrics_sum.items()
        }

    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: str,
        prefix: str = "verification"
    ) -> None:
        """
        検証結果の保存

        Args:
            results: 検証結果
            output_dir: 出力ディレクトリ
            prefix: ファイル名のプレフィックス
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")

if __name__ == "__main__":
    # ロギングの設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # 使用例
    # 設定ファイルのパス
    config_path = "config/verification_config.json"
    
    # 検証の実行
    verifier = HistoricalVerification(config_path)
    
    # サンプルデータの生成
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Close': np.random.randn(len(dates)).cumsum() + 100,
        'Target': np.random.choice([0, 1], len(dates))
    })
    
    # モックモデル
    class MockModel:
        def predict(self, data):
            return np.random.choice([0, 1], len(data))
    
    model = MockModel()
    
    # 予測性能の検証
    prediction_results = verifier.verify_predictions(
        model,
        data,
        'Target'
    )
    
    # 取引性能の検証
    trading_results = verifier.verify_trading(
        model,
        data
    )
    
    # 結果の保存
    verifier.save_results(
        {
            'prediction': prediction_results,
            'trading': trading_results
        },
        'results'
    )