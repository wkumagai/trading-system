"""
live.py

リアルタイムデータを使用したモデル検証を行うモジュール。
学習済みモデルの性能をリアルタイムで検証する。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable
import logging
from datetime import datetime, timedelta
import json
import os
import time
from queue import Queue
import threading

from ..data_collection.real_time import RealTimeDataCollector
from ..evaluation.metrics.accuracy import PredictionMetrics
from ..evaluation.metrics.trading import TradingMetrics

class LiveVerification:
    """リアルタイム検証クラス"""
    
    def __init__(
        self,
        config_path: str,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Args:
            config_path: 検証設定ファイルのパス
            callback: 検証結果を受け取るコールバック関数
        """
        self.logger = logging.getLogger(__name__)
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.callback = callback
        self.is_running = False
        self.verification_thread = None
        self.results_queue = Queue()
        
        # リアルタイムデータコレクターの初期化
        self.data_collector = RealTimeDataCollector(
            symbols=self.config['symbols'],
            buffer_size=self.config.get('buffer_size', 1000)
        )

    def start_verification(
        self,
        model,
        api_key: str,
        interval: int = 60  # 検証間隔（秒）
    ) -> None:
        """
        リアルタイム検証の開始

        Args:
            model: 検証対象のモデル
            api_key: APIキー
            interval: 検証間隔（秒）
        """
        self.is_running = True
        self.model = model
        
        # データ収集の開始
        self.data_collector.start_streaming(api_key)
        
        # 検証スレッドの開始
        self.verification_thread = threading.Thread(
            target=self._verification_loop,
            args=(interval,)
        )
        self.verification_thread.daemon = True
        self.verification_thread.start()
        
        self.logger.info("Live verification started")

    def stop_verification(self) -> None:
        """リアルタイム検証の停止"""
        self.is_running = False
        self.data_collector.stop_streaming()
        
        if self.verification_thread:
            self.verification_thread.join()
        
        self.logger.info("Live verification stopped")

    def _verification_loop(self, interval: int) -> None:
        """
        検証ループの実行

        Args:
            interval: 検証間隔（秒）
        """
        while self.is_running:
            try:
                # 各銘柄の検証
                for symbol in self.config['symbols']:
                    # バッファデータの取得
                    data = self.data_collector.get_buffer_data(symbol)
                    if data.empty:
                        continue
                    
                    # 予測の実行
                    predictions = self.model.predict(data)
                    
                    # 検証結果の計算
                    result = self._calculate_verification_metrics(
                        symbol,
                        data,
                        predictions
                    )
                    
                    # 結果のキューへの追加
                    self.results_queue.put(result)
                    
                    # コールバックの実行
                    if self.callback:
                        self.callback(result)
                
                # 指定間隔待機
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in verification loop: {str(e)}")
                continue

    def _calculate_verification_metrics(
        self,
        symbol: str,
        data: pd.DataFrame,
        predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        検証指標の計算

        Args:
            symbol: 銘柄コード
            data: 市場データ
            predictions: モデルの予測値

        Returns:
            検証結果
        """
        # 予測精度の計算
        if 'Target' in data.columns:
            accuracy_metrics = PredictionMetrics.calculate_classification_metrics(
                data['Target'].values,
                predictions
            )
        else:
            accuracy_metrics = {}
        
        # 取引指標の計算
        returns = data['Close'].pct_change().values
        trading_metrics = TradingMetrics.calculate_returns_metrics(
            returns,
            risk_free_rate=self.config.get('risk_free_rate', 0.0)
        )
        
        risk_metrics = TradingMetrics.calculate_risk_metrics(
            returns,
            data['Close'].values
        )
        
        efficiency_metrics = TradingMetrics.calculate_trading_efficiency(
            returns,
            predictions
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'accuracy_metrics': accuracy_metrics,
            'trading_metrics': trading_metrics,
            'risk_metrics': risk_metrics,
            'efficiency_metrics': efficiency_metrics,
            'latest_price': data['Close'].iloc[-1],
            'latest_prediction': predictions[-1]
        }

    def get_latest_results(self) -> Optional[Dict[str, Any]]:
        """
        最新の検証結果を取得

        Returns:
            最新の検証結果
        """
        if self.results_queue.empty():
            return None
        return self.results_queue.get()

    def save_results(
        self,
        output_dir: str,
        prefix: str = "live_verification"
    ) -> None:
        """
        検証結果の保存

        Args:
            output_dir: 出力ディレクトリ
            prefix: ファイル名のプレフィックス
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        results = []
        while not self.results_queue.empty():
            results.append(self.results_queue.get())
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")

if __name__ == "__main__":
    # ロギングの設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # コールバック関数の定義
    def print_result(result):
        print(f"\nNew result for {result['symbol']}:")
        print(f"Latest price: {result['latest_price']}")
        print(f"Latest prediction: {result['latest_prediction']}")
    
    # 設定ファイルのパス
    config_path = "config/live_verification_config.json"
    
    # モックモデル
    class MockModel:
        def predict(self, data):
            return np.random.choice([0, 1], len(data))
    
    # 検証の実行
    verifier = LiveVerification(config_path, callback=print_result)
    verifier.start_verification(
        MockModel(),
        api_key="your_api_key_here",
        interval=60
    )
    
    # 一定時間実行
    try:
        time.sleep(3600)  # 1時間実行
    finally:
        verifier.stop_verification()
        verifier.save_results('results')