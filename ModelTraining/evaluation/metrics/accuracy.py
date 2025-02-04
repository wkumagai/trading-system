"""
accuracy.py

モデルの予測精度に関する評価指標を計算するモジュール。
分類問題（方向予測）と回帰問題（価格予測）の両方に対応。
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

class PredictionMetrics:
    """予測精度の評価指標を計算するクラス"""
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        分類問題（方向予測）の評価指標を計算

        Args:
            y_true: 実際の値
            y_pred: 予測値

        Returns:
            評価指標の辞書
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary')
        }
        
        # 混同行列の計算
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 追加の指標
        metrics.update({
            'true_negative_rate': tn / (tn + fp),  # 特異度
            'false_positive_rate': fp / (fp + tn),
            'false_negative_rate': fn / (fn + tp),
            'true_positive_rate': tp / (tp + fn)   # 感度
        })
        
        return metrics

    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        回帰問題（価格予測）の評価指標を計算

        Args:
            y_true: 実際の値
            y_pred: 予測値

        Returns:
            評価指標の辞書
        """
        # 基本的な回帰指標
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # 平均絶対パーセント誤差（MAPE）
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics['mape'] = mape
        
        # 方向的中率（Directional Accuracy）
        direction_true = np.sign(np.diff(y_true))
        direction_pred = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(direction_true == direction_pred)
        metrics['directional_accuracy'] = directional_accuracy
        
        return metrics

    @staticmethod
    def calculate_probability_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        確率予測の評価指標を計算

        Args:
            y_true: 実際の値
            y_prob: 予測確率
            threshold: 分類の閾値

        Returns:
            評価指標の辞書
        """
        # 確率を二値分類に変換
        y_pred = (y_prob >= threshold).astype(int)
        
        # 基本的な分類指標
        metrics = PredictionMetrics.calculate_classification_metrics(y_true, y_pred)
        
        # 確率に特有の指標
        # Brier Score（予測確率の二乗誤差）
        brier_score = np.mean((y_prob - y_true) ** 2)
        metrics['brier_score'] = brier_score
        
        # Log Loss（交差エントロピー）
        epsilon = 1e-15  # ゼロ除算を防ぐための小さな値
        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
        log_loss = -np.mean(
            y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
        )
        metrics['log_loss'] = log_loss
        
        return metrics

    @staticmethod
    def calculate_time_series_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizon: int = 1
    ) -> Dict[str, float]:
        """
        時系列予測の評価指標を計算

        Args:
            y_true: 実際の値
            y_pred: 予測値
            horizon: 予測ホライズン（何期先の予測か）

        Returns:
            評価指標の辞書
        """
        # 基本的な回帰指標
        metrics = PredictionMetrics.calculate_regression_metrics(y_true, y_pred)
        
        # 時系列特有の指標
        # 自己相関
        error = y_true - y_pred
        autocorr = np.corrcoef(error[:-1], error[1:])[0, 1]
        metrics['error_autocorrelation'] = autocorr
        
        # ホライズンごとの精度
        if len(y_true) >= horizon:
            horizon_mse = mean_squared_error(
                y_true[horizon:],
                y_pred[:-horizon]
            )
            metrics[f'horizon_{horizon}_mse'] = horizon_mse
        
        return metrics

if __name__ == "__main__":
    # 使用例
    # 分類問題のサンプル
    y_true_cls = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    y_pred_cls = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    
    cls_metrics = PredictionMetrics.calculate_classification_metrics(
        y_true_cls,
        y_pred_cls
    )
    print("\nClassification Metrics:")
    for metric, value in cls_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 回帰問題のサンプル
    y_true_reg = np.array([10, 11, 12, 13, 14, 15])
    y_pred_reg = np.array([10.2, 10.8, 12.3, 12.8, 13.9, 15.2])
    
    reg_metrics = PredictionMetrics.calculate_regression_metrics(
        y_true_reg,
        y_pred_reg
    )
    print("\nRegression Metrics:")
    for metric, value in reg_metrics.items():
        print(f"{metric}: {value:.4f}")