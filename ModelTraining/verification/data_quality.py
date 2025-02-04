"""
data_quality.py

データの品質を検証するモジュール。
欠損値、異常値、データの整合性などをチェック。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta
import json

class DataQualityChecker:
    """データ品質検証クラス"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 検証設定ファイルのパス
        """
        self.logger = logging.getLogger(__name__)
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def check_missing_values(
        self,
        df: pd.DataFrame,
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        欠損値のチェック

        Args:
            df: 検証対象のデータ
            threshold: 許容される欠損値の割合

        Returns:
            検証結果
        """
        total_rows = len(df)
        missing_stats = {}
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_ratio = missing_count / total_rows
            
            missing_stats[column] = {
                'missing_count': int(missing_count),
                'missing_ratio': float(missing_ratio),
                'exceeds_threshold': missing_ratio > threshold
            }
        
        return {
            'total_rows': total_rows,
            'threshold': threshold,
            'column_stats': missing_stats,
            'has_issues': any(
                stat['exceeds_threshold']
                for stat in missing_stats.values()
            )
        }

    def check_data_types(
        self,
        df: pd.DataFrame,
        expected_types: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        データ型の検証

        Args:
            df: 検証対象のデータ
            expected_types: 期待されるデータ型

        Returns:
            検証結果
        """
        type_issues = {}
        
        for column, expected_type in expected_types.items():
            if column not in df.columns:
                type_issues[column] = {
                    'error': 'Column not found',
                    'expected_type': expected_type,
                    'actual_type': None
                }
                continue
            
            actual_type = str(df[column].dtype)
            if actual_type != expected_type:
                type_issues[column] = {
                    'error': 'Type mismatch',
                    'expected_type': expected_type,
                    'actual_type': actual_type
                }
        
        return {
            'type_issues': type_issues,
            'has_issues': len(type_issues) > 0
        }

    def check_value_ranges(
        self,
        df: pd.DataFrame,
        range_rules: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        値の範囲チェック

        Args:
            df: 検証対象のデータ
            range_rules: カラムごとの許容範囲

        Returns:
            検証結果
        """
        range_issues = {}
        
        for column, rules in range_rules.items():
            if column not in df.columns:
                continue
                
            min_val = rules.get('min')
            max_val = rules.get('max')
            
            violations = []
            if min_val is not None:
                below_min = df[df[column] < min_val]
                if not below_min.empty:
                    violations.append({
                        'type': 'below_minimum',
                        'threshold': min_val,
                        'count': len(below_min)
                    })
            
            if max_val is not None:
                above_max = df[df[column] > max_val]
                if not above_max.empty:
                    violations.append({
                        'type': 'above_maximum',
                        'threshold': max_val,
                        'count': len(above_max)
                    })
            
            if violations:
                range_issues[column] = violations
        
        return {
            'range_issues': range_issues,
            'has_issues': len(range_issues) > 0
        }

    def check_temporal_consistency(
        self,
        df: pd.DataFrame,
        date_column: str = 'Date'
    ) -> Dict[str, Any]:
        """
        時系列の整合性チェック

        Args:
            df: 検証対象のデータ
            date_column: 日付カラムの名前

        Returns:
            検証結果
        """
        if date_column in df.columns:
            dates = pd.to_datetime(df[date_column])
        else:
            dates = pd.to_datetime(df.index)
        
        # 日付の重複チェック
        duplicates = dates.duplicated()
        duplicate_dates = dates[duplicates]
        
        # 日付の欠損チェック
        date_diff = dates.diff()
        expected_diff = pd.Timedelta(days=1)  # 想定される間隔
        gaps = date_diff[date_diff > expected_diff]
        
        # 日付の順序チェック
        is_monotonic = dates.is_monotonic_increasing
        
        return {
            'duplicate_dates': {
                'count': len(duplicate_dates),
                'dates': duplicate_dates.tolist() if not duplicate_dates.empty else []
            },
            'gaps': {
                'count': len(gaps),
                'gaps': [
                    {
                        'start': dates[i-1].isoformat(),
                        'end': dates[i].isoformat(),
                        'gap_days': gap.days
                    }
                    for i, gap in enumerate(gaps)
                    if i > 0
                ]
            },
            'is_monotonic': is_monotonic,
            'has_issues': (
                len(duplicate_dates) > 0 or
                len(gaps) > 0 or
                not is_monotonic
            )
        }

    def check_data_consistency(
        self,
        df: pd.DataFrame,
        rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        データの整合性チェック

        Args:
            df: 検証対象のデータ
            rules: 整合性チェックルール

        Returns:
            検証結果
        """
        consistency_issues = []
        
        for rule in rules:
            rule_type = rule['type']
            
            if rule_type == 'comparison':
                # カラム間の比較
                col1, col2 = rule['columns']
                operator = rule['operator']
                
                if operator == '>':
                    violations = df[~(df[col1] > df[col2])]
                elif operator == '>=':
                    violations = df[~(df[col1] >= df[col2])]
                elif operator == '<':
                    violations = df[~(df[col1] < df[col2])]
                elif operator == '<=':
                    violations = df[~(df[col1] <= df[col2])]
                elif operator == '==':
                    violations = df[~(df[col1] == df[col2])]
                
                if not violations.empty:
                    consistency_issues.append({
                        'rule': rule,
                        'violation_count': len(violations),
                        'sample_violations': violations.head().to_dict()
                    })
            
            elif rule_type == 'calculation':
                # 計算結果の検証
                result = eval(rule['formula'], {'df': df, 'np': np})
                violations = df[~result]
                
                if not violations.empty:
                    consistency_issues.append({
                        'rule': rule,
                        'violation_count': len(violations),
                        'sample_violations': violations.head().to_dict()
                    })
        
        return {
            'consistency_issues': consistency_issues,
            'has_issues': len(consistency_issues) > 0
        }

    def verify_data_quality(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        すべての品質チェックを実行

        Args:
            df: 検証対象のデータ

        Returns:
            検証結果
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': {
                'rows': len(df),
                'columns': len(df.columns)
            }
        }
        
        # 欠損値チェック
        results['missing_values'] = self.check_missing_values(
            df,
            threshold=self.config.get('missing_threshold', 0.05)
        )
        
        # データ型チェック
        results['data_types'] = self.check_data_types(
            df,
            self.config.get('expected_types', {})
        )
        
        # 値の範囲チェック
        results['value_ranges'] = self.check_value_ranges(
            df,
            self.config.get('range_rules', {})
        )
        
        # 時系列の整合性チェック
        results['temporal_consistency'] = self.check_temporal_consistency(
            df,
            self.config.get('date_column', 'Date')
        )
        
        # データの整合性チェック
        results['data_consistency'] = self.check_data_consistency(
            df,
            self.config.get('consistency_rules', [])
        )
        
        # 全体の判定
        results['has_quality_issues'] = any([
            results['missing_values']['has_issues'],
            results['data_types']['has_issues'],
            results['value_ranges']['has_issues'],
            results['temporal_consistency']['has_issues'],
            results['data_consistency']['has_issues']
        ])
        
        return results

if __name__ == "__main__":
    # ロギングの設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # 設定ファイルのパス
    config_path = "config/data_quality_config.json"
    
    # サンプルデータの生成
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.randn(len(dates)).cumsum() + 100,
        'High': np.random.randn(len(dates)).cumsum() + 102,
        'Low': np.random.randn(len(dates)).cumsum() + 98,
        'Close': np.random.randn(len(dates)).cumsum() + 101,
        'Volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # データ品質の検証
    checker = DataQualityChecker(config_path)
    results = checker.verify_data_quality(data)
    
    # 結果の出力
    print("\nData Quality Verification Results:")
    print(json.dumps(results, indent=2))