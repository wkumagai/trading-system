# Trading2 v2

複数の取引戦略を比較・評価できる株式取引システム

## 構造

```
Trading2_v2/
├── v1/                    # 既存の実装
├── v2/                    # 新しい実装
│   ├── strategies/        # 取引戦略
│   │   ├── base.py       # 基底戦略クラス
│   │   ├── moving_average.py
│   │   └── deep_learning.py
│   ├── core/             # コア機能
│   │   └── strategy_manager.py
│   ├── evaluation/       # 評価機能
│   │   └── evaluator.py
│   └── reporting/        # レポート機能
│       └── reporter.py
├── config/               # 設定ファイル
│   ├── v1_config.py
│   └── v2_config.py
└── main.py              # エントリーポイント
```

## 主な機能

1. 複数の取引戦略
   - 移動平均クロス戦略
   - LSTM深層学習戦略
   - 容易に新しい戦略を追加可能

2. パフォーマンス評価
   - シャープレシオ
   - 最大ドローダウン
   - 勝率
   - プロフィットファクター
   - その他の各種指標

3. 可視化・レポート
   - ポートフォリオ価値の推移
   - 戦略間の相関
   - 月次リターンヒートマップ
   - ドローダウン分析

## 使用方法

### 環境設定

1. 必要なパッケージのインストール
```bash
pip install -r requirements.txt
```

2. 環境変数の設定
```bash
cp .env.example .env
# .envファイルを編集してAPIキーなどを設定
```

### 実行方法

1. 日次処理の実行
```bash
python main.py --version v2 --mode daily
```

2. バックテストの実行
```bash
python main.py --version v2 --mode backtest --symbol AAPL --start-date 2023-01-01 --end-date 2023-12-31
```

3. 戦略比較の実行
```bash
python main.py --version v2 --mode compare --symbol AAPL --start-date 2023-01-01 --end-date 2023-12-31
```

### 新しい戦略の追加方法

1. `v2/strategies/`に新しい戦略クラスを作成
2. `BaseStrategy`を継承して必要なメソッドを実装
3. `strategy_manager.py`で新しい戦略を登録

例：
```python
from .base import BaseStrategy

class MyNewStrategy(BaseStrategy):
    @property
    def strategy_name(self):
        return "my_new_strategy"

    def create_features(self, df):
        # 特徴量生成ロジックを実装
        pass

    def train_model(self, df):
        # 学習ロジックを実装
        pass

    def predict(self, df):
        # 予測ロジックを実装
        pass
```

## 設定のカスタマイズ

`config/v2_config.py`で以下の設定が可能：

- 取引パラメータ
- リスク管理設定
- バックテスト設定
- レポート設定
- アラート閾値
- その他

## 注意事項

- バックテストのパフォーマンスは実際の取引結果を保証するものではありません
- リスク管理を適切に行い、実取引では十分な検証を行ってください
- APIの利用制限に注意してください

## 今後の展開

1. 追加予定の機能
   - より多くの戦略の実装
   - リアルタイムデータへの対応
   - 分散投資機能
   - リスク管理の強化

2. 改善予定の項目
   - パフォーマンスの最適化
   - バックテストの精緻化
   - レポート機能の拡充