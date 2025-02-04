# ModelTraining

株価予測のための機械学習モデル学習システム（MVP版）

## 概要

- Alpha Vantage APIを使用した株価データの取得
- LSTMを使用した価格予測モデルの学習
- 1分足データでの予測（最新100件のデータを使用）

## 必要条件

- Python 3.8以上
- Alpha Vantage APIキー

## インストール

```bash
# 必要なパッケージのインストール
pip install -r requirements.txt

# 環境変数の設定
cp .env.example .env
# .envファイルを編集してAPIキーなどを設定
```

## 設定

.envファイルで以下の設定が可能：

```env
# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY=your_api_key_here

# データ設定
TARGET_SYMBOL=NVDA
DATA_INTERVAL=1min
START_DATE=2024-12-01

# モデル設定
MODEL_TYPE=lstm
SEQUENCE_LENGTH=10
PREDICTION_TARGET=1
```

## 使用方法

### 基本的な実行

```bash
# デフォルト設定での実行
python main.py

# パラメータを指定して実行
python main.py --symbol NVDA --interval 1min
```

### データ構造

```
ModelTraining/
├── data/
│   ├── raw/        # 生データ
│   ├── processed/  # 前処理済みデータ
│   └── features/   # 特徴量データ
├── models/         # 学習済みモデル
└── logs/          # ログファイル
```

## 実装済み機能

1. データ収集
- Alpha Vantage APIを使用した株価データ取得
- 1分足データの取得（最新100件）
- データのキャッシュ機能

2. 特徴量生成
- テクニカル指標の計算
- シーケンスデータの生成
- データのスケーリング

3. モデル学習
- LSTMモデルの構築
- 訓練/検証/テストデータの分割
- モデルの保存機能

## 制限事項（MVP版）

1. データ取得
- Alpha Vantage無料プランの制限（1分あたり5リクエスト）
- 1分足データは最新100件まで

2. モデル
- シンプルなLSTMモデルのみ
- 限られたデータでの学習

3. 評価
- 基本的な評価指標のみ
- リアルタイム検証なし

## 今後の展開

1. データ収集の拡張
- より長期のデータ収集
- 複数の時間枠対応
- 他のデータソースの追加

2. モデルの改善
- 複数のモデルアーキテクチャ
- ハイパーパラメータ最適化
- アンサンブル学習

3. 評価機能の強化
- より詳細なパフォーマンス分析
- バックテスト機能の追加
- リアルタイム検証の実装

## 注意事項

- このシステムはMVP（最小実行可能製品）版です
- 実際のトレーディングには十分な検証が必要です
- API制限に注意してください

## Google Colabでの実行方法

1. ファイル構成
```
Trading/
└── ModelTraining/
    └── notebooks/
        └── ModelTraining.ipynb  # Google Colab用ノートブック
```

2. 実行手順
- Google Driveに`Trading`フォルダをアップロード
- `ModelTraining.ipynb`をGoogle Colabで開く
- ノートブック内の指示に従って実行

3. 注意事項
- 初回実行時は必要なパッケージのインストールと初期設定が行われます
- 2回目以降は初期設定のセルは実行不要です
- GPUを使用する場合は、「ランタイム」→「ランタイムのタイプを変更」でGPUを選択してください
- 学習結果は`ModelTraining/models`ディレクトリに保存されます

4. トラブルシューティング
- 実行エラーが発生した場合は、「ランタイム」→「すべてのランタイムを再起動」を試してください
- パッケージのインストールに失敗した場合は、個別に`!pip install`を実行してください