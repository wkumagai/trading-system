# API Tests

APIキーの検証用テストスクリプト集

## ファイル構成

- `test_api.py`: Alpha Vantage APIの基本テスト
- `test_llm.py`: LLM API（Claude）の基本テスト
- `test_llm_alt.py`: LLM API（Claude）の詳細テスト（モデル指定可能）
- `simple_test.py`: シンプルな株価APIテスト
- `curl_test.py`: curlベースの株価APIテスト

## 使用方法

1. プロジェクトルートの`.env`ファイルにAPIキーが設定されていることを確認

```env
STOCK_API_KEY=your_stock_api_key
LLM_API_KEY=your_llm_api_key
```

2. テストの実行

```bash
# Alpha Vantage APIのテスト
python3 tests/api/test_api.py

# LLM APIのテスト
python3 tests/api/test_llm_alt.py
```

## テスト内容

### Alpha Vantage API
- 1分足の株価データ取得
- 銘柄情報の取得
- エラーハンドリング

### LLM API（Claude）
- 基本的な応答機能
- モデルバージョンの指定
- エラーハンドリング

## 注意事項

- APIキーは`.env`ファイルで管理し、Gitにはコミットしない
- API利用制限に注意（Alpha Vantageは1分あたり5リクエスト）
- テスト実行前に必要なパッケージ（requests, python-dotenv）をインストール