import os
from dotenv import load_dotenv
import requests
import json

def test_alpha_vantage_api():
    """Alpha Vantage APIのテスト"""
    load_dotenv()
    api_key = os.getenv('STOCK_API_KEY')
    
    print("Testing Alpha Vantage API...")
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=NVDA&interval=1min&apikey={api_key}'
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            print("❌ Error:", data["Error Message"])
            return False
        
        if "Time Series (1min)" in data:
            print("✅ Alpha Vantage API: Success")
            print("Sample data:", json.dumps(list(data["Time Series (1min)"].items())[0], indent=2))
            return True
        else:
            print("❌ Unexpected response:", data)
            return False
            
    except Exception as e:
        print("❌ Error:", str(e))
        return False

def test_llm_api():
    """LLM APIのテスト"""
    load_dotenv()
    api_key = os.getenv('LLM_API_KEY')
    
    print("\nTesting LLM API...")
    url = "https://api.claude.ai/v1/messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json={
                "prompt": "Say hello",
                "max_tokens": 10
            }
        )
        
        if response.status_code == 200:
            print("✅ LLM API: Success")
            print("Response:", response.json())
            return True
        else:
            print("❌ Error:", response.text)
            return False
            
    except Exception as e:
        print("❌ Error:", str(e))
        return False

if __name__ == "__main__":
    test_alpha_vantage_api()
    test_llm_api()