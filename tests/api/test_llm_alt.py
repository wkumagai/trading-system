import os
from dotenv import load_dotenv
import urllib.request
import json

def test_llm():
    load_dotenv()
    api_key = os.getenv('LLM_API_KEY')
    print(f"Testing LLM API Key: {api_key}")
    
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = json.dumps({
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": "Say hello"
            }
        ]
    }).encode('utf-8')
    
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        with urllib.request.urlopen(req) as response:
            result = response.read().decode()
            print("\nAPI Response:")
            print(result)
            
    except urllib.error.HTTPError as e:
        print(f"\nHTTP Error: {e.code}")
        print(e.read().decode())
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    test_llm()