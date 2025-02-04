import os
from dotenv import load_dotenv
import urllib.request
import json

def test_llm():
    load_dotenv()
    api_key = os.getenv('LLM_API_KEY')
    print(f"Testing LLM API Key: {api_key}")
    
    url = "https://api.claude.ai/v1/messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = json.dumps({
        "prompt": "Say hello",
        "max_tokens": 10
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