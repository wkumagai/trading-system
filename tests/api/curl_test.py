import os
import subprocess
from dotenv import load_dotenv

def test_api():
    load_dotenv()
    api_key = os.getenv('STOCK_API_KEY')
    print(f"Testing API Key: {api_key}")
    
    curl_command = f'curl "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=NVDA&interval=1min&apikey={api_key}"'
    
    try:
        result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
        print("\nAPI Response:")
        print(result.stdout)
        
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_api()