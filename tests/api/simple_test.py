import os
from dotenv import load_dotenv
import urllib.request
import json

def test_alpha_vantage():
    load_dotenv()
    api_key = os.getenv('STOCK_API_KEY')
    print(f"Using API Key: {api_key}")
    
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=NVDA&interval=1min&apikey={api_key}'
    
    try:
        print("Sending request to Alpha Vantage...")
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            
        if "Error Message" in data:
            print("Error:", data["Error Message"])
        elif "Time Series (1min)" in data:
            print("Success! Received data:")
            first_entry = list(data["Time Series (1min)"].items())[0]
            print(json.dumps(first_entry, indent=2))
        else:
            print("Unexpected response:", data)
            
    except Exception as e:
        print("Error occurred:", str(e))