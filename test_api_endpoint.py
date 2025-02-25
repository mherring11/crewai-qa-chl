import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_url = os.getenv("CHL_API_URL")
api_referer = os.getenv("CHL_API_REFERER")

print("Testing API Endpoint:")
print(f"URL: {api_url}")
print(f"Referer: {api_referer}")

# Test question
test_question = "What services does CHL offer?"

def test_api_connection(headers, question=test_question):
    """Test the API connection with given headers"""
    
    data = {
        "question": question,
        "test": True
    }
    
    try:
        print(f"\nSending request to {api_url}...")
        response = requests.post(api_url, headers=headers, json=data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        # Try to parse JSON response
        try:
            response_json = response.json()
            print(f"\nResponse JSON: {json.dumps(response_json, indent=2)}")
            return True
        except json.JSONDecodeError:
            print(f"\nResponse Text (not JSON): {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"\nRequest Error: {str(e)}")
        return False

# Test 1: Basic test with default headers
print("\n\n=== TEST 1: Default Headers ===")
default_headers = {
    'Content-Type': 'application/json',
    'Referer': api_referer,
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

}
test_api_connection(default_headers)


print("\n\nAPI Test Complete. Please check the results above to identify which configuration works.")