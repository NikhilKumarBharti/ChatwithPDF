import requests
import os

def test_openrouter_connection():
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("API key not found!")
        return
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "ChatWithPDF"
    }
    
    data = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [{"role": "user", "content": "Hello, can you respond?"}],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=data,
            headers=headers,
            timeout=30
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Connection test failed: {e}")

test_openrouter_connection()