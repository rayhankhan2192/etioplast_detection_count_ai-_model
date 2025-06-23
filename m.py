import os
import requests

# Replace with your actual API key and endpoint
API_KEY = "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"
API_URL = "https://api.groq.com/openai/v1/chat/completions"


headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "llama-3.3-70b-versatile",
    "messages": [
        {"role": "user", "content": "tell me about ML"},
    ],
}

response = requests.post(API_URL, headers=headers, json=data)

if response.status_code == 200:
    resp = response.json()
    print(resp["choices"][0]["message"]["content"])
else:
    print("Error", response.status_code, response.text)
