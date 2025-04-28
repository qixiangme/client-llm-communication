import requests
import json

url = "http://llm-server:8000/generate"
prompt = "Once upon a time"

# 서버에 요청 보내기
response = requests.post(url, json={"prompt": prompt})

if response.status_code == 200:
    generated_text = response.json().get("generated_text")
    print("Generated Text:", generated_text)
else:
    print(f"Error: {response.status_code}")
