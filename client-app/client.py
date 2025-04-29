import requests
import time
import json

def fine_tune_model(texts):
    url = "http://llm-server:8000/finetune"
    
    # 서버가 준비될 때까지 기다리기
    while True:
        try:
            response = requests.post(url, json={"texts": texts})
            response.raise_for_status()  # 응답 코드가 2xx가 아니면 예외 발생
            return response.json()
        except requests.exceptions.RequestException as e:
            print("서버 준비되지 않음, 재시도 중...", e)
            time.sleep(5)  # 5초마다 재시도

def generate_text(prompt):
    url = "http://llm-server:8000/generate"
    data = {"prompt": prompt}

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        generated_text = response.json().get("generated_text")
        return generated_text
    except requests.exceptions.RequestException as e:
        print("텍스트 생성 실패:", e)
        return None

if __name__ == "__main__":
    texts = [
        {"input": "I am passionate about software development and have gained experience through various projects.", "output": "How would you express your passion for software development?"},
        {"input": "I have grown by learning various technologies and have especially realized the importance of teamwork.", "output": "What do you think is the most important aspect of teamwork?"}
    ]

    # Fine-tuning 시작
    print("\n파인튜닝 시작...\n")
    fine_tune_response = fine_tune_model(texts)
    print("서버 응답:", fine_tune_response.get("message", "응답 없음"))

    prompt = "what is essential for a successful software project?"
    generated_text = generate_text(prompt)
    if generated_text:
        print("\n생성된 텍스트:")
        print("="*50)
        print(f"{generated_text}")
        print("="*50)
