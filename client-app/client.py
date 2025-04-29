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
        {"input": "저는 소프트웨어 개발에 열정을 가지고 있으며, 다양한 프로젝트에서 경험을 쌓아왔습니다.", "output": "소프트웨어 개발에 대한 열정을 어떻게 표현하시겠어요?"},
        {"input": "저는 다양한 기술을 배우며 성장해왔고, 특히 팀워크의 중요성을 많이 느꼈습니다.", "output": "팀워크에서 가장 중요한 점은 무엇인가요?"}
    ]
 # Fine-tuning 시작
    print("파인튜닝 시작...")
    fine_tune_response = fine_tune_model(texts)
    print(fine_tune_response)

    prompt = "소프트웨어 개발에 있어 중요한 기술은 무엇인가요?"
    generated_text = generate_text(prompt)
    if generated_text:
        print(repr(generated_text))
        print(f"생성된 텍스트: {generated_text}")