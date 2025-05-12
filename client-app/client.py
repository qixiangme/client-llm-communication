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
    while True:
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            generated_text = response.json().get("generated_text")
            return generated_text
        except requests.exceptions.RequestException as e:
            print("텍스트 생성 실패:", e)
            time.sleep(5)
if __name__ == "__main__":
    texts = [
        {"input": "I am passionate about software development and have gained experience through various projects.", "output": "How would you express your passion for software development?"},
        {"input": "I have grown by learning various technologies and have especially realized the importance of teamwork.", "output": "What do you think is the most important aspect of teamwork?"},
        {"input": "A good leader must listen to their team and guide them towards a common goal.", "output": "What makes a great leader?"},
        {"input": "Collaboration and communication are key when working on a project with a team.", "output": "How do you ensure effective communication in a team project?"},
        {"input": "Continuous learning is essential for personal and professional growth.", "output": "What is the importance of continuous learning in one's career?"},
        {"input": "I believe in the power of collaboration and the importance of sharing knowledge.", "output": "How do you promote knowledge sharing in a team?"},
        {"input": "I have experience in agile methodologies and understand the importance of adaptability.", "output": "What is your experience with agile methodologies?"},
        {"input": "I enjoy solving complex problems and finding efficient solutions.", "output": "How do you approach problem-solving in software development?"},
        {"input": "I value feedback and see it as an opportunity for growth.", "output": "How do you handle feedback from peers?"},
        {"input": "I am committed to delivering high-quality work and meeting deadlines.", "output": "What is your approach to time management?"},
        {"input": "I have a strong understanding of software development principles and best practices.", "output": "What are the key principles of software development?"},
        {"input": "I am always looking for ways to improve my skills and knowledge.", "output": "How do you stay updated with the latest trends in technology?"},
        {"input": "I believe in the importance of mentorship and helping others grow.", "output": "How do you approach mentorship in your career?"},
        {"input": "I have experience working in diverse teams and appreciate different perspectives.", "output": "How do you handle diversity in a team?"},
        {"input": "I am passionate about open-source projects and contributing to the community.", "output": "What is your experience with open-source contributions?"},
        {"input": "I understand the importance of user experience and design in software development.", "output": "How do you prioritize user experience in your projects?"},
        {"input": "I have experience with cloud technologies and understand their impact on software development.", "output": "What is your experience with cloud technologies?"},
        {"input": "I am committed to writing clean and maintainable code.", "output": "What is your approach to code quality?"},
        {"input": "I believe in the importance of testing and quality assurance.", "output": "How do you ensure the quality of your code?"},
        {"input": "I have experience with version control systems and understand their importance.", "output": "What is your experience with version control?"},
        {"input": "I am always looking for ways to optimize performance and efficiency.", "output": "How do you approach performance optimization in your projects?"},
        {"input": "I value collaboration and believe that great ideas can come from anyone.", "output": "How do you foster a collaborative environment in your team?"},
        {"input": "I have experience with data analysis and understand its importance in decision-making.", "output": "What is your experience with data analysis?"},
        {"input": "I am committed to continuous improvement and always seek feedback.", "output": "How do you approach self-improvement in your career?"},
        {"input": "I believe in the importance of work-life balance and mental health.", "output": "How do you maintain a healthy work-life balance?"},
        {"input": "I have experience with project management and understand its challenges.", "output": "What is your experience with project management?"},
        {"input": "I am passionate about learning new technologies and expanding my skill set.", "output": "How do you approach learning new technologies?"},
        {"input": "I believe in the importance of empathy and understanding in teamwork.", "output": "How do you practice empathy in your work?"},
        {"input": "I have experience with remote work and understand its challenges.", "output": "What is your experience with remote work?"},
        {"input": "I am committed to diversity and inclusion in the workplace.", "output": "How do you promote diversity and inclusion in your team?"},
        {"input": "I believe in the importance of networking and building relationships.", "output": "How do you approach networking in your career?"},
        {"input": "I have experience with mentoring and coaching others.", "output": "What is your approach to mentoring?"}

    ]

    # Fine-tuning 시작
    # print("\n파인튜닝 시작...\n")
    # fine_tune_response = fine_tune_model(texts)
    # print("서버 응답:", fine_tune_response.get("message", "응답 없음"))

    prompt = "i know the importance of teamwork and collaboration in software development."
    generated_text = generate_text(prompt)
    if generated_text:
        print("\n생성된 텍스트:")
        print("="*50)
        print(generated_text.strip().split('\n')[-1])
        print("="*50)
