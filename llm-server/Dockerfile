# Python slim 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필수 라이브러리 설치
RUN apt update && apt install -y git curl

# 의존성 파일 복사
COPY requirements.txt .

# 의존성 설치
RUN pip install -r requirements.txt

# 서버 코드 복사
COPY server.py .

# 서버 실행
CMD ["python", "server.py"]
