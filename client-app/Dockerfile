FROM python:3.11-slim

WORKDIR /app

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 의존성 파일 복사
COPY requirements.txt .

# 의존성 설치
RUN pip install -r requirements.txt

# 클라이언트 코드 복사
COPY client.py .



# 클라이언트 실행
CMD ["python", "client.py"]
