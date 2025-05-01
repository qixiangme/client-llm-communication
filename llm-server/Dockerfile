FROM nvidia/cuda:12.8.0-base-ubuntu22.04

# 기본 의존성 설치
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git curl \
    && apt-get clean

# Python 최신 alias 설정
RUN ln -sf python3 /usr/bin/python && ln -sf pip3 /usr/bin/pip

WORKDIR /app

# PyTorch (CUDA 12.1 호환) 설치
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Transformers + Accelerate + 기타
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# CUDA 가시성 설정
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# 앱 코드 복사
COPY server.py /app

CMD ["python", "server.py"]
