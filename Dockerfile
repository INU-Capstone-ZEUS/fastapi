# 1. 베이스 이미지 설정 (Python 3.11-slim 사용)
FROM python:3.11-slim


# 2. 작업 디렉토리 설정
WORKDIR /code

# 3. 필요한 패키지 설치 (Chrome 및 의존성 포함)
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg \
    ca-certificates \
    unzip \
    locales \
    && rm -rf /var/lib/apt/lists/*

# 로케일 설정
RUN locale-gen ko_KR.UTF-8
ENV LANG=ko_KR.UTF-8 \
    LANGUAGE=ko_KR:ko \
    LC_ALL=ko_KR.UTF-8

# 4. Google Chrome의 GPG 키 및 리포지토리 추가
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list && \
    apt-get update && apt-get install -y google-chrome-stable

# 5. pytorch 다운로드
RUN pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu

# 6. 인증서 파일 복사
COPY ./app/fullchain.pem /etc/ssl/certs/fullchain.pem
COPY ./app/privkey.pem /etc/ssl/private/privkey.pem

# 7. 파이썬 의존성 파일 복사 및 설치
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 8. 애플리케이션 파일 복사
COPY ./app /code/app
COPY ./checkpoint.pth /code/app/routers/checkpoint.pth

# 9. FastAPI에서 사용할 포트 열기
EXPOSE 8080


# 10. FastAPI 서버 실행 (uvicorn 사용)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--ssl-keyfile", "/etc/ssl/private/privkey.pem", "--ssl-certfile", "/etc/ssl/certs/fullchain.pem"]
# CMD ["python", "app/main.py"]