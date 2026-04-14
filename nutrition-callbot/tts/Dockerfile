FROM python:3.12-slim

WORKDIR /app
# espeak-ng cần cho phonemizer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl espeak-ng libespeak-ng-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY tts/requirements.txt .
# Cài torch + torchaudio CPU trước để tránh kéo bản CUDA
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY tts/ .

EXPOSE 50053
CMD ["python", "server.py"]
