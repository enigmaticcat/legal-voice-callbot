FROM python:3.12-slim

WORKDIR /app
# espeak-ng cần cho phonemizer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl espeak-ng libespeak-ng-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY tts/requirements.txt .
RUN pip install --no-cache-dir "torch==2.5.1+cu121" "torchaudio==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121
# llama-cpp-python: CUDA 12.1 pre-built wheel
RUN pip install --no-cache-dir llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

COPY tts/ .

EXPOSE 50053
CMD ["python", "server.py"]
