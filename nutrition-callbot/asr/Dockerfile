FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY asr/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY asr/ .
COPY observability/ ./observability/

EXPOSE 50051
CMD ["python", "server.py"]
