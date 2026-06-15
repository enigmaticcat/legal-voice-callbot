FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY brain/requirements.txt .
# Cài torch CPU trước để tránh pip kéo bản CUDA (~2GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY brain/ ./brain/
COPY observability/ ./observability/

EXPOSE 50052
CMD ["python", "-m", "brain.server"]
