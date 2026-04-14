# ── Stage 1: Build React frontend ──────────────────────────────────────
FROM node:20-slim AS web-builder
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ .
RUN npm run build

# ── Stage 2: Gateway ────────────────────────────────────────────────────
FROM python:3.12-slim
WORKDIR /app/gateway

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway/ .
# Web dist được serve tại /app/web/dist (Path(__file__).parent.parent / "web" / "dist")
COPY --from=web-builder /web/dist /app/web/dist

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
