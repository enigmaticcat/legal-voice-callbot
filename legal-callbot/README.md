# 🏛️ Legal CallBot

> Hệ thống tư vấn pháp luật Việt Nam bằng giọng nói AI

## 📐 Kiến Trúc

```
Client (Web) → WebSocket → Gateway (FastAPI) → gRPC → ASR / Brain / TTS
```

| Service | Port | Mô tả |
|:--------|:-----|:-------|
| **Gateway** | 8000 | API Gateway + WebSocket Orchestrator |
| **ASR** | 50051 | Nhận diện giọng nói (Faster-Whisper) |
| **Brain** | 50052 | Suy luận pháp lý (Gemini + RAG) |
| **TTS** | 50053 | Tổng hợp giọng nói (VieNeu-TTS) |
| **Web** | 3000 | Giao diện cuộc gọi (React) |

## 🚀 Quick Start

```bash
# 1. Copy env
cp .env.example .env

# 2. Build & run
make build
make up

# 3. Kiểm tra
make health
curl localhost:8000   # → {"status": "ok"}

# 4. Dọn dẹp
make down
```

## 📁 Cấu Trúc Thư Mục

```
legal-callbot/
├── gateway/             # API Gateway (FastAPI)
│   ├── routes/          # HTTP/WS endpoints
│   ├── services/        # Business logic
│   ├── grpc_clients/    # gRPC stubs
│   ├── middleware/       # Auth, logging
│   └── models/          # Pydantic schemas
├── asr/                 # ASR Worker
│   └── core/            # VAD, Transcriber, Audio utils
├── brain/               # Brain Worker
│   └── core/            # LLM, RAG, Query Expansion, Prompts
├── tts/                 # TTS Worker
│   └── core/            # Synthesizer, Chunker, Audio utils
├── web/                 # React Frontend
│   └── src/
│       ├── components/  # UI Components
│       ├── hooks/       # Custom React hooks
│       └── services/    # API calls
├── protos/              # gRPC Proto definitions
└── scripts/             # Build & deploy tools
```

## 🔧 Makefile Commands

| Command | Mô tả |
|:--------|:-------|
| `make up` | Start production containers |
| `make down` | Stop containers |
| `make build` | Build Docker images |
| `make logs` | View logs (follow) |
| `make dev` | Start with hot-reload |
| `make health` | Check all services |
| `make clean` | Remove containers + images |
