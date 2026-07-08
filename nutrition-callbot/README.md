# Nutrition CallBot

> Hệ thống tư vấn dinh dưỡng bằng giọng nói AI — chạy hoàn toàn on-device, không cần Gemini API.

## Kiến Trúc

```
Client (Web, port 3000)
    ↕ WebSocket
Gateway (FastAPI, port 8000)
    ├→ ASR   (port 50051)  — Zipformer-RNNT, nhận dạng tiếng Việt
    ├→ Brain (port 50052)  — Qwen3-4B-Instruct-2507 (vLLM) + RAG (Qdrant)
    └→ TTS   (port 50053)  — VieNeu-TTS, 24kHz
```

| Service | Port | Mô tả |
|:--------|:-----|:-------|
| **Gateway** | 8000 | API Gateway + WebSocket Orchestrator |
| **ASR** | 50051 | Nhận dạng giọng nói (Zipformer-RNNT) |
| **Brain** | 50052 | Qwen3-4B + RAG (Qdrant + Vietnamese_Embedding) |
| **TTS** | 50053 | Tổng hợp giọng nói (VieNeu-TTS) |
| **LLM** | 8080 | vLLM inference server (chỉ chế độ GPU) |
| **Qdrant** | 6333 | Vector database |
| **Redis** | 6379 | Cache (retrieval + TTS + semantic) |
| **Web** | 3000 | Giao diện cuộc gọi (React) |

---

## Yêu Cầu

- Docker + Docker Compose v2
- **GPU mode**: NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Qdrant snapshot đặt tại `../snapshots/nutrition_articles.snapshot` (thư mục cha của `nutrition-callbot/`)

---

## Cài Đặt Lần Đầu

### 1. Clone & cấu hình môi trường

```bash
git clone <repo-url>
cd nutrition-callbot
cp .env.example .env
```

Chỉnh `.env` theo môi trường:

```env
# Không bắt buộc — để trống nếu không có HuggingFace token
HF_TOKEN=

# Tên collection Qdrant và file snapshot
QDRANT_COLLECTION=nutrition_articles
QDRANT_SNAPSHOT_FILE=nutrition_articles.snapshot

# Ports (giữ mặc định nếu không bị conflict)
GATEWAY_PORT=8000
ASR_PORT=50051
BRAIN_PORT=50052
TTS_PORT=50053
WEB_PORT=3000
QDRANT_PORT=6333
REDIS_PORT=6379
```

### 2. Đặt snapshot Qdrant

Snapshot phải nằm ở `../snapshots/` (một cấp trên thư mục `nutrition-callbot/`):

```bash
mkdir -p ../snapshots
cp /path/to/nutrition_articles.snapshot ../snapshots/
```

Khi khởi động, `qdrant-restore` sẽ tự upload snapshot vào Qdrant nếu collection chưa có dữ liệu.

---

## Chạy Hệ Thống

### Chế Độ GPU (khuyến nghị)

Chạy toàn bộ: LLM (vLLM), ASR, TTS, Brain đều dùng GPU.

```bash
# Build images (chỉ cần lần đầu hoặc khi có thay đổi code)
make gpu-build

# Khởi động tất cả services
make gpu-up
```

Lần đầu mất **5–10 phút** để:
- Pull image `vllm/vllm-openai` (~10GB)
- Download model Qwen3-4B-Instruct-2507 (~3GB) về `~/.cache/huggingface`
- Restore Qdrant snapshot

Sau khi tất cả healthy, truy cập: **http://localhost:3000**

### Chế Độ CPU (không có GPU)

```bash
make build
make up
```

> LLM sẽ chạy rất chậm ở chế độ CPU. Chỉ dùng để phát triển/test.

---

## Kiểm Tra Trạng Thái

```bash
# Xem tất cả containers
make ps

# Health check tất cả services
make health

# Xem logs real-time
make gpu-logs        # GPU mode
make logs            # CPU mode

# Xem log của một service cụ thể
docker logs callbot-brain -f
docker logs callbot-tts -f
docker logs nutrition-callbot-llm-1 -f
```

### Thứ tự healthy khi khởi động

```
redis → qdrant → qdrant-restore (exit 0) → brain/asr/tts → gateway
llm (song song, mất ~3 phút)
```

Brain **chờ** `qdrant-restore` hoàn thành và `llm` healthy trước khi start.

---

## Test Pipeline

```bash
# Test Brain trực tiếp
curl -s -X POST http://localhost:50052/think \
  -H "Content-Type: application/json" \
  -d '{"query": "vitamin C có trong thực phẩm nào?", "session_id": "test"}' \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('TEXT:', d['text'][:200])
print('TIMING:', d['timing'])
"

# Test Gateway (end-to-end text)
curl -s http://localhost:8000/health | python3 -m json.tool

# Benchmark latency 5 câu hỏi
python3 -c "
import requests

questions = [
    'Bệnh gút nên kiêng ăn gì?',
    'Phụ nữ mang thai cần bổ sung axit folic bao nhiêu?',
    'Chất xơ có vai trò gì trong tiêu hóa?',
    'Người ăn chay có thiếu B12 không?',
    'Kẽm có trong thực phẩm nào?',
]
print(f'{\"Query\":<45} {\"expand\":>7} {\"rag\":>7} {\"ttft\":>7} {\"total\":>7}')
print('-'*75)
for q in questions:
    r = requests.post('http://localhost:50052/think',
        json={'query': q, 'session_id': 'bench'}, timeout=30)
    t = r.json()['timing']
    print(f'{q:<45} {t[\"expand_ms\"]:>6.0f}ms {t[\"rag_ms\"]:>6.0f}ms {t[\"llm_ttft_ms\"]:>6.0f}ms {t[\"total_ms\"]:>6.0f}ms')
"
```

---

## Makefile Commands

| Command | Mô tả |
|:--------|:-------|
| `make gpu-up` | Khởi động tất cả services với GPU |
| `make gpu-down` | Dừng tất cả services (GPU mode) |
| `make gpu-build` | Build lại Docker images (GPU mode) |
| `make gpu-logs` | Xem logs real-time (GPU mode) |
| `make up` | Khởi động (CPU mode) |
| `make down` | Dừng (CPU mode) |
| `make build` | Build images (CPU mode) |
| `make health` | Health check tất cả services |
| `make ps` | Xem trạng thái containers |
| `make logs` | Xem logs real-time (CPU mode) |
| `make clean` | Xóa containers + images + volumes |

---

## Xử Lý Sự Cố

### LLM không kết nối được (Brain báo connection error)

```bash
# Kiểm tra LLM có trong đúng network không
docker inspect nutrition-callbot-llm-1 --format '{{json .NetworkSettings.Networks}}'
# Phải thấy "nutrition-callbot_callbot-net"

# Nếu sai network, force recreate
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --force-recreate llm
```

### Qdrant restore timeout

```bash
# Kiểm tra trạng thái collection
curl -s http://localhost:6333/collections/nutrition_articles | python3 -m json.tool

# Nếu status=green nhưng qdrant-restore exit 1, start brain/gateway thủ công
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --no-deps brain gateway
```

### TTS không load được model

```bash
docker logs callbot-tts --tail=50
# Nếu lỗi CUDA: kiểm tra nvidia-container-toolkit
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Port bị conflict với project khác

```bash
# Dừng toàn bộ containers đang chạy
docker ps -q | xargs docker stop

# Hoặc dừng project cụ thể
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
```

---

## Cấu Trúc Thư Mục

```
nutrition-callbot/
├── gateway/          FastAPI app + WebSocket /ws/voice
├── asr/              Speech-to-text (Zipformer-RNNT, ONNX)
├── brain/            LLM + RAG (Qwen3 qua vLLM + Qdrant)
├── tts/              Text-to-speech (VieNeu-TTS, 24kHz)
├── web/              React 18 + Vite frontend
├── protos/           gRPC proto definitions
├── scripts/          Dev utilities + latency measurement
├── tests/            Unit tests
├── docker-compose.yml        Base config (CPU)
├── docker-compose.gpu.yml    GPU override (LLM/ASR/TTS trên GPU)
├── docker-compose.dev.yml    Dev mode (hot-reload)
└── Makefile
```
