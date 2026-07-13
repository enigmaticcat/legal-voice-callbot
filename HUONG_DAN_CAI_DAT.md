# Hướng dẫn cài đặt và chạy Nutrition CallBot

## 1. Nội dung gói nộp

File zip này chứa mã nguồn và cấu hình chạy của hệ thống voicebot tư vấn dinh dưỡng:

- `nutrition-callbot/`: mã nguồn chương trình chính gồm giao diện, Gateway, ASR, mô-đun xử lý hội thoại, TTS, Qdrant và Redis.
- `data-pipeline/`: mã nguồn thu thập, làm sạch và phân đoạn dữ liệu.
- `evaluation/`: mã nguồn phục vụ đánh giá ASR, TTS, RAG và độ trễ.
- `docker-compose.yml`, `docker-compose.gpu.yml`: cấu hình triển khai bằng Docker.
- `README.md`, `CACHE_VERIFICATION.md`, các file `.md`: tài liệu mô tả kiến trúc và kiểm thử.

Gói nộp không bao gồm file PDF, slide, ảnh báo cáo, dữ liệu lớn, môi trường ảo, `node_modules`, model checkpoint, log, file build LaTeX hoặc file `.env` chứa cấu hình riêng.

## 2. Yêu cầu môi trường

### Môi trường chung

- Hệ điều hành Linux/macOS.
- Docker và Docker Compose v2.
- Tối thiểu 16 GB RAM để chạy các thành phần cơ bản.
- Kết nối Internet ở lần chạy đầu để tải Docker image và model cần thiết.

### Nếu chạy GPU

- GPU NVIDIA (khuyến nghị VRAM >= 8 GB).
- Driver NVIDIA đã cài trên host.
- NVIDIA Container Toolkit — xem hướng dẫn tại [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- Kiểm tra GPU trong Docker bằng lệnh:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 3. Giải nén và chuẩn bị cấu hình

```bash
unzip nutrition-callbot-source-code.zip
cd nutrition-callbot
cp .env.example .env
```

Sau đó chỉnh `.env` nếu cần. Một số biến thường dùng:

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `HF_TOKEN` | _(trống)_ | Token Hugging Face nếu cần tải model gated |
| `GATEWAY_PORT` | `8000` | Cổng API Gateway |
| `WEB_PORT` | `3000` | Cổng giao diện web |
| `QDRANT_COLLECTION` | `nutrition_articles` | Tên collection Qdrant |
| `QDRANT_SNAPSHOT_FILE` | `nutrition_articles.snapshot` | Tên file snapshot khôi phục |

File `.env` không được đưa vào zip để tránh lộ khóa hoặc cấu hình cá nhân.

## 4. Chuẩn bị kho tri thức Qdrant

Hệ thống dùng Qdrant để lưu vector của kho tri thức dinh dưỡng. Gói source code không kèm snapshot để giữ kích thước nhỏ.

Nếu có file snapshot, đặt vào thư mục `snapshots/` ở **cùng cấp** với `nutrition-callbot/`:

```bash
mkdir -p snapshots
cp /path/to/nutrition_articles.snapshot snapshots/
```

Cấu trúc thư mục:

```text
Callbot/
├── nutrition-callbot/   ← mã nguồn
└── snapshots/
    └── nutrition_articles.snapshot
```

Khi khởi động, service `qdrant-restore` tự động khôi phục snapshot nếu collection chưa có dữ liệu. Quá trình này mất khoảng **3–5 phút** tùy kích thước file.

## 5. Chạy hệ thống

### Chế độ GPU (khuyến nghị)

```bash
cd nutrition-callbot
make gpu-build   # Build images lần đầu hoặc sau khi thay đổi code
make gpu-up      # Khởi động tất cả services
```

Lần chạy đầu mất **5–10 phút** để tải image vLLM (~10 GB) và model Qwen3-4B (~3 GB). Các lần sau nhanh hơn do đã cache.

Xem log theo dõi quá trình khởi động:

```bash
make gpu-logs
```

Dừng hệ thống:

```bash
make gpu-down
```

### Chế độ CPU

```bash
cd nutrition-callbot
make build
make up
```

> LLM sẽ chạy rất chậm ở chế độ CPU. Chỉ dùng để phát triển hoặc kiểm thử.

## 6. Kiểm tra sau khi khởi động

Thứ tự healthy khi khởi động:

```
redis → qdrant → qdrant-restore (exit 0) → asr / brain / tts → gateway
llm (song song, mất ~3 phút để load model lên GPU)
```

Kiểm tra trạng thái tất cả services:

```bash
make health
```

Hoặc kiểm tra từng service:

```bash
curl http://localhost:8000/health   # Gateway
curl http://localhost:50051/health  # ASR
curl http://localhost:50052/health  # Brain
curl http://localhost:50053/health  # TTS
```

Mở giao diện:

```
http://localhost:3000
```

Test pipeline Brain trực tiếp:

```bash
curl -s -X POST http://localhost:50052/think \
  -H "Content-Type: application/json" \
  -d '{"query": "vitamin C có trong thực phẩm nào?", "session_id": "test"}' \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('TEXT:', d['text'][:200])
print('TIMING:', d['timing'])
"
```

## 7. Tính năng upload tài liệu

Trong phiên hội thoại, người dùng có thể upload tài liệu dinh dưỡng cá nhân (`.pdf`, `.txt`, `.md`, tối đa 5 MB) bằng nút 📄 trên giao diện. Hệ thống sẽ:

1. Parse tài liệu bằng **LangChain** (`PyPDFLoader` cho PDF, `RecursiveCharacterTextSplitter` để chia chunk).
2. Embed các chunk bằng model `AITeamVN/Vietnamese_Embedding`.
3. Lưu vào Qdrant collection riêng `user_documents`, gắn `session_id` để cách ly giữa các phiên.
4. Brain tự động tìm kiếm tài liệu này khi trả lời — song song với kho tri thức chính.

Tài liệu chỉ tồn tại trong phạm vi phiên hội thoại hiện tại.

## 8. Xử lý sự cố thường gặp

**LLM không kết nối được (Brain báo connection error):**

```bash
# Kiểm tra LLM có trong đúng network không
docker inspect nutrition-callbot-llm-1 \
  --format '{{json .NetworkSettings.Networks}}' | python3 -m json.tool
# Phải thấy "nutrition-callbot_callbot-net"

# Nếu sai network, force recreate
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  up -d --force-recreate llm
```

**Qdrant restore timeout:**

```bash
# Kiểm tra trạng thái collection
curl -s http://localhost:6333/collections/nutrition_articles | python3 -m json.tool
# Nếu status=green nhưng qdrant-restore exit 1, start brain/gateway thủ công
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  up -d --no-deps brain gateway
```

**Port bị conflict với project khác:**

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
```

## 9. Kiểm thử mã nguồn

Kiểm tra cú pháp Python:

```bash
cd nutrition-callbot
python3 -m compileall -q gateway brain asr tts tests test_pipeline.py test_full_pipeline.py
```

Chạy unit test:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest tests/test_unit.py -q
```

Build giao diện:

```bash
npm --prefix web install
npm --prefix web run build
```

## 10. Makefile — danh sách lệnh

| Lệnh | Mô tả |
|------|-------|
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

## 11. Cấu trúc chương trình chính

```text
nutrition-callbot/
├── gateway/      Bộ điều phối phiên, WebSocket và luồng audio/text
├── asr/          Nhận dạng tiếng nói tiếng Việt
├── brain/        Guardrail, RAG, upload tài liệu, cache, prompt và sinh câu trả lời
├── tts/          Tổng hợp tiếng nói và streaming PCM
├── web/          Giao diện người dùng React/Vite
├── tests/        Kiểm thử đơn vị
├── scripts/      Script hỗ trợ kiểm tra cache và đo độ trễ
└── docker-compose*.yml
```
