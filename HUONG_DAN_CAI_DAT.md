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
- Docker và Docker Compose.
- Tối thiểu 16 GB RAM để chạy các thành phần cơ bản.
- Kết nối Internet ở lần chạy đầu để tải Docker image và model cần thiết.

### Nếu chạy GPU

- GPU NVIDIA.
- Driver NVIDIA đã cài trên host.
- NVIDIA Container Toolkit.
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

- `HF_TOKEN`: token Hugging Face nếu cần tải model gated.
- `GATEWAY_PORT`: cổng Gateway, mặc định `8000`.
- `WEB_PORT`: cổng giao diện, mặc định `3000`.
- `QDRANT_COLLECTION`: tên collection Qdrant, mặc định `nutrition_articles`.

File `.env` không được đưa vào zip để tránh lộ khóa hoặc cấu hình cá nhân.

## 4. Chạy hệ thống bằng Docker CPU

```bash
cd nutrition-callbot
docker compose up --build -d
```

Kiểm tra trạng thái:

```bash
docker compose ps
docker compose logs -f
```

Kiểm tra Gateway:

```bash
curl http://localhost:8000/health
```

Mở giao diện:

```text
http://localhost:3000
```

## 5. Chạy hệ thống bằng Docker GPU

```bash
cd nutrition-callbot
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  up --build -d
```

Xem log theo từng service:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  logs -f gateway asr brain tts
```

Dừng hệ thống:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  down
```

## 6. Lưu ý về kho tri thức Qdrant

Hệ thống sử dụng Qdrant để lưu vector của kho tri thức dinh dưỡng. Gói source code không kèm snapshot hoặc dữ liệu vector lớn để giữ kích thước dưới 30 MB.

Nếu có snapshot Qdrant, đặt file snapshot vào thư mục `snapshots/` ở cùng cấp với `nutrition-callbot/`, sau đó cấu hình:

```bash
QDRANT_SNAPSHOT_FILE=nutrition_articles.snapshot
QDRANT_COLLECTION=nutrition_articles
```

Khi khởi động, service `qdrant-restore` sẽ khôi phục snapshot nếu collection chưa có dữ liệu.

## 7. Kiểm thử nhanh mã nguồn

Chạy kiểm tra Python:

```bash
cd nutrition-callbot
python3 -m compileall -q gateway brain asr tts tests test_pipeline.py test_full_pipeline.py
```

Chạy unit test nếu đã cài dependency phát triển:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest tests/test_unit.py -q
```

Build giao diện:

```bash
npm --prefix web install
npm --prefix web run build
```

## 8. Cấu trúc chương trình chính

```text
nutrition-callbot/
├── gateway/      Bộ điều phối phiên, WebSocket và luồng audio/text
├── asr/          Nhận dạng tiếng nói tiếng Việt
├── brain/        Guardrail, RAG, cache, prompt và sinh câu trả lời
├── tts/          Tổng hợp tiếng nói và streaming PCM
├── web/          Giao diện người dùng React/Vite
├── tests/        Kiểm thử đơn vị
├── scripts/      Script hỗ trợ kiểm tra cache
└── docker-compose*.yml
```

