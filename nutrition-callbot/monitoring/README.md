# CallBot scale monitoring

Mục tiêu của stack này là đo khả năng scale theo TTFA, không chỉ đo throughput.

## Chạy app + monitoring

Từ thư mục `nutrition-callbot/`:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  -f docker-compose.lmdeploy.yml \
  -f docker-compose.monitoring.yml \
  up --build -d
```

Nếu host có NVIDIA GPU và muốn thu GPU metrics:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  -f docker-compose.lmdeploy.yml \
  -f docker-compose.monitoring.yml \
  --profile gpu-monitoring \
  up --build -d
```

## Cổng dịch vụ

- Gateway: <http://localhost:8000>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3001>
- cAdvisor: <http://localhost:8081>
- DCGM Exporter: <http://localhost:9400> nếu bật profile GPU.
- LMDeploy TTS server: <http://localhost:23333/v1/models>

## Chạy riêng TTS qua LMDeploy

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  -f docker-compose.lmdeploy.yml \
  up --build -d tts-lmdeploy tts
```

Backend TTS lúc này là:

```text
TTS_BACKEND=lmdeploy
TTS_LMDEPLOY_API_BASE=http://tts-lmdeploy:23333/v1
TTS_LMDEPLOY_MODEL=pnnbao-ump/VieNeu-TTS
TTS_MAX_CONCURRENT=4
```

Không dùng bản GGUF cho LMDeploy. Bản GGUF chỉ dành cho backend local.
Khi benchmark, tăng dần `TTS_MAX_CONCURRENT` theo các mốc `1`, `2`, `4`, `8`; không tăng thẳng lên cao nếu chưa kiểm tra audio lỗi và GPU memory.

Grafana mặc định:

```text
user: admin
password: admin
```

Nên đổi bằng biến môi trường `GRAFANA_ADMIN_PASSWORD` khi chạy trên server.

## Metrics chính

- `callbot_tts_time_to_first_audio_seconds`: thời gian tới audio đầu tiên của TTS.
- `callbot_tts_request_duration_seconds`: thời gian xử lý TTS.
- `callbot_tts_rtf`: real-time factor của TTS.
- `callbot_tts_active_requests`: số request TTS đang xử lý.
- `callbot_gateway_ws_sessions`: số phiên WebSocket voice đang mở tại Gateway.
- `callbot_brain_time_to_first_text_seconds`: thời gian tới text đầu tiên từ LLM/RAG.
- `callbot_brain_request_duration_seconds`: thời gian xử lý LLM/RAG.
- `callbot_asr_transcribe_duration_seconds`: thời gian xử lý ASR.
- `callbot_asr_ws_sessions`: số phiên ASR WebSocket đang mở.

## Benchmark TTS concurrent

Từ thư mục `nutrition-callbot/`:

```bash
python load_tests/tts_concurrency.py \
  --url http://localhost:50053/speak/stream \
  --concurrency 1,2,4,8,10,16 \
  --requests-per-level 30 \
  --output tts_concurrency_results.json
```

Tiêu chí dừng hợp lý:

```text
p95 TTFA > 2s
error rate > 1%
GPU memory > 90%
audio lỗi, lặp, méo hoặc timeout
```

Kết luận scale nên viết theo mẫu:

```text
TTS backend giữ p95 TTFA dưới 2 giây tới X request đồng thời trên cấu hình GPU thử nghiệm.
```
