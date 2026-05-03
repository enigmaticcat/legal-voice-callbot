# Kiến Trúc Callbot — Tư Vấn Dinh Dưỡng Qua Giọng Nói

## Tổng quan

```
Browser (React, port 3000)
    │  WebSocket /ws/voice  (binary PCM + JSON events)
    ▼
Gateway  ─── FastAPI, port 8000  (orchestrator)
    │  HTTP streaming
    ├─► ASR   ─── FastAPI, port 50051
    ├─► Brain ─── FastAPI, port 50052
    └─► TTS   ─── FastAPI, port 50053
                        │
                   Qdrant + Gemini API
```

---

## Luồng dữ liệu

### Voice path (người dùng nói)

```
[Web] mic 48kHz
  → downsample 16kHz PCM int16 (OfflineAudioContext)
  → WS binary frames  → [gateway/routes/websocket.py]
  → mở WS tới ASR /ws/transcribe, stream từng chunk 100ms
  → user bấm dừng → {"type":"end"} → ASR.finalize_stream()
  → Gateway nhận transcript → orchestrator.process_text()
```

### Text path (người dùng gõ)

```
[Web] {"type":"text","text":"..."}
  → [gateway/routes/websocket.py]
  → thẳng vào orchestrator.process_text()
```

### Brain pipeline  (`gateway/services/orchestrator.py`)

```
orchestrator.process_text(query)
  │
  ├─ _brain_stream()  →  POST /think/stream  (NDJSON)
  │                         │
  │                    BrainServiceHandler.think()
  │                         ├─ expand_query()          # từ điển chuẩn hoá
  │                         ├─ RAGPipeline.search()    # embed → Qdrant top-5
  │                         ├─ build_prompt()          # few-shot + context + history
  │                         └─ LLMClient.generate_stream()  # Gemini streaming
  │
  ├─ buffer text (>= 40 ký tự hoặc gặp dấu chấm/phẩy)
  │
  └─ _tts_stream()  →  POST /speak/stream  →  raw PCM bytes
       → WS binary → browser play
```

### Sequence đầy đủ

```
User nói
  → [ASR] PCM 16kHz → sherpa-onnx Transducer → transcript
  → [Brain] expand_query → Qdrant search → build_prompt → Gemini stream
  → [chunk_llm_stream] cắt tại dấu chấm, min 40 ký tự
  → [TTS] text chunk → VieNeu-TTS GGUF → PCM 24kHz frames
  → [Web] AudioContext decode → loa
```

---

## Cấu trúc thư mục & file quan trọng

```
nutrition-callbot/
│
├── gateway/
│   ├── routes/
│   │   └── websocket.py        ← WebSocket /ws/voice, phân loại binary/JSON/end_speech
│   ├── services/
│   │   └── orchestrator.py     ← Pipeline ASR→Brain→TTS, buffer → TTS sớm nhất có thể
│   └── config.py
│
├── brain/
│   ├── server.py               ← FastAPI: POST /think, POST /think/stream, POST /pipeline/audio-stream
│   ├── grpc_handler.py         ← BrainServiceHandler.think(): ghép toàn bộ pipeline
│   ├── core/
│   │   ├── rag.py              ← RAGPipeline: multilingual-e5-large → Qdrant query_points
│   │   ├── llm.py              ← GeminiLLMClient / OpenAILLMClient (factory LLMClient)
│   │   ├── prompt.py           ← NUTRITION_SYSTEM_PROMPT + few-shot examples + build_prompt()
│   │   ├── query_expander.py   ← NUTRITION_ALIASES dict, expand_query() regex replace
│   │   ├── chunker.py          ← chunk_llm_stream(): cắt stream tại dấu câu, min 40 ký tự
│   │   └── voice_preprocessing.py
│   └── config.py
│
├── asr/
│   ├── server.py               ← FastAPI: POST /transcribe (batch), WS /ws/transcribe (streaming)
│   ├── grpc_handler.py         ← ASRServiceHandler wraps Transcriber
│   ├── core/
│   │   ├── transcriber.py      ← Sherpa-Onnx OnlineRecognizer, accept_wave/finalize_stream
│   │   ├── audio_utils.py
│   │   └── vad.py
│   └── config.py
│
├── tts/
│   ├── server.py               ← FastAPI: POST /speak (batch), POST /speak/stream
│   ├── grpc_handler.py         ← TTSServiceHandler.speak(): chunk_text → synthesize_stream
│   ├── core/
│   │   ├── synthesizer.py      ← VieNeu-TTS GGUF wrapper, synthesize_stream()
│   │   ├── chunker.py          ← chunk_text(): word-safe split, min 60 ký tự
│   │   └── audio_utils.py
│   ├── vieneu/                 ← VieNeu-TTS model code
│   └── config.py
│
└── web/
    └── src/
        ├── App.jsx             ← State machine: idle/connecting/listening/thinking/speaking
        ├── hooks/
        │   ├── useWebSocket.js ← WS connect/send, tách binary (audio) vs JSON (events)
        │   └── useAudioPlayer.js ← AudioContext, downsample mic, playPcm buffer queue
        ├── components/
        │   ├── CallButton.jsx
        │   ├── StatusBar.jsx
        │   └── Transcript.jsx
        └── services/
            └── api.js          ← getWebSocketUrl()
```

---

## Các thành phần chính giải thích chi tiết

### 1. `gateway/routes/websocket.py` — Entry point

Nhận WebSocket từ browser, phân loại 3 loại message:

| Message | Xử lý |
|---------|-------|
| `bytes` (binary) | Mở WS tới ASR `/ws/transcribe`, stream từng chunk |
| `{"type":"end_speech"}` | Gửi `{"type":"end"}` tới ASR → nhận transcript → gọi `orchestrator.process_text()` |
| `{"type":"text","text":"..."}` | Bỏ qua ASR, gọi thẳng `orchestrator.process_text()` |

### 2. `gateway/services/orchestrator.py` — Điều phối pipeline

Chiến lược **buffering để giảm latency**:
- Tích lũy text từ Brain stream vào `buffer`
- Khi `buffer >= 40 ký tự` HOẶC gặp dấu `.?!` → gọi TTS ngay, không đợi Brain xong
- Flush phần `buffer` còn lại sau khi Brain kết thúc

Events yield về WebSocket:

| Event type | Nội dung |
|------------|----------|
| `transcript` | Văn bản ASR nhận dạng được |
| `bot_response` | Text streaming từ Brain (is_final=False) hoặc kết thúc (is_final=True) |
| `audio_start` | Báo browser chuẩn bị nhận audio (sample_rate=24000) |
| `audio_chunk` | Raw PCM bytes → gửi `send_bytes()` |
| `error` | Lỗi với `code` và `message` |

### 3. `brain/grpc_handler.py` — BrainServiceHandler.think()

Pipeline trong 1 hàm async generator:

```
query
  → expand_query()           ~0ms  (regex dict lookup)
  → RAGPipeline.search()     ~50-200ms  (E5-large embed + Qdrant)
  → build_prompt()           ~0ms  (string concat)
  → LLMClient.generate_stream()   (Gemini stream, thinking_budget=0)
  → chunk_llm_stream()       (cắt câu để TTS dùng được)
  → yield {text, timing}
```

Timing metrics trên chunk đầu tiên: `expand_ms`, `rag_ms`, `llm_ttft_ms`, `first_chunk_total_ms`.

### 4. `brain/core/rag.py` — RAGPipeline

- **Embedding model**: `intfloat/multilingual-e5-large-instruct` (1024-dim), query prefix: `"Instruct: Tìm thông tin dinh dưỡng liên quan\nQuery: "`
- **Vector DB**: Qdrant cloud hoặc local embedded
- **Hỗ trợ snapshot restore**: upload `.snapshot` → poll đến khi collection `green`
- `search()` chạy `asyncio.to_thread` để không block event loop

### 5. `brain/core/llm.py` — LLMClient (factory)

Hai backend, chọn qua env `LLM_BACKEND`:

| Backend | Class | Khi nào dùng |
|---------|-------|--------------|
| `gemini` (default) | `GeminiLLMClient` | Gemini API, `thinking_budget=0` để giảm latency |
| `openai` | `OpenAILLMClient` | vLLM / Ollama / LM Studio, dùng OpenAI SDK |

`LLMClient(...)` là factory function, không phải class.

### 6. `brain/core/prompt.py` — Prompt engineering

- **System prompt**: bác sĩ dinh dưỡng, câu ngắn phù hợp TTS, không trích dẫn nguồn
- **Few-shot**: 2 ví dụ Q&A dinh dưỡng thực tế (từ benhvienthucuc.vn)
- **`build_prompt()`**: ghép few-shot + RAG context + conversation history (6 lượt gần nhất) + query

### 7. `brain/core/chunker.py` — chunk_llm_stream()

Cắt stream LLM thành các đoạn phù hợp để TTS:
- Min 40 ký tự
- Cắt tại dấu `.!?;:,` gần nhất (nhưng sau vị trí `min/2`)
- Nếu buffer > 80 ký tự mà không có dấu câu → cắt tại khoảng trắng gần nhất

### 8. `asr/core/transcriber.py` — Sherpa-Onnx

| Method | Dùng khi |
|--------|---------|
| `create_stream()` | Bắt đầu session mới |
| `accept_wave(stream, pcm)` | Đẩy từng chunk 100ms vào |
| `finalize_stream(stream)` | User dừng nói → flush buffer |
| `transcribe(pcm)` | Batch mode (audio file upload) |

Model: Online Transducer (encoder + decoder + joiner ONNX), 16kHz mono PCM int16.

### 9. `tts/grpc_handler.py` — TTSServiceHandler.speak()

- `chunk_text(text, min_size=60)` → danh sách câu (word-safe)
- Mỗi câu: chạy `synthesizer.synthesize_stream()` trong thread riêng
- Dùng `asyncio.Queue` để bridge thread → coroutine, yield PCM frame ngay khi có

### 10. `web/src/App.jsx` — React state machine

States: `idle → connecting → listening → thinking → speaking → idle`

| Action | Trigger |
|--------|---------|
| Bấm Call | Mở WebSocket, set `callActive=true` |
| Bấm mic (lần 1) | `startCapture()`, stream chunk 100ms qua WS |
| Bấm mic (lần 2) | Đợi 200ms padding → `stopCapture()` → gửi `end_speech` |
| Nhận binary WS | `playPcm(buf)` → AudioContext queue |
| Nhận `audio_start` | `resetPlayback()`, set status `speaking` |
| Nhận `bot_response` is_final | Set status `idle` |
| Upload file audio | Decode → resample 16kHz → Float32→Int16 → gửi 1 lần |

---

## Ghi chú kiến trúc

**Tên "grpc_handler" là tên lịch sử** — thực tế tất cả service giao tiếp qua HTTP/WebSocket, không có gRPC client. File `.proto` trong `protos/` chỉ là định nghĩa, không được dùng trong runtime.

**Streaming end-to-end để giảm latency** — Gateway không đợi Brain trả xong mới gọi TTS. Mỗi câu đủ dài (≥40 ký tự) được gửi TTS ngay lập tức trong khi Gemini vẫn đang generate phần tiếp theo.

**Dual-backend LLM** — Có thể chuyển sang vLLM/Ollama bằng cách set `LLM_BACKEND=openai` + `LLM_BASE_URL` + `LLM_MODEL` mà không sửa code.

**Conversation history** — Được truyền qua mỗi request từ Gateway xuống Brain (list JSON), lưu phía Gateway theo session. Không dùng Redis (khác với bản thiết kế v2).

---

## Audio specs

| | Value |
|---|---|
| Mic input (browser) | 48kHz → downsample 16kHz |
| ASR input | 16kHz, mono, PCM int16 LE |
| TTS output | 24kHz, mono, PCM int16 LE |
| TTS chunk size | ~4800 bytes (~100ms audio) |
