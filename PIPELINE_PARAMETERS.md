# Tham số Tunable — Toàn bộ Pipeline Callbot

> Tài liệu này tổng hợp tất cả các tham số có thể điều chỉnh trong pipeline ASR → Brain → TTS → Gateway.
> Cập nhật: 2026-05-07

---

## 1. LLM (`nutrition-callbot/brain/core/llm.py`)

### Gemini backend (`LLM_BACKEND=gemini`, mặc định)

| Tham số | Giá trị hiện tại | Ghi chú |
|---|---|---|
| `model` | `gemini-2.5-flash` | Đặt qua `LLM_MODEL` env hoặc `brain/config.py` |
| `temperature` | `0.3` | Hardcoded default, không expose env |
| `max_output_tokens` | `4096` | Đủ dùng cho câu trả lời dinh dưỡng |
| `thinking_budget` | `0` | Tắt thinking để giảm latency |

### OpenAI-compatible backend (`LLM_BACKEND=openai`, dùng với vLLM/Ollama)

| Tham số | Giá trị hiện tại | Ghi chú |
|---|---|---|
| `model` | Từ `LLM_MODEL` env | Ví dụ: `Qwen/Qwen3-8B` |
| `base_url` | `LLM_BASE_URL` env | Mặc định `http://localhost:8000/v1` |
| `temperature` | `0.3` | Hardcoded |
| `max_output_tokens` | `512` | ⚠️ **Critically low** — có thể cắt câu trả lời dài; nên tăng lên 1024–2048 |
| `enable_thinking` | `False` | Tắt thinking Qwen3 (`chat_template_kwargs`) |

**Cách điều chỉnh:** Sửa trực tiếp trong `llm.py` hoặc thêm env var và đọc trong `__init__`.

---

## 2. vLLM Server (`docker-compose.gpu.yml`)

Áp dụng khi deploy vLLM thay Gemini.

| Tham số CLI | Giá trị hiện tại | Ghi chú |
|---|---|---|
| `--max-model-len` | `4096` | Context window tối đa |
| `--gpu-memory-utilization` | `0.45` | Conservative — nếu không share GPU có thể tăng lên 0.7–0.8 |
| `--dtype` | `half` (float16) | Tiết kiệm VRAM |
| `--quantization` | `bitsandbytes` | 4-bit/8-bit quant |
| `--enable-prefix-caching` | (flag) | Tăng tốc với system prompt lặp lại nhiều |
| `--tensor-parallel-size` | Không set | Thêm nếu có multi-GPU |
| `--max-num-seqs` | Không set | Throughput khi nhiều request đồng thời |

---

## 3. RAG Pipeline (`nutrition-callbot/brain/core/rag.py`)

| Tham số | Giá trị hiện tại | Ghi chú |
|---|---|---|
| `fetch_k` | `15` | Số candidates fetch từ Qdrant trước rerank; hardcoded |
| `top_k` | `5` | Số docs trả về sau rerank; tham số của `search()` |
| `QUERY_PREFIX` | `"Instruct: Tìm thông tin dinh dưỡng liên quan\nQuery: "` | Prefix cho E5-instruct; thay đổi nếu đổi domain |
| `MODEL_NAME` | `intfloat/multilingual-e5-large-instruct` | Embedding model; hardcoded |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker; hardcoded |
| `normalize_embeddings` | `True` | Chuẩn hóa vector trước khi query |

**Trade-off `fetch_k / top_k`:**
- `fetch_k` cao → reranker thấy nhiều ứng viên hơn → chất lượng tốt hơn nhưng latency cao hơn
- `top_k` cao → LLM nhận nhiều context hơn → câu trả lời đầy đủ hơn nhưng prompt dài hơn

---

## 4. Brain Chunker (`nutrition-callbot/brain/core/chunker.py`)

Chunker chia response LLM thành từng đoạn để gửi sang TTS.

| Tham số | Giá trị hiện tại | Ghi chú |
|---|---|---|
| `MIN_CHUNK_SIZE` | `40` chars | Kích thước tối thiểu của mỗi chunk |
| `PUNCTUATION` | `[.!?;:,।।]` | Các ký tự cắt câu ưu tiên |
| Hard break | `MIN_CHUNK_SIZE * 2 = 80` chars | Cắt tại ranh giới từ nếu quá dài |

**Trade-off:** `MIN_CHUNK_SIZE` nhỏ → TTS nhận chunk sớm hơn (latency thấp) nhưng audio nghe đứt quãng hơn.

---

## 5. System Prompt & Few-Shot (`nutrition-callbot/brain/core/prompt.py`)

| Tham số | Giá trị hiện tại | Ghi chú |
|---|---|---|
| Few-shot examples | 2 / 3 examples dùng (`FEW_SHOT_EXAMPLES[:2]`) | Bỏ example thứ 3 để giảm token |
| Giới hạn từ | `"tối đa 80 từ"` | Constraint trong system prompt |
| Conversation history | `conversation_history[-6:]` | Giữ 6 turn gần nhất (3 Q&A) |

---

## 6. Query Expander (`nutrition-callbot/brain/core/query_expander.py`)

| Tham số | Giá trị hiện tại | Ghi chú |
|---|---|---|
| `NUTRITION_ALIASES` | ~25 entries | Dict chuẩn hóa thuật ngữ dinh dưỡng tiếng Việt |
| Regex flags | `re.IGNORECASE` | Case-insensitive |

> **Lưu ý:** `voice_preprocessing.py` (LEGAL_ALIASES) là code legacy từ legal bot cũ — **không được dùng** trong pipeline hiện tại.

---

## 7. Gateway Orchestrator (`nutrition-callbot/gateway/services/orchestrator.py`)

| Tham số | Giá trị hiện tại | Ghi chú |
|---|---|---|
| `_ready_for_tts min_chars` | `40` | Hardcoded — phải khớp với Brain chunker |
| `tts_queue maxsize` | `4` | Buffer queue giữa Brain và TTS |
| TTS chunk size | `4800` bytes | ~0.1s audio ở 24kHz PCM int16 |
| HTTP timeout (connect) | `10s` | |
| HTTP timeout (read) | `120s` | Đủ cho câu trả lời dài |
| HTTP timeout (write) | `30s` | |
| HTTP timeout (pool) | `30s` | |

---

## 8. TTS (`nutrition-callbot/tts/`)

### Config (`tts/config.py`)

| Tham số | Giá trị hiện tại | Ghi chú |
|---|---|---|
| `backbone_repo` | `pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf` | GGUF q4 quantized |
| `codec_repo` | `neuphonic/distill-neucodec` | |
| `streaming_frames_per_chunk` | `15` | Số frames mỗi chunk streaming |
| `streaming_lookforward` | `5` | Look-ahead frames |
| `min_chunk_size` | `40` | Ký tự tối thiểu trước khi bắt đầu stream |
| `sample_rate` | `24000` Hz | Output audio rate |

### Synthesizer (`tts/core/synthesizer.py` — `infer_stream`)

| Tham số | Giá trị hiện tại | Ghi chú |
|---|---|---|
| `temperature` | `1.0` | Diversity của voice synthesis |
| `top_k` | `50` | Sampling parameter |
| `max_chars` | `256` | Giới hạn ký tự mỗi lần gọi |
| `backbone_device` | `"gpu"` (llama.cpp) | |
| `codec_device` | `"cuda"` (PyTorch) | |

---

## 9. ASR (`nutrition-callbot/asr/`)

### Config (`asr/config.py`)

| Tham số | Giá trị / Env var | Ghi chú |
|---|---|---|
| `num_threads` | `4` / `ASR_NUM_THREADS` | CPU threads cho sherpa-onnx |
| `provider` | `"cpu"` / `ASR_PROVIDER` | `"cpu"` hoặc `"cuda"` |
| Model files | `asr/data/` | Zipformer-30M RNNT 6000h (encoder, decoder, joiner, tokens) |

---

## 10. Tổng hợp — Các tham số đáng tune nhất

| Ưu tiên | Tham số | Vị trí | Lý do |
|---|---|---|---|
| 🔴 Cao | `max_output_tokens=512` (OpenAI backend) | `llm.py` | Quá thấp, có thể cắt câu trả lời dinh dưỡng dài |
| 🔴 Cao | `gpu_memory_utilization=0.45` | `docker-compose.gpu.yml` | Conservative, nếu không share GPU nên tăng |
| 🟡 Trung | `fetch_k=15` / `top_k=5` | `rag.py` | Ảnh hưởng chất lượng retrieval và latency |
| 🟡 Trung | `temperature=0.3` | `llm.py` | Có thể thử 0.1–0.5 tùy mức độ sáng tạo mong muốn |
| 🟡 Trung | `MIN_CHUNK_SIZE=40` | `chunker.py` + `orchestrator.py` | Trade-off latency vs ngắt câu tự nhiên |
| 🟢 Thấp | `conversation_history[-6:]` | `prompt.py` | Tăng nếu cần nhớ context dài hơn |
| 🟢 Thấp | `thinking_budget=0` | `llm.py` | Bật nếu cần câu trả lời phức tạp hơn (tăng latency) |
| 🟢 Thấp | `streaming_frames_per_chunk=15` | `tts/config.py` | Điều chỉnh granularity streaming TTS |

---

## 11. Các tham số không phải env var (phải sửa code)

Các tham số sau **chưa được expose** qua `.env` hoặc `config.py` — phải sửa trực tiếp trong source:

- `temperature`, `max_output_tokens` trong `llm.py`
- `fetch_k`, `QUERY_PREFIX`, model names trong `rag.py`
- `MIN_CHUNK_SIZE`, `PUNCTUATION` trong `chunker.py`
- `min_chars=40` trong `orchestrator.py`
- `temperature`, `top_k`, `max_chars` trong `synthesizer.py`
