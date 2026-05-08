# Hướng dẫn Tuning Pipeline Callbot

> Mục tiêu: tối ưu đồng thời **latency** (TTFB ≤ 1.5s) và **quality** (RAGAS faithfulness ≥ 0.75, context_recall ≥ 0.70)

---

## 1. Framework đo lường

Trước khi thay đổi bất cứ thứ gì, **đo baseline** trên cả hai trục:

### 1a. Latency — dùng `eval_pipeline_e2e.py`
```bash
cd evaluation
python eval_pipeline_e2e.py --eval-dirs eval_1 --sample 30 --concurrency 1
```
Metrics quan trọng: `speech_end_to_first_audio_ms` (TTFB), `brain_first_flush_ms`, `asr_ms`, `tts_first_chunk_ms`

### 1b. Quality — dùng `eval_brain_responses.py` (không cần audio, nhanh hơn)
```bash
python evaluation/eval_brain_responses.py --sample 50 --concurrency 2
```
Metrics: `answer_relevancy`, `faithfulness`, `context_precision`, `context_recall`

### 1c. Ghi kết quả baseline vào bảng tracking
Mỗi lần thay param → chạy lại → so sánh với baseline. Cấu trúc bảng:

| Config | TTFB (p50) | TTFB (p90) | faithfulness | context_recall | Ghi chú |
|---|---|---|---|---|---|
| baseline | _ms | _ms | _ | _ | Cấu hình hiện tại |
| fetch_k=10 | _ms | _ms | _ | _ | ... |

---

## 2. Thứ tự tuning — từ impact cao đến thấp

```
Pha 1: Quick wins (không ảnh hưởng quality)
  → OpenAI max_output_tokens, gpu_memory_utilization, ASR provider

Pha 2: RAG (ảnh hưởng cả latency lẫn quality)
  → fetch_k, top_k

Pha 3: Chunking (latency vs audio naturalness)
  → MIN_CHUNK_SIZE

Pha 4: LLM generation (quality vs latency)
  → temperature, max_output_tokens (Gemini)

Pha 5: Fine-tuning nhỏ
  → few-shot examples, conversation history depth, TTS params
```

---

## 3. Pha 1 — Quick Wins

### 3.1 Fix `max_output_tokens=512` (OpenAI/vLLM backend)

**File:** `nutrition-callbot/brain/core/llm.py`, dòng ~116

```python
# Hiện tại — quá thấp, sẽ truncate câu trả lời dài
max_output_tokens: int = 512,

# Sửa thành
max_output_tokens: int = 1500,
```

**Không cần đo:** đây là bug fix, không phải trade-off. 512 tokens ≈ ~380 từ tiếng Việt — không đủ cho nhiều câu trả lời dinh dưỡng chi tiết.

---

### 3.2 Tăng `gpu_memory_utilization` (nếu chạy vLLM trên GPU riêng)

**File:** `docker-compose.gpu.yml`

```yaml
# Hiện tại
--gpu-memory-utilization 0.45

# Thử các mức: 0.55 → 0.65 → 0.75
# Dừng khi bắt đầu OOM
```

**Đo:** `brain_first_flush_ms` trong e2e eval. Kỳ vọng giảm 10–25%.

---

### 3.3 Bật GPU cho ASR (nếu có GPU khả dụng)

**File:** `nutrition-callbot/.env` hoặc `asr/config.py`

```bash
# .env
ASR_PROVIDER=cuda
ASR_NUM_THREADS=1   # không cần nhiều thread khi có GPU
```

**Đo:** `asr_ms`. Kỳ vọng giảm từ ~200–400ms xuống ~50–100ms cho file 5–10s.

---

## 4. Pha 2 — Tuning RAG (`fetch_k` và `top_k`)

Đây là tham số ảnh hưởng **nhiều nhất** đến cả quality lẫn latency của Brain.

### Thiết lập experiment

Expose `fetch_k` và `top_k` qua config để dễ thay đổi.

**Sửa `nutrition-callbot/brain/core/rag.py`:**

```python
# Thay dòng fetch_k = 15 hardcoded thành nhận từ ngoài vào
async def search(self, query: str, filters: dict = None,
                 top_k: int = 5, fetch_k: int = 15) -> List[Dict]:
```

**Sửa `nutrition-callbot/brain/grpc_handler.py`** — truyền từ config:

```python
# Trong BrainConfig (brain/config.py), thêm:
rag_fetch_k: int = int(os.getenv("RAG_FETCH_K", "15"))
rag_top_k: int   = int(os.getenv("RAG_TOP_K", "5"))
```

### Grid cần thử

| `fetch_k` | `top_k` | Kỳ vọng |
|---|---|---|
| 15 | 5 | Baseline |
| 10 | 5 | Ít rerank hơn → nhanh hơn ~15%, quality giảm nhẹ |
| 20 | 5 | Rerank nhiều hơn → quality tăng, latency +20% |
| 15 | 3 | Prompt ngắn hơn → LLM nhanh hơn, nhưng context ít |
| 15 | 7 | Nhiều context hơn → faithfulness có thể tăng |
| 10 | 3 | Latency thấp nhất nhóm này |

### Script chạy ablation

```bash
# Chạy eval với từng cấu hình — thay env var và restart brain
RAG_FETCH_K=10 RAG_TOP_K=5 \
  python evaluation/eval_brain_responses.py --sample 50 --concurrency 2

# So sánh RAGAS scores và brain_first_flush_ms
```

### Heuristic chọn kết quả

- Nếu `context_recall` giảm khi giảm `fetch_k` → giữ nguyên hoặc tăng
- Nếu `faithfulness` không đổi khi giảm `top_k` → giảm `top_k` để tiết kiệm token
- **Điểm ngọt thường ở: `fetch_k=12, top_k=4`** (giảm nhẹ cả hai)

---

## 5. Pha 3 — Tuning `MIN_CHUNK_SIZE` (Brain → TTS latency)

`MIN_CHUNK_SIZE` quyết định khi nào Brain gửi text sang TTS. Nhỏ hơn = TTS bắt đầu sớm hơn nhưng audio bị đứt câu.

### Expose qua env

**`brain/config.py`** đã có `min_chunk_size: int = 40`, nhưng chưa dùng trong `chunker.py`.

**Sửa `brain/core/chunker.py`:**
```python
# Truyền min_size từ config thay vì hardcode
# Gọi: chunk_llm_stream(stream, min_size=config.min_chunk_size)
```

**Sửa `gateway/services/orchestrator.py`** — `_ready_for_tts` cũng hardcode 40:
```python
def _ready_for_tts(buf: str, min_chars: int = 40) -> bool:
    ...
# Đọc min_chars từ env GATEWAY_TTS_MIN_CHARS
```

### Ranges cần thử

| `MIN_CHUNK_SIZE` | `brain_first_flush_ms` | Chất lượng audio |
|---|---|---|
| 20 | Rất thấp | TTS hay bị ngắt giữa câu |
| 30 | Thấp | Đôi khi ngắt |
| 40 | Baseline | Ổn |
| 60 | Cao hơn ~15–20% | Câu đầy đủ hơn |
| 80 | Cao nhất | Rất tự nhiên nhưng chậm hơn |

**Đo latency:** `brain_first_flush_ms` trong e2e eval.  
**Đo audio quality:** nghe thử bằng tai — không có metric tự động tốt cho việc này.

**Khuyến nghị:** nếu TTFB đã dưới mục tiêu 1.5s → thử tăng lên 50–60 để audio tự nhiên hơn.

---

## 6. Pha 4 — Tuning LLM Generation

### 6.1 Temperature

| `temperature` | Tính chất output | Phù hợp khi |
|---|---|---|
| 0.1 | Rất deterministic | Câu trả lời cần chính xác cao (liều lượng, dinh dưỡng y tế) |
| 0.3 | Balanced (hiện tại) | OK cho domain dinh dưỡng tư vấn |
| 0.5 | Đa dạng hơn | Câu trả lời sáng tạo, tránh lặp lại |
| 0.7+ | Quá creative | Không phù hợp cho tư vấn y tế |

**Đo:** `answer_relevancy` trong RAGAS. Thường không ảnh hưởng latency.

### 6.2 `max_output_tokens` (Gemini)

4096 là đủ. Không cần tăng. Giảm xuống 1500–2000 nếu muốn force câu trả lời ngắn gọn hơn (giảm `brain_total_ms`), nhưng kết hợp với việc giảm giới hạn trong system prompt.

### 6.3 System prompt — thay "80 từ" bằng giá trị khác

**File:** `brain/core/prompt.py`

```
# Hiện tại trong system prompt:
"tối đa 80 từ"

# Thử:
"tối đa 100 từ" → câu trả lời đầy đủ hơn, brain_total_ms tăng
"tối đa 60 từ" → ngắn gọn hơn, latency thấp hơn
```

**Đo đồng thời:** `faithfulness` (câu trả lời còn đủ thông tin không?) và `brain_total_ms`.

---

## 7. Pha 5 — Fine-tuning nhỏ

### 7.1 Few-shot examples

**File:** `brain/core/prompt.py`

```python
FEW_SHOT_EXAMPLES[:2]  # Hiện dùng 2/3

# Thử:
FEW_SHOT_EXAMPLES[:1]  # 1 example → prompt ngắn hơn, TTFT thấp hơn
FEW_SHOT_EXAMPLES[:3]  # 3 examples → LLM hiểu format tốt hơn
FEW_SHOT_EXAMPLES[:0]  # 0 examples → nhanh nhất, chất lượng format có thể giảm
```

**Đo:** `answer_relevancy` và `llm_ttft_ms`.

### 7.2 Conversation history depth

```python
conversation_history[-6:]  # Hiện dùng 6 turns

# Thử:
conversation_history[-4:]  # Ít context hơn → prompt ngắn hơn → nhanh hơn
conversation_history[-8:]  # Nhớ lâu hơn → chatbot coherent hơn trong session dài
```

### 7.3 TTS streaming params

**File:** `tts/config.py`

| Tham số | Hiện tại | Thử | Kỳ vọng |
|---|---|---|---|
| `streaming_frames_per_chunk` | 15 | 10 | TTFA thấp hơn, audio jitter hơn |
| `streaming_lookforward` | 5 | 3 | TTFA thấp hơn nhẹ |
| `temperature` (TTS) | 1.0 | 0.8 | Voice ít biến động hơn |

---

## 8. Workflow thực tế — Chạy thế nào

### Bước 1: Tạo baseline

```bash
# Terminal 1: khởi động services
./scripts/start_local_4_services.sh

# Terminal 2: chạy baseline eval
python evaluation/eval_brain_responses.py \
  --sample 80 --concurrency 2 \
  --out evaluation/results/tuning_baseline.jsonl

python evaluation/eval_pipeline_e2e.py \
  --sample 30 --concurrency 1 \
  --out-dir evaluation/results/
```

### Bước 2: Thay một tham số, restart service bị ảnh hưởng

```bash
# Ví dụ: thay RAG_FETCH_K
RAG_FETCH_K=10 RAG_TOP_K=5 uvicorn brain.main:app --port 50052 --reload

# Chạy lại eval
python evaluation/eval_brain_responses.py \
  --sample 80 --concurrency 2 \
  --out evaluation/results/tuning_fetch_k10_top_k5.jsonl
```

### Bước 3: So sánh kết quả

```python
# Script nhanh để compare 2 result files
import json, statistics

def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]

baseline = load("evaluation/results/tuning_baseline.jsonl")
variant  = load("evaluation/results/tuning_fetch_k10_top_k5.jsonl")

for key in ["faithfulness", "answer_relevancy", "context_recall"]:
    b = [r.get("ragas", {}).get(key) for r in baseline if r.get("ragas", {}).get(key)]
    v = [r.get("ragas", {}).get(key) for r in variant  if r.get("ragas", {}).get(key)]
    if b and v:
        print(f"{key:25s}: baseline={statistics.mean(b):.3f}  variant={statistics.mean(v):.3f}  Δ={statistics.mean(v)-statistics.mean(b):+.3f}")

for key in ["rag_ms", "llm_ttft_ms", "total_ms"]:
    b = [r.get("timing", {}).get(key) for r in baseline if r.get("timing", {}).get(key)]
    v = [r.get("timing", {}).get(key) for r in variant  if r.get("timing", {}).get(key)]
    if b and v:
        print(f"{key:25s}: baseline={statistics.mean(b):.0f}ms  variant={statistics.mean(v):.0f}ms  Δ={statistics.mean(v)-statistics.mean(b):+.0f}ms")
```

### Bước 4: Lưu config tốt nhất

Khi tìm được config tốt, cập nhật `.env` hoặc `config.py` và commit.

---

## 9. Pareto Frontier — Cân bằng Latency vs Quality

Với domain tư vấn dinh dưỡng, **faithfulness quan trọng hơn latency**. Người dùng thà chờ thêm 0.5s còn hơn nhận thông tin sai về liều lượng vitamin hay chế độ ăn.

### Ngưỡng đề xuất

| Metric | Không chấp nhận | Chấp nhận | Tốt |
|---|---|---|---|
| TTFB (p50) | > 3s | 1.5–3s | < 1.5s |
| TTFB (p90) | > 5s | 2–5s | < 2s |
| faithfulness | < 0.65 | 0.65–0.75 | > 0.75 |
| context_recall | < 0.60 | 0.60–0.70 | > 0.70 |
| answer_relevancy | < 0.70 | 0.70–0.80 | > 0.80 |

### Rule of thumb

- Nếu faithfulness < 0.70 → ưu tiên tăng `top_k` trước
- Nếu context_recall < 0.65 → tăng `fetch_k` trước  
- Nếu TTFB > 3s → giảm `MIN_CHUNK_SIZE` và `fetch_k` trước
- Nếu TTFB ổn nhưng audio đứt quãng → tăng `MIN_CHUNK_SIZE`

---

## 10. Ablation Study cho Đồ án

### Danh sách ablation hoàn chỉnh

| # | Ablation | Loại | Metric chính |
|---|---|---|---|
| 1 | fetch_k × top_k (3 config) | Hyperparameter | context_recall, rag_ms |
| 2 | With vs Without reranker | Component | context_precision, rag_ms |
| 3 | With vs Without query expansion | Component | context_recall, answer_relevancy |
| 4 | min_chunk_size (3 config) | Hyperparameter | brain_first_flush_ms |

> **Lưu ý:** `temperature` của pipeline LLM fix cứng = 0 — RAG chỉ cần tóm tắt context, không cần sáng tạo. Không đưa vào ablation.

---

### Ablation 1 — fetch_k × top_k (3 config)

| Config | fetch_k | top_k | Mục đích |
|---|---|---|---|
| Conservative | 10 | 3 | Latency thấp, context ít |
| Baseline | 15 | 5 | Cấu hình hiện tại |
| Aggressive | 20 | 7 | Quality tối đa |

**Câu hỏi nghiên cứu:** Tăng độ sâu retrieval có thực sự cải thiện context_recall không, hay chỉ tăng latency?

---

### Ablation 2 — With vs Without Reranker

Tắt bước rerank, dùng thẳng raw Qdrant cosine score. Sửa `rag.py`:

```python
# Bỏ reranker, trả thẳng top_k từ Qdrant
reranked = docs[:top_k]  # thay vì chạy cross-encoder
```

| Config | context_precision | context_recall | rag_ms |
|---|---|---|---|
| No reranker | ? | ? | thấp hơn |
| With reranker (baseline) | ? | ? | cao hơn |

**Câu hỏi nghiên cứu:** Cross-encoder reranker có đáng đánh đổi latency không?

---

### Ablation 3 — With vs Without Query Expansion

Tắt `expand_query()`, truyền raw query thẳng vào RAG. Sửa `grpc_handler.py`:

```python
# expanded = expand_query(query)  ← comment out
expanded = query  # dùng raw query
```

| Config | context_recall | answer_relevancy |
|---|---|---|
| No expansion | ? | ? |
| With expansion (baseline) | ? | ? |

**Câu hỏi nghiên cứu:** NUTRITION_ALIASES có thực sự cải thiện retrieval cho domain dinh dưỡng tiếng Việt không?

---

### Ablation 4 — min_chunk_size (latency vs naturalness)

| Config | min_chunk_size | brain_first_flush_ms (p50) | brain_first_flush_ms (p90) |
|---|---|---|---|
| Fast | 25 | ? | ? |
| Baseline | 40 | ? | ? |
| Natural | 70 | ? | ? |

**Câu hỏi nghiên cứu:** Giảm chunk size có giảm TTFB đáng kể không? Trade-off với độ tự nhiên của audio ra sao?

> Ablation 4 đo bằng `eval_pipeline_e2e.py`, không cần RAGAS.

---

### Tại sao Component Ablation (2, 3) quan trọng hơn Hyperparameter (1, 4)

Fetch_k/top_k là tuning số — không ai tranh luận việc tăng/giảm có ảnh hưởng. Reranker và query expansion là **component ablation** — chứng minh từng thành phần có đóng góp gì vào hệ thống. Đây là cấu trúc mà reviewer đồ án mong đợi và là nền tảng để justify các design decision.

---

### Setup RAGAS judge

```python
# Dùng Gemini Flash làm judge (khác family với Qwen pipeline → tránh self-bias)
# temperature=0 bắt buộc — judge phải deterministic và reproducible
ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Nếu dùng Qwen 27B làm judge: thêm vào thesis note limitation self-family bias
# extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

---

### Sample size

- **50 câu mỗi config** cho RAGAS (CI 95% ≈ ±0.03 — phân biệt được diff ≥ 0.05)
- **30 file audio** cho latency eval
- Báo cáo **mean ± std**, không chỉ mean

---

## 11. Các thay đổi KHÔNG nên làm

| Thay đổi | Lý do |
|---|---|
| Tắt reranker (dùng raw Qdrant score) | Context_recall giảm đáng kể |
| Giảm `fetch_k` xuống < 8 | Reranker không có đủ ứng viên để chọn |
| Bật `thinking_budget > 0` trên Gemini | Latency tăng vài giây, không đáng cho tư vấn thường |
| Tăng `temperature` TTS > 1.2 | Voice mất tự nhiên, có thể bị noise |
| Giảm `conversation_history` xuống 0 | Chatbot mất ngữ cảnh hoàn toàn trong multi-turn |
