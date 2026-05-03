# Danh sách cải tiến — Nutrition Callbot

> File này dùng để theo dõi toàn bộ các cải tiến cần làm.  
> Mỗi mục có: mô tả vấn đề, file/dòng cụ thể, hướng fix, mức ưu tiên.  
> Đọc file này trước khi bắt đầu bất kỳ thay đổi nào.

---

## 🔴 BUG — Cần fix ngay

### ~~B1. Conversation history không bao giờ được cập nhật~~ ✅ DONE
**File:** `gateway/routes/websocket.py:23` và `gateway/services/orchestrator.py`  
**Vấn đề:** `conversation_history = []` được khởi tạo nhưng không bao giờ được append sau mỗi lượt hỏi-đáp. Brain luôn nhận list rỗng → bot không nhớ gì từ các câu trước trong cùng session. Tính năng multi-turn hoàn toàn không hoạt động dù code đã có trong `brain/core/prompt.py`.  
**Fix:**
```python
# Sau khi orchestrator.process_text() kết thúc, append vào history:
full_bot_text = ""
async for event in orchestrator.process_text(session_id, transcript, conversation_history):
    if event.get("type") == "bot_response" and not event.get("is_final"):
        full_bot_text += event.get("text", "")
    ...

conversation_history.append({"role": "user", "text": transcript})
conversation_history.append({"role": "assistant", "text": full_bot_text})
# Giữ sliding window 10 lượt
conversation_history = conversation_history[-20:]
```

---

### ~~B2. `synthesizer.cancel()` là hàm rỗng~~ ✅ DONE
**File:** `tts/core/synthesizer.py:111`  
**Vấn đề:** `cancel()` chỉ log, không thực sự dừng inference. Nếu implement barge-in ở gateway thì gọi cancel cũng vô dụng vì TTS vẫn chạy tiếp.  
**Fix:** Thêm `threading.Event` cancel flag vào `synthesize_stream()`, check flag trong vòng lặp infer_stream.
```python
def cancel(self, session_id: str):
    self._cancel_flag.set()

def synthesize_stream(self, text, ...):
    self._cancel_flag.clear()
    for audio_chunk in self._tts.infer_stream(...):
        if self._cancel_flag.is_set():
            return
        yield audio_i16.tobytes()
```

---

### ~~B3. Markdown từ Gemini vào TTS gây đọc sai~~ ✅ DONE
**File:** `gateway/services/orchestrator.py:32` (`_clean_for_tts`)  
**Vấn đề:** Gemini thường trả về `**text**`, `*text*`, `- bullet`, `### heading`. TTS đọc thành "dấu hoa thị text dấu hoa thị" hoặc "gạch ngang". `_clean_for_tts()` hiện chỉ collapse whitespace, không strip markdown.  
**Fix:**
```python
@staticmethod
def _clean_for_tts(text: str) -> str:
    import re
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)   # bold/italic
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # headings
    text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)   # bullet
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)       # links
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()
```

---

## 🟠 Cải tiến Performance

### P1. httpx.AsyncClient tạo mới cho mỗi request (không có connection pool)
**File:** `gateway/services/orchestrator.py:38,62,75`  
**Vấn đề:** Mỗi `_asr_transcribe`, `_brain_stream`, `_tts_stream` đều `async with httpx.AsyncClient(...) as client:` → tạo mới TCP connection mỗi lần, tốn ~10-50ms overhead.  
**Fix:** Tạo shared `AsyncClient` tại `__init__`, reuse qua mọi request. Đóng trong `aclose()`.
```python
def __init__(self):
    self._client = httpx.AsyncClient(timeout=self.http_timeout)

async def _asr_transcribe(self, ...):
    response = await self._client.post(...)  # dùng shared client
```

---

### P2. Retry + Exponential Backoff cho Gemini API
**File:** `brain/core/llm.py:77`  
**Vấn đề:** Khi Gemini trả lỗi 429 (rate limit) hoặc 503 (overload), code ngay lập tức yield error message. Một retry đơn giản giải quyết được phần lớn lỗi thoáng qua.  
**Fix:**
```python
import asyncio

MAX_RETRIES = 2
for attempt in range(MAX_RETRIES + 1):
    try:
        async for chunk in response:
            ...
        break
    except Exception as e:
        if attempt < MAX_RETRIES:
            await asyncio.sleep(2 ** attempt)  # 1s, 2s
            continue
        yield {"text": "Xin lỗi, hệ thống đang gặp sự cố...", "is_final": True, "error": True}
```

---

### P3. TTS `synthesize_stream` có max_chars=256 hardcoded
**File:** `tts/core/synthesizer.py:91`  
**Vấn đề:** `max_chars=256` là limit của VieNeu nhưng chunker ở TTS đã cắt text trước rồi (min 60 chars). Nếu chunk > 256 chars thì VieNeu tự truncate không báo. Cần đảm bảo chunker luôn cắt trước ngưỡng này.  
**Fix:** Đặt `max_chars` là config, đồng bộ với `_CHUNK_MIN_CHARS` trong `grpc_handler.py` (hiện 60, nên < 256 là OK nhưng cần document rõ).

---

### P4. Speculative RAG — search ngay khi ASR partial
**File:** `gateway/routes/websocket.py` và `brain/core/rag.py`  
**Vấn đề:** Hiện tại phải đợi user bấm dừng → ASR finalize → rồi mới bắt đầu RAG search. Có thể bắt đầu embed + search Qdrant ngay từ khi user còn đang nói (dùng partial transcript), đến khi có final thì chỉ cần re-rank hoặc dùng luôn kết quả.  
**Tiết kiệm:** ~50-200ms RAG latency.  
**Cách làm:** Trong ASR streaming session, mỗi khi `accept_wave` trả partial text có độ dài > 10 ký tự → trigger background RAG task, lưu kết quả vào dict theo `session_id`.

---

### P5. Semantic Query Cache
**Vấn đề:** Câu hỏi dinh dưỡng có pattern lặp lại cao ("bà bầu nên ăn gì", "trẻ biếng ăn bổ sung gì"). Mỗi lần đều gọi Gemini API tốn tiền + latency.  
**Fix:** Sau khi Brain trả lời, lưu `(embedding_vector, answer)` vào cache. Request mới: embed query → cosine similarity > 0.95 với cache → trả luôn, bỏ qua RAG + LLM.  
**Công nghệ:** In-memory dict (đơn giản) hoặc Redis (persist qua restart).

---

## 🟡 Tính năng UX

### U1. Barge-in (ngắt lời bot giữa chừng)
**Vấn đề:** Khi bot đang nói, user không thể ngắt. Phải đợi bot nói xong.  
**Cần làm:**
1. **Frontend** ([web/src/App.jsx](nutrition-callbot/web/src/App.jsx)): Trong state `speaking`, vẫn cho phép bấm mic. Khi mic bật → gửi `{"type": "barge_in"}` lên WS.
2. **Gateway** ([gateway/routes/websocket.py](nutrition-callbot/gateway/routes/websocket.py)): Khi nhận `barge_in` → set cancel flag để dừng TTS stream đang yield.
3. **TTS** ([tts/core/synthesizer.py](nutrition-callbot/tts/core/synthesizer.py)): Fix B2 trước — `cancel()` phải thực sự dừng.
4. **Context**: Lưu phần text bot đã nói trước khi bị cắt vào conversation history (đánh dấu `interrupted: true`).
5. **Frontend**: Gửi `{"type":"STOP_AUDIO"}` → `audioContext.close()` để dừng speaker ngay.

---

### U2. Client-side VAD — chỉ gửi audio khi có giọng nói
**File:** [web/src/App.jsx](nutrition-callbot/web/src/App.jsx) và `hooks/useAudioPlayer.js`  
**Vấn đề:** Hiện tại stream mọi chunk 100ms từ mic lên WS, kể cả tiếng ồn phòng. ASR phải xử lý nhiều hơn cần thiết.  
**Fix:** Tích hợp `@ricky0123/vad-web`:
```bash
npm install @ricky0123/vad-web
```
```jsx
import { useMicVAD } from "@ricky0123/vad-web"
// Chỉ gọi send(chunk) khi VAD detect speech, không phải mọi chunk
```

---

### U3. Filler Utterances — che giấu thời gian chờ
**Vấn đề:** Khoảng 500-1000ms từ lúc user dứt lời đến khi bot bắt đầu nói là im lặng. Gây cảm giác lag.  
**Fix:** Ngay khi Gateway nhận `end_speech`, trước khi Brain bắt đầu xử lý, phát ngay 1 audio ngắn từ thư viện pre-recorded:
```python
FILLERS = [
    "audio/filler_vang_a.pcm",
    "audio/filler_de_toi_kiem_tra.pcm",
    "audio/filler_toi_nghe_roi.pcm",
]
# Chọn random, yield ngay → TTS thật bắt đầu nối tiếp sau
```
Audio filler nên được TTS render offline 1 lần, lưu thành file `.pcm`.

---

### U4. Text input UI
**File:** [web/src/App.jsx](nutrition-callbot/web/src/App.jsx)  
**Vấn đề:** Backend hỗ trợ `{"type":"text"}` nhưng frontend không có text box. User không thể gõ câu hỏi khi mic hỏng hoặc môi trường ồn.  
**Fix:** Thêm `<input type="text">` + nút gửi bên cạnh nút mic. Gửi `JSON.stringify({type:"text", text: value})`.

---

### U5. Auto-reconnect WebSocket
**File:** `web/src/hooks/useWebSocket.js`  
**Vấn đề:** Khi WS disconnect (network flap, server restart), user phải bấm lại "Kết thúc" → "Gọi tư vấn". Không có auto-reconnect.  
**Fix:** Trong `useWebSocket`, khi `onclose` event → nếu `callActive` vẫn là true → sau 2s thử reconnect (tối đa 3 lần, exponential backoff).

---

## 🟢 Cải tiến RAG / AI Quality

### A1. Hybrid Search (Dense + Sparse)
**File:** `brain/core/rag.py`  
**Vấn đề:** Chỉ dùng dense vector search. Với các thuật ngữ cụ thể ("vitamin B12", "omega-3", "DHA") hoặc tên bệnh ("đái tháo đường type 2"), BM25/sparse search chính xác hơn dense.  
**Fix:**
- Đổi embedding model sang `BGE-M3` (hỗ trợ cả dense + sparse)
- Qdrant hỗ trợ hybrid search natively từ v1.7
- Search top-20 → rerank → top-5

---

### A2. Reranking với FlashRank
**File:** `brain/core/rag.py` (sau khi làm A1)  
**Vấn đề:** Sau dense search, kết quả top-5 chưa được sắp xếp lại theo relevance thực sự với query.  
**Fix:**
```bash
pip install flashrank  # ~80MB, CPU only, 30-50ms
```
```python
from flashrank import Ranker
ranker = Ranker()
results = ranker.rerank(query, top20_results)[:5]
```

---

### A3. Multi-query RAG
**File:** `brain/grpc_handler.py`  
**Vấn đề:** 1 câu hỏi → 1 query vector → search. Câu hỏi phức tạp hoặc đa nghĩa thì miss nhiều document liên quan.  
**Fix:** Dùng Gemini Flash (rẻ, nhanh) generate 2-3 sub-queries từ câu gốc, search song song, dedup kết quả:
```python
sub_queries = await llm.generate(f"Tạo 3 cách diễn đạt khác nhau cho câu hỏi: {query}")
# search song song 3 queries → merge → dedup by id → rerank
```

---

### A4. Query Expander — thêm fuzzy matching
**File:** `brain/core/query_expander.py`  
**Vấn đề:** Hiện tại chỉ dùng exact regex match. "bà bâu" (typo) hoặc "vitaminC" (không có khoảng trắng) sẽ không match.  
**Fix:** Thêm `difflib.SequenceMatcher` hoặc `rapidfuzz` để fuzzy match alias trước khi regex:
```python
from rapidfuzz import process
best_match, score, _ = process.extractOne(word, NUTRITION_ALIASES.keys())
if score > 85:
    expanded = expanded.replace(word, NUTRITION_ALIASES[best_match])
```

---

### A5. Conversation Summarization cho history dài
**File:** `gateway/routes/websocket.py` (sau khi fix B1)  
**Vấn đề:** Nếu hội thoại dài, history 10 lượt vẫn chiếm nhiều token trong prompt.  
**Fix:** Khi history > 6 lượt → dùng Gemini Flash tóm tắt các lượt cũ thành 2-3 câu, giữ nguyên 3 lượt gần nhất. Tránh prompt quá dài.

---

## 🔵 Reliability & Ops

### R1. Health check startup — kiểm tra Qdrant + Gemini trước khi nhận request
**File:** `brain/server.py`  
**Vấn đề:** Brain service start thành công nhưng nếu Qdrant không có collection hoặc Gemini API key sai, lỗi chỉ xuất hiện khi có request thật đầu tiên.  
**Fix:** Thêm startup event:
```python
@app.on_event("startup")
async def startup():
    rag._collection_has_data()  # raises nếu Qdrant lỗi
    await llm.generate("test", max_output_tokens=1)  # raises nếu key sai
```

---

### R2. Rate limiting per session
**File:** `gateway/routes/websocket.py`  
**Vấn đề:** Không có giới hạn số request. 1 session có thể spam gây tốn Gemini API.  
**Fix:** Đơn giản: dict `{session_id: (count, window_start)}`, max 20 requests/phút/session.

---

### R3. Query length validation
**File:** `gateway/routes/websocket.py:99` và `brain/grpc_handler.py`  
**Vấn đề:** Không có max length check. Query rất dài → tốn token, có thể gây lỗi Gemini.  
**Fix:**
```python
MAX_QUERY_LEN = 1000
if len(query) > MAX_QUERY_LEN:
    query = query[:MAX_QUERY_LEN]
```

---

### R4. Streaming cancel khi client disconnect giữa chừng
**File:** `gateway/services/orchestrator.py`  
**Vấn đề:** Khi client disconnect, `WebSocketDisconnect` được raise. Nhưng nếu đang ở giữa vòng `async for event in orchestrator.process_text(...)`, Brain + TTS vẫn tiếp tục chạy đến hết.  
**Fix:** Dùng `asyncio.Task` + cancellation:
```python
task = asyncio.create_task(handle_pipeline(...))
try:
    await task
except WebSocketDisconnect:
    task.cancel()
```

---

### R5. Opus codec thay PCM raw
**File:** `web/src/hooks/useAudioPlayer.js` và `gateway/routes/websocket.py`  
**Vấn đề:** Gửi raw PCM int16 ~256KB/s (mic) và ~384KB/s (TTS output 24kHz). Với mạng yếu hoặc nhiều users, bandwidth là bottleneck.  
**Fix:**
- Frontend: encode PCM → Opus trước khi gửi WS (`opus-recorder` hoặc WebCodecs API)
- Gateway: decode Opus → PCM trước khi gửi ASR
- TTS → Gateway: encode PCM → Opus trước khi gửi WS về browser
- Bandwidth giảm từ ~640KB/s → ~6KB/s (100x)

---

### R6. Session ID trả về client để resume
**File:** `gateway/routes/websocket.py:19`  
**Vấn đề:** Client không biết `session_id` của mình. Khi WS reconnect thì session_id mới → mất toàn bộ conversation history.  
**Fix:** Ngay sau `websocket.accept()`, gửi:
```python
await websocket.send_json({"type": "session_init", "session_id": session_id})
```
Client lưu session_id, khi reconnect gửi kèm → gateway restore history.

---

## ⚡ Tối ưu Latency

> Các mục dưới đây nhắm vào từng tầng của pipeline để giảm thời gian E2E. Một số đã nhắc trong conversation nhưng chưa có trong file — ghi lại đây để tiện theo dõi.

---

### L1. Brain và TTS chạy tuần tự — cần tách bằng asyncio.Queue *(tác động lớn nhất)*
**File:** `gateway/services/orchestrator.py:112-151`  
**Vấn đề:** Đây là bottleneck lớn nhất của pipeline. Khi Brain gửi câu đầu tiên, orchestrator gọi TTS và **block** toàn bộ vòng lặp — Brain stream bị tạm dừng trong suốt thời gian TTS đang synthesize. Hai service không bao giờ thực sự chạy song song.

```
Hiện tại (tuần tự):
Brain câu 1 [===]
TTS câu 1         [===]   ← Brain bị treo ở đây
Brain câu 2               [===]
TTS câu 2                       [===]
Tổng: 6 giai đoạn nối tiếp

Mục tiêu (song song):
Brain câu 1 [===]
Brain câu 2      [===]
Brain câu 3           [===]
TTS câu 1        [===]
TTS câu 2             [===]
TTS câu 3                  [===]
Tổng: tiết kiệm ~30-50% thời gian tổng
```

**Fix:** Dùng `asyncio.Queue` làm kênh giữa 2 coroutine:
```python
async def process_text(self, session_id, query, history):
    sentence_q = asyncio.Queue()   # Brain → TTS
    output_q   = asyncio.Queue()   # events → WebSocket

    async def _brain_producer():
        buffer = ""
        async for chunk in self._brain_stream(session_id, query, history):
            text = chunk.get("text", "")
            if text:
                await output_q.put({"type": "bot_response", "text": text, "is_final": False})
                buffer += text
                if self._ready_for_tts(buffer):
                    await sentence_q.put(self._clean_for_tts(buffer))
                    buffer = ""
        if buffer.strip():
            await sentence_q.put(self._clean_for_tts(buffer))
        await sentence_q.put(None)  # sentinel
        await output_q.put({"type": "bot_response", "text": "", "is_final": True})
        await output_q.put(None)

    async def _tts_consumer():
        sent_start = False
        while True:
            sentence = await sentence_q.get()
            if sentence is None:
                break
            if not sent_start:
                sent_start = True
                await output_q.put({"type": "audio_start", "sample_rate": 24000})
            async for pcm in self._tts_stream(sentence):
                await output_q.put({"type": "audio_chunk", "audio": pcm})

    producer = asyncio.create_task(_brain_producer())
    consumer = asyncio.create_task(_tts_consumer())
    await asyncio.gather(producer, consumer)

    while not output_q.empty():
        yield await output_q.get()
```
*(Cần refactor thêm vì async generator không yield xuyên coroutine — pattern trên là hướng, không copy nguyên)*

---

### L2. Orchestrator buffer redundant — Brain đã chunk sẵn câu
**File:** `gateway/services/orchestrator.py:26-29` và `brain/core/chunker.py`  
**Vấn đề:** Brain's `chunk_llm_stream()` đã cắt output LLM thành câu (min 40 chars, tại dấu câu). Orchestrator nhận được những câu chunk này rồi lại buffer thêm một lần nữa bằng `_ready_for_tts()`. Đây là **double-buffering không cần thiết** — mỗi brain chunk đã là một câu hoàn chỉnh và ngay lập tức trigger `_ready_for_tts` (vì đã >= 40 chars), nhưng vẫn tốn overhead string concat + check.  
**Fix:** Sau khi làm L1 (async Queue), bỏ buffer trong orchestrator — mỗi brain chunk là 1 sentence, push thẳng vào `sentence_q`.

---

### L3. Gemini Context Caching — cache system prompt + few-shot trên server Gemini
**File:** `brain/core/llm.py`  
**Vấn đề:** Mỗi request gửi lại toàn bộ: system prompt + 2 few-shot examples ≈ 800-1000 token input. Gemini API hỗ trợ explicit cache — upload phần context cố định 1 lần, các request sau chỉ gửi query + RAG context. Cache tồn tại 1h, tự renew.  
**Ước tính tiết kiệm:** ~100-200ms TTFT.  
**Fix:**
```python
from google.genai import caching
import datetime

# Khởi tạo 1 lần khi Brain service start
cached_content = await client.aio.caches.create(
    model="gemini-2.5-flash",
    config={
        "system_instruction": NUTRITION_SYSTEM_PROMPT,
        "contents": [{"role": "user", "parts": [{"text": few_shot_block}]}],
        "ttl": "3600s",
    },
)

# Trong generate_stream() dùng cached_content thay system_instruction
config = types.GenerateContentConfig(
    cached_content=cached_content.name,
    temperature=temperature,
    max_output_tokens=max_output_tokens,
    thinking_config=types.ThinkingConfig(thinking_budget=0),
)
```
**Lưu ý:** Gemini Context Caching yêu cầu tối thiểu 1024 token (flash models). Nếu system + few-shot < 1024 token thì cần thêm nội dung hoặc dùng cách khác.

---

### L4. Giảm max_output_tokens từ 4096 xuống 512
**File:** `brain/core/llm.py:34` (default `max_output_tokens=4096`)  
**Vấn đề:** Với voice, câu trả lời lý tưởng là 100-200 từ (~400-800 token). `max_output_tokens=4096` cho phép Gemini generate câu trả lời rất dài, mà với TTS thì đọc 4096 token sẽ mất hàng phút. Hơn nữa, giá trị này ảnh hưởng đến KV cache allocation của Gemini nội bộ.  
**Fix:** Giảm xuống 512. Nếu câu trả lời thực sự cần dài hơn thì tăng dần, nhưng 512 đã đủ cho mọi câu hỏi dinh dưỡng thông thường.
```python
# brain/core/llm.py
async def generate_stream(self, ..., max_output_tokens: int = 512):
```

---

### L5. Thêm chỉ thị độ dài vào system prompt
**File:** `brain/core/prompt.py:1`  
**Vấn đề:** System prompt hiện không giới hạn độ dài output. Gemini đôi khi generate câu trả lời rất dài dù câu hỏi đơn giản. Response dài hơn = generate lâu hơn + TTS lâu hơn.  
**Fix:** Thêm vào cuối `NUTRITION_SYSTEM_PROMPT`:
```python
"6. **Độ dài**: Trả lời trong 80-150 từ. Ưu tiên ngắn gọn, súc tích — người dùng nghe qua giọng nói, không đọc văn bản."
```

---

### L6. Embedding model: thay E5-Large bằng E5-Base
**File:** `brain/core/rag.py:14`  
**Vấn đề:** `multilingual-e5-large-instruct` có 560M tham số, mỗi lần embed query tốn ~150-300ms trên CPU. E5-Base (278M params) nhanh gấp đôi, độ chính xác giảm nhẹ (~2-3% trên benchmark).  
**So sánh:**

| Model | Params | Embed time (CPU) | Accuracy |
|-------|--------|-----------------|---------|
| e5-large | 560M | ~150-300ms | 100% (baseline) |
| e5-base | 278M | ~75-150ms | ~97% |
| e5-small | 117M | ~30-75ms | ~92% |

**Fix:** Đổi `MODEL_NAME` trong `rag.py`. Nếu accuracy giảm quá nhiều thì rollback.
```python
MODEL_NAME = "intfloat/multilingual-e5-base-instruct"  # thay e5-large
```

---

### L7. Qdrant HNSW ef_search parameter
**File:** `brain/core/rag.py:159`  
**Vấn đề:** Qdrant HNSW default `ef=128` — đây là accuracy/speed tradeoff. Với top-5 search trong corpus dinh dưỡng vài ngàn docs, `ef=64` vẫn đủ chính xác và nhanh hơn ~20-30%.  
**Fix:**
```python
from qdrant_client import models as qmodels

results = await asyncio.to_thread(
    self.qdrant.query_points,
    collection_name=self.collection,
    query=q_vec,
    limit=top_k,
    search_params=qmodels.SearchParams(hnsw_ef=64),  # thêm dòng này
    with_payload=True,
)
```

---

### L8. Embedding LRU cache — tránh re-embed query giống nhau
**File:** `brain/core/rag.py`  
**Vấn đề:** Cùng một câu hỏi được hỏi nhiều lần (ví dụ trong session conversation hoặc nhiều user hỏi giống nhau) → embed lại từ đầu mỗi lần. Tốn ~150-300ms mỗi lần.  
**Fix:** Cache embedding vector với LRU (max 200 entries):
```python
from functools import lru_cache

@lru_cache(maxsize=200)
def _embed_cached(self, text: str):
    return self.model.encode([QUERY_PREFIX + text], normalize_embeddings=True)[0].tolist()

# Trong search():
q_vec = await asyncio.to_thread(self._embed_cached, query)
```
**Lưu ý:** `lru_cache` không dùng được trực tiếp với method + `asyncio.to_thread` — cần wrapper.

---

### L9. Truncate RAG document content — bớt token vào prompt
**File:** `brain/grpc_handler.py:44`  
**Vấn đề:** Mỗi RAG document có thể dài 500-1000 ký tự. Top-5 docs = 2500-5000 ký tự trong prompt, chiếm phần lớn input token → Gemini xử lý lâu hơn.  
**Fix:** Cắt mỗi doc xuống tối đa 300 ký tự trước khi đưa vào prompt:
```python
context = "\n\n".join(
    f"[Tài liệu {i+1}: {d.get('title','')}]\n{_SOURCE_TAG.sub('', d.get('content',''))[:300]}"
    for i, d in enumerate(docs)
)
```

---

### L10. Giảm few-shot từ 2 xuống 1 example
**File:** `brain/core/prompt.py:84`  
**Vấn đề:** `FEW_SHOT_EXAMPLES[:2]` — mỗi example ~300-400 token. 2 examples = ~700-800 token thêm vào mỗi request. Gemini cần xử lý input này trước khi bắt đầu generate.  
**Ước tính tiết kiệm:** ~50-100ms TTFT.  
**Fix:**
```python
for ex in FEW_SHOT_EXAMPLES[:1]:  # chỉ 1 example, giữ cái rõ cấu trúc nhất
```

---

### L11. Giảm top_k RAG từ 5 xuống 3
**File:** `brain/grpc_handler.py:41`  
**Vấn đề:** Top-5 → 5 docs trong context. Top-3 → ít token hơn trong prompt (~400 token) và Qdrant query trả về ít data hơn.  
**Fix:**
```python
docs = await self.rag.search(expanded, top_k=3)  # thay top_k=5
```

---

### L12. Warm up HTTP connections khi Gateway khởi động
**File:** `gateway/services/orchestrator.py`  
**Vấn đề:** Request đầu tiên sau khởi động phải thiết lập TCP connection mới tới Brain và TTS → cold start ~50-200ms thêm.  
**Fix:** Thêm vào Gateway startup:
```python
@app.on_event("startup")
async def warmup():
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        try:
            await client.get(f"{settings.brain_http_url}/health")
            await client.get(f"{settings.tts_http_url}/health")
        except Exception:
            pass  # không fail startup nếu service chưa sẵn sàng
```

---

### L13. Gemini model configurable qua env — dễ switch sang model nhanh hơn
**File:** `brain/config.py:33`  
**Vấn đề:** `gemini_model: str = "gemini-2.5-flash"` hardcoded. Không thể switch sang model nhanh hơn (ví dụ `gemini-2.0-flash` hoặc `gemini-1.5-flash`) mà không sửa code.  
**Fix:**
```python
gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
```
Thêm vào `.env`:
```
GEMINI_MODEL=gemini-2.0-flash   # thử model nhanh hơn
```

---

### L14. Browser audio playback — giảm buffer trước khi phát
**File:** `web/src/hooks/useAudioPlayer.js`  
**Vấn đề:** Nếu browser buffer quá nhiều audio data trước khi bắt đầu phát, user cảm nhận thêm ~100-500ms chờ dù audio đã đến. Nên phát ngay khi có chunk đầu tiên.  
**Fix:** Đảm bảo `AudioContext.start()` được gọi ngay khi nhận chunk đầu tiên, không đợi buffer đầy. Kiểm tra lại `useAudioPlayer.js` để xem có `minBufferMs` hay threshold nào chưa.

---

## 📋 Tổng hợp theo thứ tự ưu tiên

**Bug (đã fix):**

| # | Mã | Mô tả | Effort | Impact |
|---|----|--------|--------|--------|
| ~~1~~ | ~~B1~~ | ~~Fix conversation history~~ | ~~Nhỏ~~ | ✅ Done |
| ~~2~~ | ~~B3~~ | ~~Strip markdown trước TTS~~ | ~~Nhỏ~~ | ✅ Done |
| ~~3~~ | ~~B2~~ | ~~Fix synthesizer.cancel()~~ | ~~Nhỏ~~ | ✅ Done |

**Latency — Quick wins (effort nhỏ, làm ngay):**

| # | Mã | Mô tả | Effort | Ước tính tiết kiệm |
|---|----|--------|--------|-------------------|
| 1 | L10 | Giảm few-shot 2→1 | 5ph | ~50-100ms TTFT |
| 2 | L11 | top_k RAG 5→3 | 5ph | ~20-50ms RAG + ~100 token |
| 3 | L4 | max_output_tokens 4096→512 | 5ph | Giới hạn độ dài câu trả lời |
| 4 | L5 | Thêm chỉ thị độ dài vào system prompt | 10ph | Ngắn hơn = nhanh hơn |
| 5 | L13 | Gemini model configurable qua env | 5ph | Cho phép thử model nhanh hơn |
| 6 | L9 | Truncate RAG doc 300 chars | 10ph | ~200-400 token ít hơn |
| 7 | L7 | Qdrant HNSW ef_search=64 | 10ph | ~20-30% Qdrant query |
| 8 | L12 | Warm up connections startup | 30ph | ~100-300ms cold start |
| 9 | P2 | Retry Gemini API | 1h | Giảm lỗi thoáng qua |
| 10 | R3 | Query length validation | 30ph | Defensive |

**Latency — Medium effort:**

| # | Mã | Mô tả | Effort | Ước tính tiết kiệm |
|---|----|--------|--------|-------------------|
| 11 | P1 | Shared httpx.AsyncClient | 1h | ~10-50ms/request |
| 12 | L8 | Embedding LRU cache | 2h | ~150-300ms cache hit |
| 13 | L6 | E5-Base thay E5-Large | 3h | ~75-150ms RAG |
| 14 | L3 | Gemini Context Caching | 4h | ~100-200ms TTFT |
| 15 | P5 | Semantic Query Cache | 2 ngày | ~800ms+ cache hit |
| 16 | P4 | Speculative RAG | 3 ngày | ~100-200ms RAG |

**Latency — Large refactor:**

| # | Mã | Mô tả | Effort | Ước tính tiết kiệm |
|---|----|--------|--------|-------------------|
| 17 | L1 | Async Queue Brain↔TTS | 2 ngày | ~30-50% tổng thời gian |
| 18 | L2 | Bỏ double-buffer orchestrator | 1h (sau L1) | Simplify code |

**UX:**

| # | Mã | Mô tả | Effort | Impact |
|---|----|--------|--------|--------|
| 19 | U4 | Text input UI | 2h | Fallback khi mic lỗi |
| 20 | U3 | Filler utterances | 1 ngày | Ẩn latency |
| 21 | U2 | Client-side VAD | 1-2 ngày | Giảm tải ASR |
| 22 | U1 | Barge-in | 2-3 ngày | UX thay đổi rõ nhất |
| 23 | U5 | Auto-reconnect WS | 1 ngày | Reliability |
| 24 | L14 | Browser audio buffer | 1h | ~100-500ms perceived |

**AI Quality:**

| # | Mã | Mô tả | Effort | Impact |
|---|----|--------|--------|--------|
| 25 | A1+A2 | Hybrid Search + FlashRank | 3-4 ngày | Tăng accuracy RAG |
| 26 | A3 | Multi-query RAG | 1 ngày | Tăng recall |
| 27 | A4 | Fuzzy query expander | 2h | Edge cases |
| 28 | A5 | Conversation summarization | 1 ngày | History dài |

**Reliability & Ops:**

| # | Mã | Mô tả | Effort | Impact |
|---|----|--------|--------|--------|
| 29 | R1 | Startup health check | 1h | Ops |
| 30 | R2 | Rate limiting | 2h | Ops |
| 31 | R4 | Cancel stream khi disconnect | 1 ngày | Resource leak |
| 32 | R5 | Opus codec | 1 tuần | Bandwidth |
| 33 | R6 | Session ID resume | 1 ngày | Prereq resume |
| 34 | P3 | TTS max_chars config | 30ph | Defensive |
