# Future Innovations — Nutrition Voicebot

Tổng hợp các hướng cải tiến dựa trên nghiên cứu 2024–2025.  
Hệ thống hiện tại là **cascade pipeline** (ASR→Brain→TTS), đây là kiến trúc đúng cho tiếng Việt do thiếu data và tài nguyên để train end-to-end speech LLM.

---

## Bối cảnh: Cascade vs End-to-End

| | Cascade (hiện tại) | End-to-End SpeechLM |
|---|---|---|
| Latency | 400–600ms | 200–320ms |
| Vietnamese support | Tốt (VieNeu-TTS, sherpa-onnx) | Gần như chưa có |
| Debug | Dễ từng layer | Khó |
| Domain tuning | Dễ | Phải train lại toàn bộ |
| Tài nguyên train | Thấp | Cần hàng trăm nghìn giờ audio |

GPT-4o (320ms), Moshi/Kyutai (200ms), VITA-Audio (NeurIPS 2025) đều đi theo hướng end-to-end — nhưng không có Vietnamese data. Cascade được tối ưu tốt có thể đạt 300–500ms, đủ cho production theo X-Talk (arxiv:2512.18706).

---

## Những gì đã có ✅

- Streaming end-to-end pipeline (ASR→Brain→TTS)
- Barge-in (user ngắt bot đang nói)
- VAD hands-free (tự detect speech start/end)
- RAG với Qdrant + E5-Large + cross-encoder reranker
- Query expansion (từ điển chuẩn hoá dinh dưỡng)
- Conversation history (6 lượt gần nhất)

---

## Nhóm 1 — Latency Reduction

### 1.1 Filler Words
**Nguồn:** Thực tiễn production (Vapi, ElevenLabs, Retell AI)

Khi LLM chưa trả về token đầu tiên sau >200ms, phát âm thanh filler đã pre-synthesize sẵn: *"Ừ để tôi xem..."*, *"À..."*, *"Được..."*. Người dùng không cảm nhận silence. Đây là kỹ thuật được dùng trong mọi production voicebot hiện tại.

- **Impact:** TTFA perceived giảm 200–400ms
- **Effort:** ~4h — pre-synthesize 3–5 clip WAV lúc startup, trigger khi TTFT > 200ms

### 1.2 Parallel RAG Prefetch
**Nguồn:** VoiceAgentRAG (arxiv:2603.02206) — 75% cache hit rate, 316× speedup

Khi ASR trả về partial transcript (chưa finalize), fire Qdrant search ngay song song thay vì chờ ASR xong. Khi LLM cần context thì retrieval đã xong hoặc gần xong.

- **Impact:** -100–200ms retrieval latency
- **Effort:** ~8h — thêm `asyncio.create_task(rag.search(partial_query))` trong orchestrator khi nhận ASR partial event

### 1.3 Semantic Cache cho RAG
**Nguồn:** VoiceAgentRAG (arxiv:2603.02206)

Dual-agent: background "Slow Thinker" predict follow-up topics sau mỗi câu trả lời → prefetch Qdrant context vào FAISS cache. Foreground "Fast Talker" đọc từ cache (<1ms) thay vì query Qdrant (50–300ms). Trong conversation dinh dưỡng, nếu user hỏi về protein → prefetch calcium, sắt, micronutrient.

- **Impact:** 100–200ms mỗi lượt follow-up (trên cache hit)
- **Effort:** ~20h — cần cache layer + topic predictor

### 1.4 Incremental ASR → LLM Trigger
**Nguồn:** LTS-VoiceAgent (arxiv:2601.19952) — Listen-Think-Speak framework

Thay vì chờ ASR finalize, dùng "Dynamic Semantic Trigger": khi partial transcript đạt ngưỡng confidence nhất định và có đủ semantic content (ví dụ "Tôi muốn tăng cân..."), bắt đầu LLM generation ngay. Nếu ASR sau đó cập nhật → rerank hoặc restart.

- **Impact:** -100–200ms TTFT
- **Effort:** ~15h — cần confidence threshold tuning, handle ASR correction

### 1.5 End-of-Turn Prediction (Voice Activity Projection)
**Nguồn:** VAP paper (arxiv:2403.06487), TEN-VAD (HuggingFace: TEN-framework/ten-vad)

Thay VAD silence-based (reactive) bằng model nhỏ ~6M params predict khi nào user sắp nói xong — bắt đầu LLM generation 300–400ms trước khi user thực sự dừng. Khác với VAD ở chỗ VAP predict future activity thay vì chỉ detect hiện tại.

- **Impact:** -300–400ms LLM start delay
- **Effort:** ~15h — thay Silero VAD bằng TEN-VAD, thêm turn_confidence threshold

### 1.6 Speculative Decoding cho LLM
**Nguồn:** PredGen (arxiv:2506.15556) — ~2× LLM latency reduction; vLLM native support

Dùng draft model nhỏ để generate candidate tokens, LLM lớn verify song song. vLLM đã hỗ trợ native. Với Qwen3-4B + draft model ~0.5B: 2–3× throughput improvement, TTFT giảm 200–300ms.

- **Impact:** 2–3× LLM throughput, -200–300ms TTFT
- **Effort:** ~5h — config vLLM speculative decoding (chủ yếu là config, không cần code)

### 1.7 Token-Level TTS Streaming
**Nguồn:** X-Talk (arxiv:2512.18706)

Hiện tại chunker đợi ≥40 chars + dấu câu mới gửi TTS. Thay bằng word-boundary streaming: mỗi khi có đủ 1 word hoàn chỉnh → gửi TTS ngay. TTS buffer internally và synthesize khi đủ context.

- **Impact:** -50–150ms TTFA
- **Effort:** ~15h — refactor chunker + TTS handler

---

## Nhóm 2 — Conversation Quality

### 2.1 Backchanneling
**Nguồn:** Retell AI blog, RESPOND framework (arxiv:2603.21682), Real-Time Backchannel Prediction (arxiv:2410.15929)

Trong khi user đang nói, nếu có pause >800ms → phát backchannel nhẹ ("Ừ", "Vâng", "Tôi nghe") mà không lấy lượt. Khác với barge-in: bot không ngắt mà chỉ báo hiệu đang lắng nghe. ElevenLabs và Hume AI đều implement cái này.

- **Impact:** UX tự nhiên hơn đáng kể, user không cảm thấy nói vào khoảng trống
- **Effort:** ~6h — monitor ASR stream, trigger pre-synthesized backchannel clip khi pause > threshold

### 2.2 Adaptive VAD Threshold
**Nguồn:** LiveKit turn-detection blog (2024)

Học pattern pause của từng user trong session (rolling average 10 lượt) → điều chỉnh ngưỡng silence detection. User hay suy nghĩ lâu sẽ có threshold cao hơn, tránh cắt sớm. Caps: 300ms–1200ms.

- **Impact:** Giảm false positive barge-in, conversation flow tự nhiên hơn
- **Effort:** ~3h

### 2.3 Persistent User Memory
Lưu thông tin user giữa các session: tình trạng sức khoẻ, mục tiêu dinh dưỡng, allergen, câu hỏi hay hỏi lại. Khi user quay lại: "Lần trước bạn hỏi về protein, lần này có liên quan không?"

- **Impact:** Personalization, không hỏi lại thông tin cũ
- **Effort:** ~10h — Redis hoặc SQLite per user_id, inject vào system prompt

### 2.4 Query Rewriting cho Follow-up Questions
**Nguồn:** SynRewrite (arxiv:2509.22325) — +15–25% retrieval recall

User hay dùng đại từ rút gọn trong follow-up: "Thế còn cái đó thì sao?" Rewrite thành query đầy đủ trước khi gửi Qdrant: "Lượng canxi khuyến nghị hàng ngày là bao nhiêu?" dùng conversation history.

- **Impact:** +15–25% retrieval recall trên ambiguous follow-up
- **Effort:** ~8h — thêm rewrite step trong brain pipeline, dùng LLM nhỏ hoặc rule-based

### 2.5 Context Compression cho Hội Thoại Dài
**Nguồn:** ACON (arxiv:2510.00615) — 26–54% token reduction, 95%+ accuracy maintained

Sau N lượt, nén history cũ: giữ 3 lượt gần nhất nguyên vẹn, summarize lượt 4–10 thành 1–2 câu. Tránh context window bloat khi conversation kéo dài.

- **Impact:** Giảm token usage, tránh context overflow
- **Effort:** ~10h

### 2.6 Adaptive Response Length
Detect từ query complexity và conversation context xem user muốn trả lời ngắn hay chi tiết. Query ngắn + lần đầu hỏi → 30–60 từ. Query phức tạp + nhiều lượt → 100–200 từ.

- **Impact:** User satisfaction, không bị overload thông tin
- **Effort:** ~4h — simple classifier hoặc heuristic dựa trên token count

---

## Nhóm 3 — Retrieval & Knowledge

### 3.1 Hybrid Retrieval: Vector + Knowledge Graph
**Nguồn:** AMG-RAG (arxiv:2502.13010) — 74.1% F1 trên MEDQA, outperform RAG thuần

Xây dựng lightweight knowledge graph cho domain dinh dưỡng: Nutrients ↔ Foods ↔ Health Conditions. Kết hợp vector search (Qdrant) cho free-text với graph traversal cho structured reasoning: "Vitamin D → hấp thụ Calcium → sức khoẻ xương."

- **Impact:** Reasoning phức tạp hơn, giảm hallucination trên câu hỏi có cấu trúc
- **Effort:** ~40h — cần build KG thủ công hoặc auto-extract, implement dual retrieval

### 3.2 Adaptive Retrieval (Không RAG khi không cần)
Detect xem query có cần retrieval không. Câu chào hỏi, câu đơn giản về kiến thức phổ biến ("Nước có calo không?") → skip Qdrant, trả lời thẳng từ LLM. Chỉ RAG khi query cần facts cụ thể.

- **Impact:** -50–200ms cho queries đơn giản (bỏ RAG round-trip)
- **Effort:** ~8h — binary classifier hoặc LLM self-assess confidence

---

## Nhóm 4 — Speech Quality

### 4.1 Prosody Control cho TTS
**Nguồn:** VoXtream2 (arxiv:2603.13518) — 74ms first-packet latency, 4× faster than real-time

Điều chỉnh speaking rate động theo nội dung: chậm hơn khi đọc số liệu quan trọng ("2000 kilocalorie mỗi ngày"), nhanh hơn với thông tin phụ. Thêm micro-pause trước số liệu.

- **Impact:** Naturalness và retention thông tin tốt hơn
- **Effort:** ~20h — cần TTS hỗ trợ prosody tags hoặc speed control

### 4.2 Emotion/Prosody Detection từ User Audio
**Nguồn:** Hume AI EVI, prosody detection research (MDPI 2025)

Extract pitch, energy, speaking rate từ user audio → detect frustrated/confused/satisfied → điều chỉnh tone bot (empathetic khi user frustrated, chi tiết hơn khi confused).

- **Impact:** Empathetic responses, UX tự nhiên hơn
- **Effort:** ~25h — librosa feature extraction + small classifier

---

## Nhóm 5 — Kiến Trúc Dài Hạn

### 5.1 Full-Duplex với Echo Cancellation
**Nguồn:** Moshi (arxiv:2410.00037), Seed full-duplex (ByteDance)

Bot có thể backchannel trong khi user đang nói mà ASR không "nghe thấy" chính mình. Cần WebRTC AEC (Acoustic Echo Cancellation). Phức tạp nhất trong danh sách này.

- **Impact:** Conversation tự nhiên nhất, gần human-level
- **Effort:** ~30h — WebRTC AEC integration, concurrent stream handling

### 5.2 Khi Vietnamese Speech LLM xuất hiện
Trong 12–18 tháng tới nếu có Vietnamese end-to-end speech LLM (hoặc multilingual model đủ tốt cho tiếng Việt), có thể xem xét migrate từ cascade sang hybrid:
- Giữ ASR riêng (sherpa-onnx vẫn tốt cho WER)
- Replace Brain + TTS bằng speech LLM
- Latency target: 200–300ms thay vì 400–600ms hiện tại

---

## Tóm Tắt Priority

| # | Innovation | Effort | Latency gain | UX gain | Priority |
|---|-----------|--------|-------------|---------|---------|
| 1.1 | Filler words | 4h | ~200ms perceived | ⭐⭐⭐ | **Cao nhất** |
| 1.6 | Speculative decoding (vLLM config) | 5h | -200ms | ⭐⭐ | **Cao** |
| 1.2 | Parallel RAG prefetch | 8h | -100–200ms | ⭐⭐ | **Cao** |
| 2.1 | Backchanneling | 6h | — | ⭐⭐⭐ | **Cao** |
| 2.2 | Adaptive VAD threshold | 3h | — | ⭐⭐ | **Cao** |
| 2.4 | Query rewriting | 8h | — | ⭐⭐ | Trung bình |
| 1.4 | Incremental ASR trigger | 15h | -100–200ms | ⭐⭐ | Trung bình |
| 1.5 | End-of-turn prediction (VAP) | 15h | -300ms | ⭐⭐⭐ | Trung bình |
| 1.7 | Token-level TTS streaming | 15h | -50–150ms | ⭐⭐ | Trung bình |
| 2.3 | Persistent user memory | 10h | — | ⭐⭐⭐ | Trung bình |
| 1.3 | Semantic RAG cache | 20h | -100–200ms | ⭐⭐ | Trung bình |
| 3.2 | Adaptive retrieval | 8h | -50–200ms | ⭐ | Thấp |
| 3.1 | KG + RAG hybrid | 40h | — | ⭐⭐ | Thấp |
| 4.1 | Prosody control TTS | 20h | — | ⭐⭐ | Thấp |
| 5.1 | Full-duplex AEC | 30h | — | ⭐⭐⭐ | Nghiên cứu |

---

## Tài liệu tham khảo

| Paper | Arxiv ID |
|-------|---------|
| Moshi: speech-text foundation model for real-time dialogue | 2410.00037 |
| Mini-Omni2: Open-source GPT-4o with Speech and Duplex | 2410.11190 |
| VITA-Audio: Fast Interleaved Cross-Modal Token Generation | 2505.03739 |
| X-Talk: Underestimated Potential of Modular Speech-to-Speech Systems | 2512.18706 |
| LTS-VoiceAgent: Listen-Think-Speak via Semantic Triggering | 2601.19952 |
| VoiceAgentRAG: Dual-Agent RAG for Real-Time Voice Agents | 2603.02206 |
| PredGen: Input-Time Speculation for Voice Interaction | 2506.15556 |
| Voice Activity Projection (VAP) | 2403.06487 |
| ACON: Context Compression for LLM Agents | 2510.00615 |
| AMG-RAG: Agentic Medical Knowledge Graphs | 2502.13010 |
| VoXtream2: Full-stream TTS with Dynamic Speaking Rate | 2603.13518 |
| SynRewrite: Synthetic Query Rewrites for RAG | 2509.22325 |
| LLaSA: Scaling Llama-based Speech Synthesis | 2502.04128 |
| Seed-TTS: High-Quality Versatile Speech Generation | 2406.02430 |
| Real-Time Backchannel Prediction | 2410.15929 |
| Recent Advances in Speech Language Models: A Survey | 2410.03751 |
