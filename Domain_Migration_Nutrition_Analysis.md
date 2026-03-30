# Phân tích chuyển domain: Legal → Dinh dưỡng

## 1) Kết luận nhanh

- **Có thể chuyển domain mà giữ nguyên kiến trúc microservice hiện tại** (`ASR → Gateway → Brain → TTS`).
- **80% thay đổi nằm ở Brain + dữ liệu RAG** (prompt, query expansion, schema payload, bộ dữ liệu ingest).
- **ASR/TTS gần như giữ nguyên**; Gateway chỉ cần đổi text mô tả/branding.

---

## 2) Phần có thể tái sử dụng ngay

### Hạ tầng / runtime
- Docker Compose, gRPC wiring, WebSocket pipeline.
- Luồng streaming chunk (chunker + latency timing).
- Cơ chế gọi LLM streaming hiện tại.

### Service không phụ thuộc domain mạnh
- `legal-callbot/asr/**`
- `legal-callbot/tts/**`
- phần lớn `legal-callbot/gateway/**` (trừ text mô tả)

---

## 3) Điểm khóa cứng legal cần đổi

### Prompt và logic trả lời
- `legal-callbot/brain/core/prompt.py`
  - `LEGAL_SYSTEM_PROMPT` đang ép format pháp lý (Điều/Khoản/Disclaimer pháp lý).
  - `FEW_SHOT_EXAMPLES` đều là câu hỏi luật.
  - `build_prompt(... legal_context=...)` dùng naming legal.

- `legal-callbot/brain/grpc_handler.py`
  - Import `LEGAL_SYSTEM_PROMPT`.
  - Biến `legal_docs`, `legal_context`.

### Retrieval payload / schema
- `legal-callbot/brain/core/rag.py`
  - Trả về field `dieu`, fallback `"Không rõ Điều luật"`.
  - Phụ thuộc payload legal (`ten_dieu`, `text`).

- `legal-callbot/brain/config.py`
  - `qdrant_collection = "phap_dien_khoan"` đang hard-code legal collection.

### Query expansion
- `legal-callbot/brain/core/query_expander.py`
  - Dictionary alias legal (`sổ đỏ`, `NĐ100`, `Điều...`).

### Pipeline dữ liệu ingest/preprocess
- Root scripts + `brain/scripts/**` hiện bám schema pháp luật:
  - `prepare_qdrant_schema.py`
  - `crawl_laws_full.py`, `parse_extracted_laws.py`
  - `brain/scripts/ingest.py`, `refine_data.py`, `preprocess_data.py`
- Các field hiện tại: `vbqppl`, `ten_dieu`, `chu_de`, `so_hieu`, ...

### Test data
- `legal-callbot/brain/data/train_qa_dataset.jsonl`
- `legal-callbot/brain/data/test_qa_dataset.jsonl`
- `brain/stress_test.py`, `test_rag_pipeline.py` chứa query pháp lý mẫu.

---

## 4) Đề xuất schema dữ liệu mới cho domain dinh dưỡng

### Payload Qdrant gợi ý
```json
{
  "doc_id": "nutrition_000123",
  "title": "Vai trò chất xơ trong kiểm soát đường huyết",
  "topic": "đái tháo đường",
  "subtopic": "carbohydrate",
  "audience": "người trưởng thành",
  "evidence_level": "systematic_review",
  "source": "WHO|Bộ Y tế|NCCN|journal",
  "published_at": "2024-05-01",
  "language": "vi",
  "text": "...",
  "tags": ["glycemic-index", "fiber", "meal-planning"],
  "contraindications": ["suy thận giai đoạn cuối"],
  "metadata": {
    "url": "...",
    "license": "..."
  }
}
```

### Chuẩn hóa tối thiểu bắt buộc
- `title`, `text`, `source`, `published_at`, `evidence_level`.
- Có cờ an toàn: `medical_disclaimer_required: true`.

---

## 5) Thiết kế prompt mới cho dinh dưỡng

### Nguyên tắc
- Trả lời theo **mức độ bằng chứng** (ưu tiên guideline/systematic review).
- Không chẩn đoán bệnh; không kê toa thuốc.
- Có disclaimer y tế: "Thông tin tham khảo, không thay thế tư vấn bác sĩ/chuyên gia dinh dưỡng".

### Cấu trúc output khuyến nghị
1. Kết luận ngắn gọn (1-2 câu).
2. Lý do chính (2-3 ý).
3. Khuyến nghị thực hành an toàn.
4. Khi nào cần gặp bác sĩ.

---

## 6) Kế hoạch migration theo 3 pha

## Pha A — "Switchable domain" (1-2 ngày)
Mục tiêu: code chạy được với profile domain.

- Tạo `domain profile` trong Brain (`legal`, `nutrition`).
- Tách prompt/few-shot theo profile.
- Param hóa collection (`QDRANT_COLLECTION`).
- Đổi naming trung tính trong code (`domain_docs`, `domain_context`).

## Pha B — Data pipeline dinh dưỡng (2-4 ngày)
Mục tiêu: có dữ liệu vector dùng được.

- Chuẩn bị dataset dinh dưỡng sạch (FAQ + guideline + tài liệu chuẩn).
- Viết script convert schema mới → payload Qdrant.
- Ingest vào collection mới (vd: `nutrition_kb_v1`).
- Tạo filter metadata (`topic`, `audience`, `evidence_level`).

## Pha C — Safety + eval (1-2 ngày)
Mục tiêu: giảm hallucination và tăng an toàn.

- Bộ test câu hỏi: giảm cân, tiểu đường, gout, thai kỳ, trẻ em.
- Rule từ chối câu hỏi vượt phạm vi (kê thuốc, chẩn đoán chắc chắn).
- So sánh retrieval quality: top-k relevance, groundedness.

---

## 7) Rủi ro chính khi chuyển sang dinh dưỡng

1. **Rủi ro an toàn nội dung y tế**
   - Cần guardrail mạnh hơn domain pháp lý.
2. **Chất lượng nguồn dữ liệu**
   - Nếu dùng nguồn blog/SEO dễ nhiễu và sai.
3. **Thiếu metadata bằng chứng**
   - Không có `evidence_level` sẽ khó ưu tiên nguồn tốt.
4. **Prompt cũ gây giọng điệu legal**
   - Cần thay hoàn toàn cấu trúc trích dẫn kiểu Điều/Khoản.

---

## 8) Lộ trình thực thi ngắn nhất (MVP)

- Ngày 1: Param hóa domain + prompt nutrition + collection env.
- Ngày 2: Ingest 5k–20k chunk dinh dưỡng chuẩn.
- Ngày 3: Chạy eval 100 câu + tinh chỉnh query expansion + guardrail.

Nếu cần, có thể giữ song song 2 domain bằng cách:
- `DOMAIN=legal`, `QDRANT_COLLECTION=phap_dien_khoan`
- `DOMAIN=nutrition`, `QDRANT_COLLECTION=nutrition_kb_v1`

---

## 9) Ưu tiên kỹ thuật đề xuất (theo impact)

1. Param hóa `domain profile` trong `brain/core/prompt.py` + `grpc_handler.py`.
2. Bỏ hard-code collection ở `brain/config.py`.
3. Viết mới script ingest schema dinh dưỡng.
4. Tạo bộ test + safety checklist cho tư vấn dinh dưỡng.

---

## 10) Checklist trước khi go-live dinh dưỡng

- [ ] Tất cả response có disclaimer y tế.
- [ ] Không đưa liều thuốc cụ thể trừ khi có nguồn chuẩn và cảnh báo.
- [ ] Có fallback "không đủ dữ liệu" khi retrieval yếu.
- [ ] Câu trả lời có grounding từ context RAG.
- [ ] Có test các nhóm nguy cơ cao: thai kỳ, bệnh nền, trẻ em, người cao tuổi.
