# RAGAS Evaluation — Ghi chú tham khảo

## Tại sao 360 mẫu là đủ

Paper gốc RAGAS (Es et al., 2024, EACL) chỉ dùng **50 samples** (WikiEval).
360 mẫu của dự án này nhiều hơn 7x so với paper gốc.

Câu trích dẫn gợi ý dùng trong thesis:

> *"The evaluation set of 360 samples exceeds the 50-sample WikiEval dataset used in the original RAGAS paper (Es et al., 2024, EACL), and is consistent with related work using 100–150 samples for RAG evaluation (RAGVue, 2025; ARES, 2023)."*

---

## Các paper tham khảo

### Paper gốc RAGAS
- **Tên:** RAGAS: Automated Evaluation of Retrieval Augmented Generation
- **Tác giả:** Shahul Es, Jithin James, Luis Espinosa Anke, Steven Schockaert
- **Venue:** EACL 2024 (demo track)
- **Arxiv:** https://arxiv.org/abs/2309.15217
- **Dataset:** WikiEval — **50 samples**
- **Metrics:** Faithfulness, Answer Relevance, Context Relevance

### RAGVue (2025)
- **Tên:** RAGVue: A Diagnostic View for Explainable and Automated Evaluation of RAG
- **Arxiv:** https://arxiv.org/html/2601.04196
- **Dataset:** StrategyQA — **100 samples**
- **Ghi chú:** So sánh trực tiếp với RAGAS, chỉ ra RAGAS tốn ~18s/query

### ARES (2023)
- **Tên:** ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems
- **Arxiv:** https://arxiv.org/html/2311.09476v2
- **Dataset:** ~**150 samples** mỗi RAG system
- **Ghi chú:** Kendall's τ cao hơn RAGAS trong context relevance và answer relevance

### Legal RAG (2025) — dùng RAGAS nhiều nhất
- **Tên:** All for law and law for all: Adaptive RAG Pipeline for Legal Research
- **Arxiv:** https://arxiv.org/html/2508.13107
- **Dataset:** LegalBenchRAG-mini — **776 samples** (194 × 4 domains)
- **Metrics dùng:** Faithfulness, Answer Relevancy + BERTScore-F1

### Nutrition/Food RAG — gần domain nhất
- **Tên:** Evaluation of LLMs in retrieving food and nutritional context for RAG systems
- **Arxiv:** https://arxiv.org/html/2603.09704v2
- **Dataset:** **150 samples** (50 easy / 50 medium / 50 hard)
- **Ghi chú:** Không dùng RAGAS, dùng F1/Precision/Recall thay thế
- **Domain:** Nutrition — gần nhất với dự án dinh dưỡng này

---

## Chiến lược chạy RAGAS tiết kiệm chi phí

RAGAS tốn tiền vì mỗi sample cần 3–5 LLM call để đánh giá.

**Đề xuất: 2 tầng**

| Tầng | Số mẫu | Metrics | Chi phí |
|---|---|---|---|
| Tầng 1 | 360 | Latency, word count, success rate | Rất rẻ (không gọi LLM) |
| Tầng 2 | 100 (stratified) | Faithfulness, Answer Relevancy, Context Recall | ~$1-3 với Gemini Flash |

**Stratified sample 100 câu:**
```
suckhoedoisong:  30 câu
viendinhduong:   30 câu
vinmec:          25 câu
thucuc:          15 câu
```

**LLM evaluator rẻ nhất:**
```python
from langchain_google_genai import ChatGoogleGenerativeAI
evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
```

---

## RAGAS metrics — ý nghĩa từng cái

| Metric | Đo gì | Cần gì |
|---|---|---|
| **Faithfulness** | Câu trả lời có dựa trên context không, hay hallucinate? | answer + contexts |
| **Answer Relevancy** | Câu trả lời có trả lời đúng câu hỏi không? | answer + question |
| **Context Recall** | RAG có lấy đúng tài liệu không? | contexts + reference_answer |
| **Context Precision** | Tài liệu lấy về có liên quan không (không thừa)? | contexts + question |

Với callbot dinh dưỡng, **Faithfulness** quan trọng nhất — tránh LLM bịa thông tin y tế.

---

## File dữ liệu liên quan

| File | Mô tả |
|---|---|
| `evaluation/synthetic_qa.jsonl` | 360 Q&A, gen bởi Gemini từ full_docs.jsonl |
| `evaluation/eval_brain_responses.py` | Script gen response + đo latency (360 câu) |
| `evaluation/results/` | Output của eval (jsonl + metrics json + chart png) |
| `evaluation/eval_pipeline_ragas.py` | Script chạy RAGAS đã có sẵn |

`synthetic_qa.jsonl` có sẵn các field cần cho RAGAS:
- `question` → `user_input`
- `reference_answer` → `reference`
- `contexts` → `retrieved_contexts` (từ corpus gốc, không phải Qdrant)
- `response` → cần gen thêm bằng `eval_brain_responses.py`
