"""
gen_corpus_from_llm.py
Dùng LLM (Cerebras qwen-3-235b hoặc Groq) để sinh corpus cho các câu hỏi chưa crawl được.

Chạy: python evaluation/gen_corpus_from_llm.py
Output: new_corpus_from_llm.jsonl (append vào corpus chính sau khi kiểm tra)
"""

import os, json, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ==============================================================
# ⚙️ CẤU HÌNH
# ==============================================================

# Cerebras (ưu tiên): https://cloud.cerebras.ai → API Keys
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_MODEL   = "qwen-3-235b-a22b-instruct-2507"

# Fallback: Groq
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = "llama-3.3-70b-versatile"

INPUT_JSON  = "corpus_summary/questions_no_docs_final.json"
OUTPUT_JSONL = "new_corpus_from_llm.jsonl"
CHECKPOINT   = "corpus_summary/llm_corpus_checkpoint.json"

PAUSE_SEC   = 2.0   # giữa các request
# ==============================================================

SYSTEM_PROMPT = """Bạn là chuyên gia dinh dưỡng và y tế người Việt Nam.
Nhiệm vụ: viết một đoạn văn bản y tế/dinh dưỡng CHÍNH XÁC, CHI TIẾT để làm tài liệu tham khảo (corpus) cho hệ thống RAG.

Yêu cầu:
- Viết bằng tiếng Việt, văn phong chuyên môn nhưng dễ hiểu
- Độ dài: 300-600 từ
- Nội dung phải bao phủ đầy đủ các khía cạnh của câu hỏi
- Trình bày rõ ràng: cơ chế, nguyên nhân, lưu ý thực hành
- KHÔNG bịa đặt thông tin — chỉ viết những gì chính xác về mặt y học
- KHÔNG mở đầu bằng "Tôi sẽ..." hay "Đây là..."
- Bắt đầu thẳng vào nội dung"""

PROMPT_TEMPLATE = """Viết tài liệu tham khảo y tế cho câu hỏi sau:

Câu hỏi: {question}

Gợi ý nội dung cần bao phủ (dựa trên đáp án chuẩn):
{reference_hint}

Viết đoạn corpus:"""


def build_reference_hint(reference: str) -> str:
    """Trích các ý chính từ reference để gợi ý cho LLM."""
    # Lấy 300 ký tự đầu của reference làm gợi ý
    hint = reference[:400].strip()
    if len(reference) > 400:
        hint += "..."
    return hint


def call_cerebras(question: str, reference: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=CEREBRAS_API_KEY,
        base_url="https://api.cerebras.ai/v1"
    )
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        reference_hint=build_reference_hint(reference)
    )
    resp = client.chat.completions.create(
        model=CEREBRAS_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def call_groq(question: str, reference: str) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        reference_hint=build_reference_hint(reference)
    )
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def generate_corpus(question: str, reference: str) -> str:
    if CEREBRAS_API_KEY:
        return call_cerebras(question, reference)
    elif GROQ_API_KEY:
        return call_groq(question, reference)
    else:
        raise ValueError("Cần CEREBRAS_API_KEY hoặc GROQ_API_KEY trong .env")


# ==============================================================
# MAIN
# ==============================================================

# Load questions
with open(INPUT_JSON, encoding="utf-8") as f:
    questions = json.load(f)

# Load checkpoint
done_ids = set()
if Path(CHECKPOINT).exists():
    with open(CHECKPOINT, encoding="utf-8") as f:
        done_ids = set(json.load(f))
    print(f"♻️  Tiếp tục từ checkpoint — đã xong: {len(done_ids)} câu")

to_process = [q for q in questions if q["id"] not in done_ids]
print(f"🎯 Cần xử lý: {len(to_process)} câu\n")

provider = "Cerebras" if CEREBRAS_API_KEY else "Groq"
print(f"🤖 Provider: {provider}\n{'='*60}")

for i, item in enumerate(to_process):
    qid      = item["id"]
    question = item["question"]
    reference = item.get("reference", "")

    print(f"\n[{i+1}/{len(to_process)}] {qid}")
    print(f"  Q: {question[:80].replace(chr(10), ' ')}...")

    try:
        corpus_text = generate_corpus(question, reference)
        print(f"  ✅ {len(corpus_text)} chars")

        # Append to output
        with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "text":            corpus_text,
                "source":          f"llm_generated/{provider.lower()}",
                "relevance_score": 1.0,
                "from_question":   question,
                "question_id":     qid,
                "original_recall": item.get("context_recall", 0),
                "generated":       True,
            }, ensure_ascii=False) + "\n")

        done_ids.add(qid)
        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump(list(done_ids), f)

    except Exception as e:
        print(f"  ⚠️  Lỗi: {e}")

    time.sleep(PAUSE_SEC)

print(f"\n{'='*60}")
print(f"✅ Xong! Output: {OUTPUT_JSONL}")
print(f"   Tổng đã sinh: {len(done_ids)} / {len(questions)} câu")
print(f"\n⚠️  Kiểm tra nội dung trong {OUTPUT_JSONL} trước khi merge vào corpus chính!")
