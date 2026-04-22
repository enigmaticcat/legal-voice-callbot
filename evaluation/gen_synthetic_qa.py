"""
gen_synthetic_qa.py
Sinh câu hỏi + câu trả lời tham chiếu từ full_docs.jsonl dùng Gemini API.

Input : evaluation/full_docs.jsonl  (6,001 bài đã lọc)
Output: evaluation/synthetic_qa.jsonl

Mỗi dòng output:
{
  "id":               str,       # syn_{source}_{n:04d}
  "type":             "A",       # A = single-doc
  "source":           str,
  "doc_id":           str,       # url gốc
  "title":            str,
  "contexts":         [str],     # [article text] — RAGAS-compatible format
  "question":         str,
  "reference_answer": str,
}

Cách dùng:
  python evaluation/gen_synthetic_qa.py                   # sinh 400 câu
  python evaluation/gen_synthetic_qa.py --sources viendinhduong --n 50
  python evaluation/gen_synthetic_qa.py --resume          # tiếp tục từ checkpoint
  python evaluation/gen_synthetic_qa.py --dry-run         # preview docs, không gọi API
  python evaluation/gen_synthetic_qa.py --clean           # xóa mục xấu khỏi output hiện có
"""

import argparse
import json
import os
import re
import random
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

EVAL_DIR  = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent

load_dotenv(REPO_ROOT / ".env")
load_dotenv(REPO_ROOT / "nutrition-callbot" / ".env")

INPUT_PATH      = EVAL_DIR / "full_docs.jsonl"
OUTPUT_PATH     = EVAL_DIR / "synthetic_qa.jsonl"
SKIPPED_PATH    = EVAL_DIR / "synthetic_qa_skipped.jsonl"
CHECKPOINT_PATH = EVAL_DIR / "synthetic_qa_checkpoint.json"

DEFAULT_N_PER_SOURCE = {
    "viendinhduong":  120,
    "benhvienthucuc": 120,
    "vinmec":         80,
    "suckhoedoisong": 80,
}

VERTEX_MODEL  = "gemini-2.5-flash"
PAUSE_SEC     = 1.0   # Vertex AI: ~1000 RPM
MAX_RETRIES   = 4
CONTEXT_CHARS = 5000

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
Bạn là chuyên gia dinh dưỡng và y tế. Nhiệm vụ: đọc bài viết dưới đây và tạo ra \
MỘT câu hỏi cùng câu trả lời tham chiếu cho bộ dữ liệu đánh giá hệ thống RAG.

=== YÊU CẦU CÂU HỎI ===
- Viết bằng tiếng Việt, 1–2 câu, tự nhiên như người dùng thật hỏi chatbot tư vấn dinh dưỡng
- Câu hỏi phải trả lời được RÕ RÀNG chỉ từ nội dung bài — không hỏi thêm thông tin ngoài
- Ưu tiên: triệu chứng/nguyên nhân bệnh, liều lượng dinh dưỡng, thực phẩm nên/không nên ăn, \
cách phòng ngừa hoặc điều trị
- KHÔNG hỏi về: tên người cụ thể (cá nhân, bác sĩ), tên thương hiệu/sản phẩm thương mại, \
sự kiện lịch sử, nhân vật tiểu sử
- KHÔNG dùng cụm "trong bài viết" hay "theo tác giả" trong câu hỏi

=== YÊU CẦU CÂU TRẢ LỜI ===
- Dựa HOÀN TOÀN vào nội dung bài — không thêm thông tin ngoài
- Độ dài: 90–250 từ, đủ chi tiết để đánh giá faithfulness
- Trình bày mạch lạc, không bullet list cụt ngủn
- Nếu bài chỉ đề cập một nhóm đối tượng/loại thực phẩm cụ thể, câu trả lời PHẢI nêu rõ \
phạm vi đó — KHÔNG tổng quát hóa sang nhóm khác \
(ví dụ: bài nói về "kẹo hương trái cây" thì không được viết "tất cả các loại kẹo")

=== TRƯỜNG HỢP ĐẶC BIỆT ===
Nếu bài viết KHÔNG liên quan đến dinh dưỡng hoặc y tế (ví dụ: quảng cáo sản phẩm, \
tiểu sử nhân vật, tin tức không liên quan), trả về:
{"skip": true, "reason": "lý do ngắn"}

=== OUTPUT FORMAT ===
Trả về JSON object DUY NHẤT (không markdown, không giải thích):
{"question": "...", "reference_answer": "..."}\
"""

USER_PROMPT_TEMPLATE = """\
=== BÀI VIẾT ===
Tiêu đề: {title}
Nguồn: {source}

{text}
=== KẾT THÚC BÀI VIẾT ===

Tạo câu hỏi và câu trả lời tham chiếu từ bài viết trên.\
"""


# ── Validation ────────────────────────────────────────────────────────────────

_BAD_QUESTION_PATTERNS = [
    r"trong bài viết",
    r"theo tác giả",
    r"bài viết (này|đó|trên)",
    r"\b(giáo sư|gs\.|ts\.|bác sĩ)\s+[A-ZÀÁẢÃẠĂẮẶẰẲẴÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ][a-zàáảãạăắặằẳẵâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+",
]

_BAD_PATTERNS_RE = [re.compile(p, re.IGNORECASE) for p in _BAD_QUESTION_PATTERNS]

MIN_ANSWER_WORDS = 90


def validate_qa(qa: dict) -> tuple[bool, str]:
    if qa.get("skip"):
        return False, f"skip: {qa.get('reason', '')}"

    q = qa.get("question", "")
    a = qa.get("reference_answer", "")

    if not q or not a:
        return False, "missing question or answer"

    for pat in _BAD_PATTERNS_RE:
        if pat.search(q):
            return False, f"bad question pattern: {pat.pattern}"

    a_words = len(a.split())
    if a_words < MIN_ANSWER_WORDS:
        return False, f"answer too short ({a_words}w < {MIN_ANSWER_WORDS})"

    return True, ""


# ── Vertex AI call ────────────────────────────────────────────────────────────

def call_vertex(doc: dict) -> dict:
    from google import genai
    from google.genai import types

    project  = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT chua set trong .env")

    client = genai.Client(vertexai=True, project=project, location=location)

    prompt = USER_PROMPT_TEMPLATE.format(
        title=doc["title"],
        source=doc["source"],
        text=doc["text"][:CONTEXT_CHARS],
    )

    resp = client.models.generate_content(
        model=VERTEX_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=1024,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    raw = resp.text.strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    return json.loads(raw)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_docs(sources: list[str] | None = None) -> dict[str, list[dict]]:
    by_source: dict[str, list[dict]] = {}
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            src = doc["source"]
            if sources and src not in sources:
                continue
            by_source.setdefault(src, []).append(doc)
    return by_source


def sample_docs(by_source: dict, n_per_source: dict) -> list[dict]:
    sampled = []
    for src, docs in by_source.items():
        n = n_per_source.get(src, 0)
        if n <= 0:
            continue
        chosen = random.sample(docs, min(n, len(docs)))
        sampled.extend(chosen)
    random.shuffle(sampled)
    return sampled


def load_checkpoint() -> set:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(done_ids: set):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(done_ids), f)


def make_id(source: str, n: int) -> str:
    prefix = {
        "viendinhduong":  "vdd",
        "benhvienthucuc": "thucuc",
        "vinmec":         "vnm",
        "suckhoedoisong": "skds",
    }.get(source, source[:4])
    return f"syn_{prefix}_{n:04d}"


# ── --clean ───────────────────────────────────────────────────────────────────

def clean_output(out_path: Path):
    if not out_path.exists():
        print("Output file không tồn tại.")
        return

    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept, removed = [], []

    for r in rows:
        if "gold_context" in r and "contexts" not in r:
            r["contexts"] = [r.pop("gold_context")]

        qa_stub = {"question": r.get("question", ""), "reference_answer": r.get("reference_answer", "")}
        ok, reason = validate_qa(qa_stub)
        if ok:
            kept.append(r)
        else:
            removed.append((r["id"], reason))

    print(f"Giữ lại: {len(kept)}  |  Xóa: {len(removed)}")
    for rid, reason in removed:
        print(f"  x {rid}: {reason}")

    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + ("\n" if kept else ""),
        encoding="utf-8",
    )
    print(f"Đã ghi lại {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--n", type=int, default=None,
                        help="Tổng số câu (phân bổ theo tỉ lệ)")
    parser.add_argument("--out", default=str(OUTPUT_PATH))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean", action="store_true",
                        help="Xóa mục xấu + migrate field cũ, không sinh mới")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.clean:
        clean_output(out_path)
        return

    random.seed(args.seed)

    if args.n:
        total_default = sum(DEFAULT_N_PER_SOURCE.values())
        n_per_source = {
            src: max(1, round(args.n * cnt / total_default))
            for src, cnt in DEFAULT_N_PER_SOURCE.items()
        }
    else:
        n_per_source = DEFAULT_N_PER_SOURCE.copy()

    if args.sources:
        n_per_source = {k: v for k, v in n_per_source.items() if k in args.sources}

    print("Số mẫu mỗi nguồn:")
    for src, n in n_per_source.items():
        print(f"  {src}: {n}")
    print(f"  Tổng: {sum(n_per_source.values())}")

    print(f"\nLoading {INPUT_PATH} ...")
    by_source = load_docs(args.sources)
    docs = sample_docs(by_source, n_per_source)
    print(f"Đã sample: {len(docs)} docs")

    if args.dry_run:
        for d in docs[:5]:
            print(f"\n--- {d['source']} | {d['doc_id'][:70]} ---")
            print(f"Title : {d['title']}")
            print(f"Words : {d['word_count']}")
            print(f"Chars : {len(d['text'])} (sẽ gửi {min(len(d['text']), CONTEXT_CHARS)})")
            print(d["text"][:200] + "...")
        return

    done_doc_ids: set = set()
    counter_by_source: dict[str, int] = {}

    if args.resume:
        done_doc_ids = load_checkpoint()
        if out_path.exists():
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    src = row["source"]
                    counter_by_source[src] = counter_by_source.get(src, 0) + 1
        print(f"Resume: đã xong {len(done_doc_ids)} docs")

    pending = [d for d in docs if d["doc_id"] not in done_doc_ids]
    print(f"Cần xử lý: {len(pending)} docs\n{'='*60}")

    written = skipped = errors = 0

    with open(out_path, "a", encoding="utf-8") as fout, \
         open(SKIPPED_PATH, "a", encoding="utf-8") as fskip:
        for i, doc in enumerate(pending):
            src   = doc["source"]
            title = doc["title"][:50]
            print(f"[{i+1}/{len(pending)}] {title} ...", end="", flush=True)

            qa = None
            last_err = ""
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    qa = call_vertex(doc)
                    break
                except json.JSONDecodeError as e:
                    last_err = f"JSON parse error: {e}"
                except Exception as e:
                    last_err = str(e)
                    err_lower = str(e).lower()
                    if any(k in err_lower for k in ("quota", "rate_limit", "resource_exhausted", "429")):
                        wait = PAUSE_SEC * (3 ** attempt)  # 13s -> 40s -> 121s
                        print(f"\n  RATE LIMIT (attempt {attempt}) - cho {wait:.0f}s ...", end="", flush=True)
                        time.sleep(wait)
                    elif attempt < MAX_RETRIES:
                        time.sleep(PAUSE_SEC)

            if qa is None:
                errors += 1
                done_doc_ids.add(doc["doc_id"])
                save_checkpoint(done_doc_ids)
                print(f"  ERROR: {last_err}")
                time.sleep(PAUSE_SEC)
                continue

            ok, reason = validate_qa(qa)
            if not ok:
                skipped += 1
                done_doc_ids.add(doc["doc_id"])
                save_checkpoint(done_doc_ids)
                skip_row = {
                    "doc_id":   doc["doc_id"],
                    "source":   src,
                    "title":    doc["title"],
                    "reason":   reason,
                    "question": qa.get("question", ""),
                    "reference_answer": qa.get("reference_answer", ""),
                }
                fskip.write(json.dumps(skip_row, ensure_ascii=False) + "\n")
                fskip.flush()
                print(f"  SKIP: {reason}")
                time.sleep(PAUSE_SEC)
                continue

            n = counter_by_source.get(src, 0) + 1
            counter_by_source[src] = n
            item_id = make_id(src, n)

            row = {
                "id":               item_id,
                "type":             "A",
                "source":           src,
                "doc_id":           doc["doc_id"],
                "title":            doc["title"],
                "contexts":         [doc["text"]],
                "question":         qa["question"],
                "reference_answer": qa["reference_answer"],
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            done_doc_ids.add(doc["doc_id"])
            save_checkpoint(done_doc_ids)

            written += 1
            print(f"  OK  [{item_id}]  q={len(qa['question'])}c  a={len(qa['reference_answer'].split())}w")
            time.sleep(PAUSE_SEC)

    print(f"\n{'='*60}")
    print(f"Ghi mới: {written}  |  Skip (xấu): {skipped}  |  Lỗi API: {errors}")
    print(f"Output : {out_path}")

    if out_path.exists():
        rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        src_cnt = Counter(r["source"] for r in rows)
        print(f"\nTổng trong file ({len(rows)} mục):")
        for src, cnt in src_cnt.most_common():
            print(f"  {src}: {cnt}")


if __name__ == "__main__":
    main()
