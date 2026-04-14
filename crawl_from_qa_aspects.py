# -*- coding: utf-8 -*-
"""
crawl_from_qa_aspects.py
========================
Pipeline:
  1. Load viendinhduong_qa.jsonl (Q&A pairs)
  2. Dùng Cerebras Qwen 235B để tách reference answer thành các aspects
  3. Với mỗi aspect → DuckDuckGo search (loại trừ domain nguồn)
  4. Crawl bài tìm được → score bằng cross-encoder
  5. Lưu vào corpus format (chunk_id, doc_id, source, url, ...)

Cài dependencies:
  pip install groq ddgs trafilatura sentence-transformers

Chạy:
  python crawl_from_qa_aspects.py --dry-run     # test 3 câu đầu
  python crawl_from_qa_aspects.py               # chạy toàn bộ
"""

import argparse
import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import trafilatura
from ddgs import DDGS
from groq import Groq
from sentence_transformers import CrossEncoder

# ──────────────────────────────────────────────
# CONFIG — điền API key vào đây hoặc .env
# ──────────────────────────────────────────────

import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL         = "qwen/qwen3-32b"

RERANK_MODEL       = "BAAI/bge-reranker-v2-m3"
RERANK_THRESHOLD   = 0.0                     # logit > 0 = relevant

INPUT_FILE         = "evaluation/viendinhduong_qa.jsonl"
OUTPUT_CORPUS      = "data_final/corpus_from_aspects.jsonl"
DONE_IDS_FILE      = "data_final/corpus_from_aspects_done.txt"
FAILED_LOG         = "data_final/corpus_from_aspects_failed.txt"

MAX_SEARCH_RESULTS = 8    # kết quả DDG mỗi query
MAX_URLS_PER_QA    = 20   # tổng URL thử mỗi Q&A
TOP_ARTICLES       = 3    # lưu tối đa bài/Q&A
ASPECTS_PER_QA     = 5    # số aspect tối đa mỗi câu

SEARCH_PAUSE       = 2.0  # giây chờ giữa các search (DDG can bi rate limited)
CRAWL_PAUSE        = 0.5

# ──────────────────────────────────────────────
# KHỞI TẠO
# ──────────────────────────────────────────────

groq_client = Groq(api_key=GROQ_API_KEY)

print(f"Load reranker: {RERANK_MODEL}...")
reranker = CrossEncoder(RERANK_MODEL)
print("Reranker ready.\n")

# ──────────────────────────────────────────────
# PROMPT TÁCH ASPECTS
# ──────────────────────────────────────────────

ASPECT_PROMPT = """\
Bạn là chuyên gia dinh dưỡng. Từ câu hỏi và câu trả lời dưới đây, hãy xác định \
các KHÍA CẠNH KIẾN THỨC CỤ THỂ cần tìm kiếm để có thể trả lời câu hỏi này.

Yêu cầu mỗi khía cạnh:
- Là một sự kiện/khái niệm độc lập, có thể tìm kiếm được trên internet
- Không trùng lặp nhau
- Tối đa {max_aspects} khía cạnh

Câu hỏi: {question}

Câu trả lời tham khảo: {answer}

Trả về JSON (không giải thích thêm):
{{
  "aspects": [
    {{
      "aspect": "mô tả ngắn khía cạnh",
      "query_vi": "từ khóa tìm kiếm tiếng Việt",
      "query_en": "english search query"
    }}
  ]
}}
"""


def extract_aspects(question: str, answer: str) -> list[dict]:
    prompt = ASPECT_PROMPT.format(
        question=question,
        answer=answer[:2000],    # giới hạn để tiết kiệm token
        max_aspects=ASPECTS_PER_QA,
    )
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip thinking tags (Qwen3 thinking mode)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Strip markdown code blocks
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

        # Parse JSON — tìm block JSON trong response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return []
        data = json.loads(match.group())
        return data.get("aspects", [])
    except Exception as e:
        print(f"  [aspect extraction error] {e}")
        return []


# ──────────────────────────────────────────────
# DUCKDUCKGO SEARCH
# ──────────────────────────────────────────────

def ddg_search(query: str, exclude_domain: str, num: int = 8) -> list[str]:
    """
    Tim kiem DuckDuckGo, loai tru domain nguon.
    Tra ve danh sach URL.
    """
    full_query = f"{query} -site:{exclude_domain}"
    try:
        with DDGS() as ddgs:
            results = ddgs.text(full_query, max_results=num)
            return [r.get("href", "") for r in results if r.get("href")]
    except Exception as e:
        print(f"  [ddg search error] {e}")
        return []


# ──────────────────────────────────────────────
# CRAWL + CHUNK + SCORE
# ──────────────────────────────────────────────

def crawl_url(url: str) -> str | None:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        return text if text and len(text) > 200 else None
    except Exception:
        return None


def chunk_text(text: str, window: int = 200, stride: int = 150) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), stride):
        chunk = " ".join(words[i: i + window])
        if len(chunk) >= 80:
            chunks.append(chunk)
    return chunks


def score_chunks(chunks: list[str], question: str) -> list[dict]:
    if not chunks:
        return []
    pairs = [[question, c] for c in chunks]
    scores = reranker.predict(pairs, batch_size=32)
    return [
        {"text": c, "score": round(float(s), 4)}
        for c, s in zip(chunks, scores)
        if s >= RERANK_THRESHOLD
    ]


# ──────────────────────────────────────────────
# CORPUS FORMAT
# ──────────────────────────────────────────────

def make_chunk_id(seed: str) -> str:
    h = hashlib.md5(seed.encode()).hexdigest()
    return str(uuid.UUID(h))


def make_corpus_rows(url: str, text: str, source_domain: str, question: str) -> list[dict]:
    """Chuyển text crawl được thành các dòng corpus_final format."""
    chunks = chunk_text(text)
    rows = []
    for i, chunk in enumerate(chunks):
        chunk_id = make_chunk_id(f"{url}_{i}")
        embed_text = chunk  # có thể thêm title prefix sau
        rows.append({
            "chunk_id":    chunk_id,
            "doc_id":      url,
            "source":      source_domain,
            "url":         url,
            "title":       "",           # trafilatura không lấy được title dễ, để trống
            "category":    "dinh-duong",
            "chunk_index": i,
            "text":        chunk,
            "embed_text":  embed_text,
            "from_question": question,   # metadata để trace sau
        })
    return rows


# ──────────────────────────────────────────────
# PIPELINE CHÍNH
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Chỉ chạy 3 câu đầu")
    args = parser.parse_args()

    # Load input
    samples = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if args.dry_run:
        samples = samples[:3]
        print(f"[DRY RUN] Chỉ chạy {len(samples)} câu\n")
    else:
        print(f"Tổng: {len(samples)} câu\n")

    # Load done IDs (resume)
    done_ids = set()
    if Path(DONE_IDS_FILE).exists():
        with open(DONE_IDS_FILE, encoding="utf-8") as f:
            done_ids = {l.strip() for l in f if l.strip()}
        print(f"Resume: {len(done_ids)} câu đã xong, bỏ qua.\n")

    pending = [s for s in samples if s.get("code", s.get("no", "")) not in done_ids]
    print(f"Còn lại: {len(pending)} câu cần xử lý\n")
    print("=" * 60)

    Path(OUTPUT_CORPUS).parent.mkdir(parents=True, exist_ok=True)
    fout        = open(OUTPUT_CORPUS,   "a", encoding="utf-8")
    fdone       = open(DONE_IDS_FILE,   "a", encoding="utf-8")
    ffailed     = open(FAILED_LOG,      "a", encoding="utf-8")

    total_chunks  = 0
    total_success = 0

    for idx, sample in enumerate(pending):
        qa_id    = str(sample.get("code", sample.get("no", idx)))
        question = sample["question"]
        answer   = sample["answer"]
        src_url  = sample.get("url", "")
        src_domain = urlparse(src_url).netloc if src_url else "viendinhduong.vn"

        print(f"\n[{idx+1}/{len(pending)}] {qa_id}")
        print(f"  Q: {question[:80]}...")

        # Bước 1: Extract aspects
        aspects = extract_aspects(question, answer)
        if not aspects:
            print("  Không tách được aspects → skip")
            ffailed.write(f"{qa_id}\n")
            ffailed.flush()
            continue

        for a in aspects:
            print(f"    - {a['aspect']}")

        # Bước 2: Search — câu hỏi gốc + các aspect queries
        seen_urls = set()
        candidate_urls = []

        # Thêm câu hỏi gốc làm query đầu tiên
        all_queries = [question] + [
            q
            for aspect in aspects
            for q in [aspect.get("query_vi", ""), aspect.get("query_en", "")]
            if q
        ]

        for query in all_queries:
            urls = ddg_search(query, src_domain, num=MAX_SEARCH_RESULTS)
            for u in urls:
                if u not in seen_urls:
                    seen_urls.add(u)
                    candidate_urls.append(u)
            time.sleep(SEARCH_PAUSE)

            if len(candidate_urls) >= MAX_URLS_PER_QA:
                break

        candidate_urls = candidate_urls[:MAX_URLS_PER_QA]
        print(f"  URLs tìm được: {len(candidate_urls)}")

        # Bước 3: Crawl + score
        scored_articles = []
        for url in candidate_urls:
            text = crawl_url(url)
            if not text:
                continue
            chunks = chunk_text(text)
            scored = score_chunks(chunks, question)
            if scored:
                best = max(c["score"] for c in scored)
                scored_articles.append({"url": url, "text": text, "best_score": best})
            time.sleep(CRAWL_PAUSE)

        scored_articles.sort(key=lambda x: -x["best_score"])
        top = scored_articles[:TOP_ARTICLES]

        if not top:
            print("  Không tìm được bài liên quan")
            ffailed.write(f"{qa_id}\n")
            ffailed.flush()
            continue

        # Bước 4: Lưu corpus
        n_chunks = 0
        for article in top:
            rows = make_corpus_rows(
                url=article["url"],
                text=article["text"],
                source_domain=urlparse(article["url"]).netloc,
                question=question,
            )
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_chunks += len(rows)

        fout.flush()
        fdone.write(f"{qa_id}\n")
        fdone.flush()

        total_chunks  += n_chunks
        total_success += 1
        print(f"  OK: {len(top)} bài, {n_chunks} chunks (best={top[0]['best_score']:.3f})")

    fout.close()
    fdone.close()
    ffailed.close()

    print(f"""
{"="*60}
XONG!
  Câu thành công : {total_success} / {len(pending)}
  Chunks mới     : {total_chunks}
  Output         : {OUTPUT_CORPUS}
  Done IDs       : {DONE_IDS_FILE}
  Failed log     : {FAILED_LOG}
{"="*60}
""")


if __name__ == "__main__":
    main()
