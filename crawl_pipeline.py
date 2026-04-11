"""
crawl_pipeline.py
Chạy: python crawl_pipeline.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ==============================================================
# ⚙️ CẤU HÌNH — CHỈ CẦN ĐIỀN VÀO ĐÂY
# ==============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Lấy tại: https://console.groq.com → API Keys → Create API Key
# Đặt vào file .env: GROQ_API_KEY=gsk_...

# ----------------------------------------------------------
# (Không cần đổi gì bên dưới nếu không muốn)
# ----------------------------------------------------------

GROQ_MODEL          = "llama-3.1-8b-instant"   # free tier, nhanh
GAP_QUESTIONS_PATH  = "crawl_remaining_58.jsonl"   # 58 câu cần crawl thêm
OUTPUT_JSONL        = "new_corpus_chunks.jsonl"         # append vào file cũ
FAILED_LOG          = "crawl_failed_extra.txt"
CHECKPOINT_DIR      = "checkpoints_extra"

MAX_URLS_PER_QUESTION = 10
SIMILARITY_THRESHOLD  = 0      # cross-encoder logit: > 0 = relevant
TOP_K_CHUNKS          = 5
CHECKPOINT_EVERY      = 20
SEARCH_PAUSE_SEC      = 1.5

# File gap_questions_remaining.jsonl đã chứa đúng tập cần crawl
# không cần filter thêm

# ==============================================================
# IMPORTS
# ==============================================================

import os, json, time, textwrap
from pathlib import Path

import torch
from groq import Groq
from ddgs import DDGS
import trafilatura
from sentence_transformers import CrossEncoder

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==============================================================
# KHỞI TẠO
# ==============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Load reranker model (lần đầu tải ~570MB) trên {device.upper()}...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)
print(f"✅ Reranker model OK ({device.upper()})")

groq_client = Groq(api_key=GROQ_API_KEY)
print("✅ Groq client OK\n")

# ==============================================================
# LOAD DỮ LIỆU
# ==============================================================

gap_questions = []
with open(GAP_QUESTIONS_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            gap_questions.append(json.loads(line))

to_process = gap_questions

print(f"🎯 Sẽ xử lý: {len(to_process)} câu\n")

# ==============================================================
# CÁC HÀM
# ==============================================================

def generate_queries(question: str) -> list:
    prompt = textwrap.dedent(f"""
        Tạo 4 query tìm kiếm tiếng Việt ngắn gọn (3-8 từ) để tìm bài y tế/dinh dưỡng.
        Quy tắc:
        - Mỗi dòng là 1 query
        - Chỉ viết từ khóa, không viết câu dài, không có giải thích
        - Không đánh số, không có dấu gạch đầu dòng, không bắt đầu bằng "Tìm kiếm" hay "Google"
        - Viết như từ khóa search thực tế

        Ví dụ:
        Câu hỏi: "Mạng thai có Ữn ốc được không?"
        Output:
        mang thai ăn ốc được không
        ốc sắn thầu đủ thai kỳ
        dinh dưỡng bà bầu thủy hải sản
        thai phụ ăn ốc an toàn

        Câu hỏi: {question}
        Output:
    """).strip()

    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    raw = resp.choices[0].message.content.strip()
    return [q.strip() for q in raw.split("\n") if q.strip()][:4]


def search_urls(queries: list) -> list:
    """Search open, không filter domain — chỉ dedup URL."""
    urls = []
    with DDGS() as ddgs:
        for query in queries:
            try:
                results = ddgs.text(query, max_results=6)
                for r in results:
                    url = r.get("href", "")
                    if url and url not in urls:
                        urls.append(url)
            except Exception:
                continue
    return urls[:MAX_URLS_PER_QUESTION]


def crawl_article(url: str):
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        return text if text and len(text) > 200 else None
    except Exception:
        return None


def chunk_text(text: str, window=200, stride=150) -> list:
    words = text.split()
    return [
        " ".join(words[i : i + window])
        for i in range(0, len(words), stride)
        if len(" ".join(words[i : i + window])) >= 80
    ]


def score_chunks(chunks: list, question: str) -> list:
    """Score chunks bằng cross-encoder (chính xác hơn bi-encoder)."""
    if not chunks:
        return []
    pairs = [[question, chunk] for chunk in chunks]
    batch = 128 if device == "cuda" else 32
    scores = reranker.predict(pairs, batch_size=batch)
    return [
        {"text": c, "score": round(float(s), 4)}
        for c, s in zip(chunks, scores)
        if s >= SIMILARITY_THRESHOLD
    ]


# ==============================================================
# PIPELINE CHÍNH
# ==============================================================

new_corpus     = []
failed_qs      = []
total_articles = 0
start_idx      = 0

# Đọc checkpoint nếu chạy dở
ckpt_files = sorted(Path(CHECKPOINT_DIR).glob("checkpoint_*.json"))
if ckpt_files:
    with open(ckpt_files[-1], encoding="utf-8") as f:
        saved = json.load(f)
    new_corpus     = saved["corpus"]
    failed_qs      = saved["failed"]
    start_idx      = saved["next_idx"]
    total_articles = sum(len(x.get("articles", [])) for x in new_corpus)
    print(f"♻️  Tiếp tục từ checkpoint '{ckpt_files[-1].name}' — câu #{start_idx}\n")

print(f"🚀 Bắt đầu...\n{'='*60}")

for idx in range(start_idx, len(to_process)):
    q        = to_process[idx]
    question = q["question"]
    recall   = q.get("context_recall", "?")

    print(f"\n[{idx+1}/{len(to_process)}] recall={recall}")
    print(f"  Q: {question[:80]}...")

    try:
        # 1. Sinh query
        queries = generate_queries(question)
        print(f"  🔍 {queries}")

        # 2. Search
        urls = search_urls(queries)
        print(f"  🌐 {len(urls)} URLs")

        # 3. Crawl + filter bằng score
        found_articles = []
        for url in urls:
            text = crawl_article(url)
            if not text:
                continue
            chunks = chunk_text(text)
            scored = score_chunks(chunks, question)
            
            if scored:
                # Nếu có chunk liên quan, thì bài này đáng để lấy toàn bộ
                best_score = max(c["score"] for c in scored)
                found_articles.append({
                    "url": url,
                    "text": text,
                    "best_score": best_score
                })

        # 4. Top K
        found_articles.sort(key=lambda x: -x["best_score"])
        top_articles = found_articles[:3] # Lưu top 3 bài báo đầy đủ mỗi câu

        if top_articles:
            new_corpus.append({
                "question":       question,
                "context_recall": recall,
                "articles":       top_articles,
                "num_urls_tried": len(urls),
            })
            total_articles += len(top_articles)
            print(f"  ✅ {len(top_articles)} articles đầy đủ (best={top_articles[0]['best_score']:.3f})")
            
            # Ghi trực tiếp luôn vào file để không mất dữ liệu
            with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                for article in top_articles:
                    f.write(json.dumps({
                        "text":            article["text"],
                        "source":          article["url"],
                        "relevance_score": article["best_score"],
                        "from_question":   question,
                        "original_recall": recall,
                    }, ensure_ascii=False) + "\n")
        else:
            failed_qs.append(question)
            print(f"  ❌ Không tìm được bài báo liên quan")

    except Exception as e:
        print(f"  ⚠️  Lỗi: {e}")
        failed_qs.append(question)

    # Checkpoint
    if (idx + 1) % CHECKPOINT_EVERY == 0 or (idx + 1) == len(to_process):
        ckpt_path = f"{CHECKPOINT_DIR}/checkpoint_{idx+1:04d}.json"
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"corpus": new_corpus, "failed": failed_qs, "next_idx": idx + 1},
                      f, ensure_ascii=False, indent=2)
        print(f"  💾 Checkpoint: {ckpt_path}")

    time.sleep(SEARCH_PAUSE_SEC)

# ==============================================================
# EXPORT
# ==============================================================

# File JSONL đã được lưu liên tục ở trên vòng lặp
# Chỉ dọn dẹp và in báo cáo ở đây

with open(FAILED_LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(failed_qs))

print(f"""
{'='*60}
✅ XONG!
   Câu thành công : {len(new_corpus)} / {len(to_process)}
   Câu thất bại   : {len(failed_qs)}
   Chunks mới     : {total_articles}
   Output         : {OUTPUT_JSONL}
   Failed log     : {FAILED_LOG}
{'='*60}
""")
