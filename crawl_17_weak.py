"""
crawl_17_weak.py
Crawl thêm corpus cho 17 câu hỏi có score < 0.9.
Chỉ append vào data_final/corpus_final.jsonl nếu best_score >= 0.9.
"""

import os, json, time, textwrap, hashlib, uuid
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parent
OUTPUT_FILE = ROOT / "data_final" / "corpus_final.jsonl"
LOG_FILE    = ROOT / "crawl_17_weak_log.jsonl"
FAILED_LOG  = ROOT / "crawl_17_weak_failed.txt"

GROQ_MODEL           = "llama-3.1-8b-instant"
MAX_URLS_PER_QUESTION = 12
SCORE_THRESHOLD       = 0.9
TOP_ARTICLES          = 3
SEARCH_PAUSE_SEC      = 1.5

WEAK_QUESTIONS = [
    {"id": "vdd_s1_087",    "question": "Vì sao sữa là thức ăn tốt cho người ốm và người già?"},
    {"id": "vdd_s4_072",    "question": "Có nên cho thêm đậu đỗ, hạt sen vào xay lẫn bột cho trẻ không?"},
    {"id": "vdd_s3_077",    "question": "Nguồn cung cấp vitamin ở đâu?"},
    {"id": "thucuc_s2_071", "question": "Dạo này em hay mệt, da xỉn, trí nhớ cũng kém đi… Có phải do ăn uống thiếu chất không bác sĩ? Em nghe nói ăn nhiều rau củ tốt cho sức khỏe, nhưng không biết loại nào thực sự 'xứng đáng' để ưu tiên mỗi ngày?"},
    {"id": "thucuc_s3_070", "question": "Mấy hôm nay em cứ mệt mỏi, hắt hơi liên tục, người thì nóng nhẹ, cổ họng rát… Em lo bị cảm nên muốn biết uống gì để giải cảm nhanh mà không cần dùng thuốc ngay. Có cách nào tự nhiên, an toàn nhưng vẫn hiệu quả không bác sĩ?"},
    {"id": "thucuc_s4_076", "question": "Dạo này em thấy người lúc nào cũng mệt mỏi, da xỉn màu, hay đầy bụng và khó tiêu. Nghe bạn bè rủ thử 'detox' để thải độc cơ thể, nhưng em không biết liệu việc này có thực sự hiệu quả hay chỉ là trào lưu? Liệu cơ thể mình có tự thải độc được không, hay phải nhờ đến các phương pháp hỗ trợ?"},
    {"id": "thucuc_s3_088", "question": "Mỗi sáng em đều vắt một cốc cam tươi để uống vì nghe nói tốt cho sức khỏe và đẹp da. Nhưng dạo gần đây, em thấy hơi ợ chua và đau bụng nhẹ sau khi uống. Liệu có phải do em uống cam mỗi ngày không? Hay là em đang làm sai điều gì?"},
    {"id": "thucuc_s5_142", "question": "Mẹ tôi vừa trải qua một đợt ốm nặng, ăn uống rất kém, người cứ gầy sọp đi. Tôi muốn mua sữa để bồi bổ nhưng không biết loại nào phù hợp với tình trạng của mẹ — có loại nào vừa dễ uống, lại đủ dinh dưỡng mà không ảnh hưởng đến bệnh nền như tiểu đường hay huyết áp không, thưa bác sĩ?"},
    {"id": "vdd_s4_050",    "question": "Mỗi khi uống sữa hay bị đau bụng đi ngoài thì nên sử dụng sữa gì là tốt nhất"},
    {"id": "thucuc_s5_020", "question": "Em ăn toàn trái cây thôi, nghĩ là lành mạnh và giúp giảm cân, nhưng dạo này lại thấy cân nặng cứ tăng dần. Không hiểu tại sao ăn 'thực phẩm sạch' mà vẫn lên ký? Liệu có phải do em ăn sai cách hay do loại trái cây mình chọn?"},
    {"id": "vdd_s1_108",    "question": "Nên sử dụng chất béo như thế nào cho hợp lý?"},
    {"id": "thucuc_s4_008", "question": "Dạo này em hay cảm thấy tim đập nhanh, tay chân tê bì, đôi lúc còn thấy như nghẹn ở cổ, hít thở không sâu được. Em lo lắng liệu có phải do thiếu canxi hay không, vì trước giờ em ít uống sữa và cũng không bổ sung gì cả."},
    {"id": "thucuc_s5_070", "question": "Dạo này tôi hay quên, chóng mặt khi đứng dậy và cảm thấy đầu óc nặng nề, kém minh mẫn. Nghe nói có những loại thực phẩm giúp 'bổ máu não' – nhưng không biết cụ thể nên ăn gì để cải thiện tình trạng này?"},
    {"id": "vdd_s4_080",    "question": "Để xương chắc khỏe, canxi cần được đưa vào khẩu phần như thế nào?"},
    {"id": "vdd_s2_030",    "question": "Làm thế nào để biết số cân nặng nên có?"},
    {"id": "vdd_s3_085",    "question": "Để chế biến các loại thực phẩm làm thức ăn cho cả tuần thì phải làm như thế nào?"},
    {"id": "vdd_s1_086",    "question": "Hàng tháng trẻ vẫn tăng cân nhưng chưa đạt chuẩn thì có đáng lo không"},
]

# ==============================================================
# IMPORTS & INIT
# ==============================================================

import torch
from groq import Groq
from ddgs import DDGS
import trafilatura
from sentence_transformers import CrossEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Load reranker trên {device.upper()}...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)
print("Reranker OK")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
print("Groq OK\n")

# ==============================================================
# HELPERS
# ==============================================================

def make_chunk_id(url: str, idx: int) -> str:
    h = hashlib.md5(f"{url}_{idx}".encode()).hexdigest()
    return str(uuid.UUID(h))

def make_doc_id(url: str) -> str:
    h = hashlib.md5(url.encode()).hexdigest()
    return str(uuid.UUID(h))

def generate_queries(question: str) -> list:
    prompt = textwrap.dedent(f"""
        Tạo 4 query tìm kiếm tiếng Việt ngắn gọn (3-8 từ) để tìm bài y tế/dinh dưỡng.
        Quy tắc:
        - Mỗi dòng là 1 query
        - Chỉ viết từ khóa, không viết câu dài, không có giải thích
        - Không đánh số, không có dấu gạch đầu dòng
        - Viết như từ khóa search thực tế

        Câu hỏi: {question[:200]}
        Output:
    """).strip()
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        return [q.strip() for q in raw.split("\n") if q.strip()][:4]
    except Exception as e:
        print(f"  Groq error: {e}")
        return [question[:80]]

def search_urls(queries: list) -> list:
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
            time.sleep(0.3)
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
    if not chunks:
        return []
    pairs = [[question, chunk] for chunk in chunks]
    batch = 128 if device == "cuda" else 32
    scores = reranker.predict(pairs, batch_size=batch)
    scored = [
        {"text": c, "score": round(float(s), 4)}
        for c, s in zip(chunks, scores)
    ]
    return sorted(scored, key=lambda x: -x["score"])

# Build existing URL set to avoid exact duplicates
print("Loading existing URLs from corpus_final...")
existing_urls = set()
with open(OUTPUT_FILE, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        u = d.get("url") or d.get("source", "")
        if u:
            existing_urls.add(u)
print(f"  {len(existing_urls):,} existing URLs\n")

# ==============================================================
# MAIN LOOP
# ==============================================================

results_log = []
failed = []
total_added = 0

for entry in WEAK_QUESTIONS:
    qid      = entry["id"]
    question = entry["question"]
    print(f"\n{'='*60}")
    print(f"[{qid}] {question[:80]}...")

    # 1. Generate queries
    queries = generate_queries(question)
    print(f"  Queries: {queries}")

    # 2. Search
    urls = search_urls(queries)
    new_urls = [u for u in urls if u not in existing_urls]
    print(f"  URLs: {len(urls)} total, {len(new_urls)} new")

    # 3. Crawl + score
    found_articles = []
    for url in urls:  # score all URLs, even if already in corpus — we want best score
        text = crawl_article(url)
        if not text:
            continue
        chunks = chunk_text(text)
        scored = score_chunks(chunks, question)
        if scored:
            best = scored[0]["score"]
            found_articles.append({
                "url": url,
                "text": text,
                "best_score": best,
                "top_chunks": scored[:5],
                "is_new_url": url not in existing_urls,
            })

    found_articles.sort(key=lambda x: -x["best_score"])

    if not found_articles:
        print(f"  No articles found")
        failed.append(qid)
        continue

    best_overall = found_articles[0]["best_score"]
    print(f"  Best score across all articles: {best_overall:.4f}")

    if best_overall < SCORE_THRESHOLD:
        print(f"  Score {best_overall:.4f} < {SCORE_THRESHOLD} — skip")
        failed.append(qid)
        results_log.append({"id": qid, "question": question, "best_score": best_overall, "added": 0})
        continue

    # 4. Only append NEW URLs with score >= threshold
    added_count = 0
    top_new = [a for a in found_articles if a["is_new_url"] and a["best_score"] >= SCORE_THRESHOLD][:TOP_ARTICLES]

    if not top_new:
        # Best article already in corpus — just log it
        print(f"  Best article already in corpus (score={best_overall:.4f}), nothing new to add")
        results_log.append({"id": qid, "question": question, "best_score": best_overall, "added": 0,
                             "note": "already_in_corpus"})
        continue

    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for art in top_new:
            url  = art["url"]
            text = art["text"]
            doc_id = make_doc_id(url)
            # Chunk and write only chunks above threshold
            chunks = chunk_text(text)
            scored = score_chunks(chunks, question)
            good_chunks = [c for c in scored if c["score"] >= SCORE_THRESHOLD]
            if not good_chunks:
                good_chunks = scored[:3]  # fallback: at least write top 3

            for i, chunk in enumerate(good_chunks):
                chunk_id = make_chunk_id(url, i)
                row = {
                    "chunk_id":    chunk_id,
                    "doc_id":      doc_id,
                    "source":      "crawled_weak",
                    "url":         url,
                    "title":       "",
                    "category":    "crawled_weak",
                    "chunk_index": i,
                    "text":        chunk["text"],
                    "embed_text":  chunk["text"],
                    "from_question": question,
                    "question_id": qid,
                    "relevance_score": chunk["score"],
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                added_count += 1
            existing_urls.add(url)
            print(f"  + {url[:70]} (score={art['best_score']:.4f}, {len(good_chunks)} chunks)")

    total_added += added_count
    results_log.append({"id": qid, "question": question, "best_score": best_overall,
                         "added": added_count, "urls_added": [a["url"] for a in top_new]})
    print(f"  Added {added_count} chunks")

    time.sleep(SEARCH_PAUSE_SEC)

# ==============================================================
# REPORT
# ==============================================================

print(f"\n{'='*60}")
print(f"DONE")
print(f"  Total chunks added : {total_added}")
print(f"  Questions improved : {sum(1 for r in results_log if r.get('added', 0) > 0)}")
print(f"  Questions failed   : {len(failed)} — {failed}")

with open(LOG_FILE, "w", encoding="utf-8") as f:
    for r in results_log:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(FAILED_LOG, "w") as f:
    f.write("\n".join(failed))

print(f"  Log: {LOG_FILE}")
print(f"  Failed: {FAILED_LOG}")
