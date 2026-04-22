"""
Step 0: Filter và chuẩn hóa bài viết gốc từ corpus_all.jsonl

Input : corpus_all.jsonl (6,065 bài gốc, chưa chunk)
Output: evaluation/full_docs.jsonl

Mỗi dòng output:
{
  "doc_id": str,      # url dùng làm id
  "source": str,
  "title": str,
  "url": str,
  "text": str,        # content đã clean
  "word_count": int
}
"""

import json
from pathlib import Path
from collections import Counter

CORPUS_PATH = Path(__file__).resolve().parent.parent / "corpus_all.jsonl"
OUTPUT_PATH = Path(__file__).resolve().parent / "full_docs.jsonl"

MIN_WORD_COUNT = 150
MAX_WORD_COUNT = 3000  # bài quá dài thường là aggregate/tổng hợp kém chất lượng


def clean_text(text: str) -> str:
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)


def main():
    docs = []
    skipped_short = skipped_long = skipped_dup = 0
    seen_urls = set()

    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            url = d.get("url", "")
            text = clean_text(d.get("content", ""))
            word_count = len(text.split())

            if url in seen_urls:
                skipped_dup += 1
                continue
            seen_urls.add(url)

            if word_count < MIN_WORD_COUNT:
                skipped_short += 1
                continue

            if word_count > MAX_WORD_COUNT:
                skipped_long += 1
                continue

            docs.append({
                "doc_id": url,
                "source": d.get("source", ""),
                "title": d.get("title", ""),
                "url": url,
                "text": text,
                "word_count": word_count,
            })

    # Stats
    print(f"Skipped duplicate URL : {skipped_dup}")
    print(f"Skipped short (<{MIN_WORD_COUNT}w): {skipped_short}")
    print(f"Skipped long  (>{MAX_WORD_COUNT}w): {skipped_long}")
    print(f"Total kept           : {len(docs):,}")

    src_counts = Counter(d["source"] for d in docs)
    wc = [d["word_count"] for d in docs]
    print(f"\nSource breakdown:")
    for src, cnt in src_counts.most_common():
        print(f"  {src}: {cnt}")
    print(f"\nWord count — min:{min(wc)}  avg:{sum(wc)//len(wc)}  max:{max(wc)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
