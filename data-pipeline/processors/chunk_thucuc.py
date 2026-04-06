"""
chunk_thucuc.py
===============
Chunk thucuc_articles_corpus.jsonl → thucuc_chunks.jsonl
Dùng cùng chiến lược line-merge như chunk_corpus.py (max 700 chars, overlap 2 dòng).

Chạy từ thư mục gốc project:
  python data-pipeline/processors/chunk_thucuc.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE  = ROOT / "thucuc_articles_corpus.jsonl"
OUTPUT_FILE = ROOT / "thucuc_chunks.jsonl"

MAX_CHARS     = 700
OVERLAP_LINES = 2


def line_merge_chunks(content: str) -> list[str]:
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if not lines:
        return []

    chunks, buffer, buffer_len = [], [], 0

    for line in lines:
        add_len = len(line) + (1 if buffer else 0)
        if buffer and buffer_len + add_len > MAX_CHARS:
            chunks.append("\n".join(buffer))
            overlap = buffer[-OVERLAP_LINES:] if len(buffer) >= OVERLAP_LINES else buffer[:]
            buffer = overlap
            buffer_len = sum(len(l) for l in buffer) + max(0, len(buffer) - 1)
        buffer.append(line)
        buffer_len += add_len

    if buffer:
        chunks.append("\n".join(buffer))
    return chunks


def main():
    total_docs = total_chunks = 0

    with open(INPUT_FILE, encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for doc_idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)

            chunks = line_merge_chunks(doc.get("content", ""))
            if not chunks:
                continue

            for chunk_idx, chunk_text in enumerate(chunks):
                title = doc.get("title", "")
                record = {
                    "chunk_id"   : f"benhvienthucuc_{doc_idx}_{chunk_idx}",
                    "doc_id"     : doc.get("url", ""),
                    "source"     : "benhvienthucuc",
                    "url"        : doc.get("url", ""),
                    "title"      : title,
                    "category"   : doc.get("category", "dinh-duong"),
                    "chunk_index": chunk_idx,
                    "text"       : chunk_text,
                    "embed_text" : f"{title}\n{chunk_text}" if title else chunk_text,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

            total_docs += 1

    print(f"Done: {total_docs} docs → {total_chunks} chunks")
    print(f"Output: {OUTPUT_FILE}")

    # Stats
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    lens = sorted(len(r["text"]) for r in records)
    n = len(lens)
    print(f"\nChunk length (chars): min={lens[0]}  avg={sum(lens)//n}  median={lens[n//2]}  max={lens[-1]}")


if __name__ == "__main__":
    main()
