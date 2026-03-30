"""
Chunk corpus_all.jsonl thành các đoạn nhỏ để nạp lên Qdrant.

Strategy: Line-merge chunking
- Split content theo '\n'
- Gom dòng cho đến ~700 chars
- Overlap: giữ lại 2 dòng cuối của chunk trước làm đầu chunk tiếp
- embed_text = "{title}\n{chunk_text}"

Output: nutrition_chunks.jsonl
Schema mỗi chunk:
  chunk_id    : "{source}_{doc_index}_{chunk_index}"
  doc_id      : url (unique per article)
  source      : vinmec / skds / vdd
  url         : str
  title       : str
  category    : str
  chunk_index : int (0-based trong bài)
  text        : nội dung chunk thuần
  embed_text  : text dùng để embed (title + text)
"""

import json
import hashlib
from pathlib import Path

INPUT_FILE = "corpus_all.jsonl"
OUTPUT_FILE = "nutrition_chunks.jsonl"
MAX_CHARS = 700
OVERLAP_LINES = 2


def line_merge_chunks(content: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_LINES):
    """
    Chia content thành các chunk bằng cách gom dòng.
    - Tách theo '\n', bỏ dòng trắng
    - Gom dòng cho đến max_chars
    - Carry OVERLAP_LINES dòng cuối vào chunk kế tiếp
    """
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if not lines:
        return []

    chunks = []
    buffer = []
    buffer_len = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        # +1 cho '\n' nối giữa các dòng
        add_len = len(line) + (1 if buffer else 0)

        if buffer and buffer_len + add_len > max_chars:
            # Lưu chunk hiện tại
            chunks.append("\n".join(buffer))
            # Overlap: giữ lại overlap dòng cuối
            overlap_lines = buffer[-overlap:] if len(buffer) >= overlap else buffer[:]
            buffer = overlap_lines
            buffer_len = sum(len(l) for l in buffer) + max(0, len(buffer) - 1)

        buffer.append(line)
        buffer_len += add_len
        i += 1

    if buffer:
        chunks.append("\n".join(buffer))

    return chunks


def make_chunk_id(source: str, doc_idx: int, chunk_idx: int) -> str:
    return f"{source}_{doc_idx}_{chunk_idx}"


def main():
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    total_docs = 0
    total_chunks = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for doc_idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)
            source = doc.get("source", "unknown")
            url = doc.get("url", "")
            title = doc.get("title", "")
            content = doc.get("content", "")
            category = doc.get("category", "")

            chunks = line_merge_chunks(content)
            if not chunks:
                continue

            for chunk_idx, chunk_text in enumerate(chunks):
                embed_text = f"{title}\n{chunk_text}" if title else chunk_text

                record = {
                    "chunk_id": make_chunk_id(source, doc_idx, chunk_idx),
                    "doc_id": url,
                    "source": source,
                    "url": url,
                    "title": title,
                    "category": category,
                    "chunk_index": chunk_idx,
                    "text": chunk_text,
                    "embed_text": embed_text,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

            total_docs += 1

    print(f"Done: {total_docs} docs -> {total_chunks} chunks")
    print(f"Output: {output_path}")

    # Quick stats
    with open(output_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    lens = [len(r["text"]) for r in records]
    print(f"\nChunk length stats (chars):")
    print(f"  min  : {min(lens)}")
    print(f"  max  : {max(lens)}")
    print(f"  avg  : {sum(lens)/len(lens):.0f}")
    print(f"  median: {sorted(lens)[len(lens)//2]}")

    by_source = {}
    for r in records:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print(f"\nChunks per source:")
    for src, cnt in sorted(by_source.items()):
        print(f"  {src}: {cnt}")


if __name__ == "__main__":
    main()
