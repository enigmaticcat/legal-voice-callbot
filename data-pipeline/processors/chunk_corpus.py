"""
Chunk full_docs.jsonl thành các đoạn nhỏ để nạp lên Qdrant.

Input: evaluation/full_docs.jsonl
  - 6,001 docs từ 4 nguồn y tế tin cậy (vinmec, suckhoedoisong, benhvienthucuc, viendinhduong)
  - Field: "text" (không phải "content")
  - 100% coverage với synthetic_qa.jsonl (360 câu hỏi eval)

Strategy: Sentence-boundary chunking (v3) — Range [MIN_CHARS, MAX_CHARS]
- Flush khi buffer >= MIN_CHARS VÀ câu tiếp sẽ vượt MAX_CHARS → chunk tự nhiên theo nội dung
- Không overlap → loại bỏ near-duplicate flooding (cũ overlap=1: ~87% adjacent overlap)
- Cắt tại ranh giới câu (.!?) → truncation thật ~0% (2.4% còn lại là danh sách/CTA)
- Không giới hạn chunks/doc → tránh mất nội dung (cũ cap=15: 47% docs bị cắt)
- embed_text = "{title}. {chunk_text}"  (title prefix giúp embedding biết ngữ cảnh bài)

Output: nutrition_chunks_v2.jsonl
Schema mỗi chunk:
  chunk_id    : "{source}_{doc_index}_{chunk_index}"
  doc_id      : url (unique per article)
  source      : vinmec / suckhoedoisong / ...
  url         : str
  title       : str
  category    : str
  chunk_index : int (0-based trong bài)
  text        : nội dung chunk thuần
  embed_text  : text dùng để embed (title + text)
"""

import json
import re
from pathlib import Path
from collections import defaultdict

INPUT_FILE  = "../../evaluation/full_docs.jsonl"
OUTPUT_FILE = "nutrition_chunks_v2.jsonl"

MIN_CHARS        = 150      # flush sớm nhất khi buffer đạt ngưỡng này
MAX_CHARS        = 600      # flush muộn nhất — không để buffer vượt ngưỡng này
MAX_CHUNKS_PER_DOC = 99999  # không giới hạn — diversity xử lý ở tầng retrieval
MIN_CHUNK_CHARS  = 60       # bỏ chunk quá ngắn (tiêu đề lạc, footer...)
OVERLAP_SENTENCES = 0       # không overlap — tránh near-duplicate flooding

# Nguồn y tế / dinh dưỡng được giữ lại
# Các nguồn thương mại (shopee, tiki, dienmayxanh...) bị loại
TRUSTED_SOURCES = {
    # Bệnh viện lớn
    "vinmec", "vinmec.com",
    "benhvienthucuc", "benhvienthucuc.vn",
    "tamanhhospital.vn",
    "hongngochospital.vn",
    "hoanmy.com",
    "tudu.com.vn",
    "bvnguyentriphuong.com.vn",
    "benhviennhitrunguong.gov.vn",
    "benhvien199.vn",
    "phuongchau.com",
    "benhvienphuongdong.vn",
    "umcclinic.com.vn",
    "favinahospital.com",
    "nhidongcantho.org.vn",
    "fvhospital.com",
    # Viện / tổ chức y tế nhà nước
    "viendinhduong",
    "nreci.org",
    "hepa.gov.vn",
    # Báo sức khỏe
    "suckhoedoisong", "suckhoedoisong.vn",
    "giadinh.suckhoedoisong.vn",
    "suckhoe.vtv.vn",
    "suckhoeviet.org.vn",
    "suckhoehangngay.vn",
    "suckhoe123.vn",
    "suckhoegiadinh.com.vn",
    # Nhà thuốc / dược
    "nhathuoclongchau.com.vn",
    "pharmacity.vn",
    "nhathuocankhang.com",
    "nhathuocdominhduong.com",
    "tiemchunglongchau.com.vn",
    # Trang y khoa / bác sĩ
    "hellobacsi.com",
    "hellodoctors.vn",
    "alobacsi.com",
    "docosan.com",
    "ivie.vn",
    "bacsihanh.vn",
    "drnguyenanhtuan.com",
    "drngoc.vn",
    "drthang.vn",
    "drtrang.org",
    "bshuyen.vn",
    "mediplus.vn",
    "medlatec.vn",
    "diag.vn",
    "msdmanuals.com",
    "yhocvasuckhoe.com.vn",
    "doctornetwork.us",
    # Dinh dưỡng chuyên biệt
    "nutrihome.vn",
    "nutricare.com.vn",
    "dinhduongmevabe.com.vn",
    "forikid.vn",
    "bioamicus.vn",
    "medipharusa.com",
    # Dữ liệu xác thực thủ công
    "manual_verified",
}

# Regex nhận biết kết thúc câu tiếng Việt
_SENT_END = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[str]:
    """Tách text thành list câu tại ranh giới .!? + khoảng trắng."""
    parts = _SENT_END.split(text)
    return [s.strip() for s in parts if s.strip()]


def _sentences_to_chunks(sentences: list[str], max_chars: int,
                          max_chunks: int, min_chars: int,
                          overlap: int) -> list[str]:
    """
    Range-based chunking: flush khi buffer >= MIN_CHARS VÀ câu tiếp sẽ vượt MAX_CHARS.
    Câu đơn > MAX_CHARS → chia tại ranh giới từ.
    overlap không dùng (=0), giữ tham số để tương thích interface.
    """
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def _flush():
        nonlocal buf, buf_len
        text = " ".join(buf).strip()
        if len(text) >= min_chars:
            chunks.append(text)
        buf.clear()
        buf_len = 0

    for sent in sentences:
        if len(chunks) >= max_chunks:
            break

        if len(sent) > max_chars:
            # Câu quá dài → flush buffer trước, rồi chia theo từ
            if buf:
                _flush()
            words = sent.split()
            sub_buf = ""
            for word in words:
                if len(chunks) >= max_chunks:
                    break
                trial = (sub_buf + " " + word).strip() if sub_buf else word
                if len(trial) <= max_chars:
                    sub_buf = trial
                else:
                    if sub_buf and len(sub_buf) >= min_chars:
                        chunks.append(sub_buf)
                    sub_buf = word
            if sub_buf and len(sub_buf) >= min_chars and len(chunks) < max_chunks:
                chunks.append(sub_buf)
            buf.clear()
            buf_len = 0
            continue

        add = len(sent) + (1 if buf else 0)
        would_exceed = buf and (buf_len + add > max_chars)
        already_enough = buf_len >= MIN_CHARS

        if would_exceed and already_enough:
            _flush()
            if len(chunks) >= max_chunks:
                break

        buf.append(sent)
        buf_len += add

    if buf and len(chunks) < max_chunks:
        _flush()

    return chunks


def sentence_chunks(content: str, max_chars: int = MAX_CHARS,
                    max_chunks: int = MAX_CHUNKS_PER_DOC,
                    min_chars: int = MIN_CHUNK_CHARS,
                    overlap: int = OVERLAP_SENTENCES) -> list[str]:
    """
    Chia content thành chunks theo range [MIN_CHARS, MAX_CHARS].

    1. Làm phẳng dòng trắng
    2. Tách câu tại .!? + khoảng trắng
    3. Tích lũy câu cho đến khi buffer >= MIN_CHARS, flush khi câu tiếp vượt MAX_CHARS
    4. Câu đơn > MAX_CHARS → chia tại ranh giới từ
    5. Bỏ chunk < MIN_CHUNK_CHARS, không giới hạn số chunks/doc
    """
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    text = " ".join(lines)
    sentences = _split_sentences(text)
    return _sentences_to_chunks(sentences, max_chars, max_chunks, min_chars, overlap)


def make_chunk_id(source: str, doc_idx: int, chunk_idx: int) -> str:
    return f"{source}_{doc_idx}_{chunk_idx}"


def print_stats(output_path: Path):
    records = [json.loads(l) for l in output_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    lens = [len(r["text"]) for r in records]
    lens_sorted = sorted(lens)
    n = len(lens)

    sent_end = re.compile(r'[.?!]\s*$')
    truncated = sum(1 for r in records if not sent_end.search(r["text"].strip()))

    doc_counts: dict[str, int] = defaultdict(int)
    for r in records:
        doc_counts[r["doc_id"]] += 1
    counts = list(doc_counts.values())

    print(f"\nTotal chunks : {n:,}")
    print(f"Total docs   : {len(doc_counts):,}")
    print(f"\nChunk size (chars):")
    print(f"  min    : {min(lens)}")
    print(f"  p25    : {lens_sorted[n // 4]}")
    print(f"  median : {lens_sorted[n // 2]}")
    print(f"  p75    : {lens_sorted[3 * n // 4]}")
    print(f"  max    : {max(lens)}")
    print(f"  avg    : {sum(lens) / n:.0f}")
    print(f"\nTruncated (không kết thúc bằng .!?): {truncated:,} ({truncated / n * 100:.1f}%)")
    print(f"\nChunks/doc — docs có ≥10 chunks  : {sum(1 for c in counts if c >= 10):,}")
    print(f"Chunks/doc — docs có ≥5 chunks   : {sum(1 for c in counts if c >= 5):,}")
    print(f"Chunks/doc — docs có 1 chunk     : {sum(1 for c in counts if c == 1):,}")

    by_source: dict[str, int] = defaultdict(int)
    for r in records:
        by_source[r["source"]] += 1
    print(f"\nChunks per source (top 10):")
    for src, cnt in sorted(by_source.items(), key=lambda x: -x[1])[:10]:
        print(f"  {src}: {cnt:,}")


def main():
    input_path  = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    total_docs   = 0
    total_chunks = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for doc_idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)
            source   = doc.get("source", "unknown")
            url      = doc.get("url", "")
            title    = doc.get("title", "")
            content  = doc.get("text", "")   # full_docs dùng "text", không phải "content"
            category = doc.get("category", "")

            chunks = sentence_chunks(content)
            if not chunks:
                continue

            for chunk_idx, chunk_text in enumerate(chunks):
                # embed_text: title dẫn đầu giúp embedding biết ngữ cảnh bài viết
                embed_text = f"{title}. {chunk_text}" if title else chunk_text

                record = {
                    "chunk_id"   : make_chunk_id(source, doc_idx, chunk_idx),
                    "doc_id"     : url,
                    "source"     : source,
                    "url"        : url,
                    "title"      : title,
                    "category"   : category,
                    "chunk_index": chunk_idx,
                    "text"       : chunk_text,
                    "embed_text" : embed_text,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

            total_docs += 1

    print(f"Done: {total_docs:,} docs → {total_chunks:,} chunks")
    print(f"Output: {output_path}")

    print_stats(output_path)


if __name__ == "__main__":
    main()
