"""
Đánh giá các chiến lược chunking — không ghi file output.
So sánh:
  A) Fixed MAX=300 (hiện tại)
  B) Fixed MAX=600
  C) Range MIN=150, MAX=600
  D) Range MIN=200, MAX=700
"""
import json, re
from pathlib import Path
from collections import defaultdict
from statistics import median, mean

INPUT_FILE = "../../evaluation/full_docs.jsonl"
_SENT_END = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> list[str]:
    parts = _SENT_END.split(text)
    return [s.strip() for s in parts if s.strip()]


def chunks_fixed(sentences: list[str], max_chars: int, min_chars: int = 60, overlap: int = 0) -> list[str]:
    """Chiến lược cũ: flush ngay khi thêm câu tiếp theo vượt max_chars."""
    chunks = []
    buf: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        text = " ".join(buf).strip()
        if len(text) >= min_chars:
            chunks.append(text)
        keep = buf[-overlap:] if overlap and len(buf) > overlap else []
        buf = keep
        buf_len = sum(len(s) + 1 for s in buf)

    for sent in sentences:
        if len(sent) <= max_chars:
            add = len(sent) + (1 if buf else 0)
            if buf and buf_len + add > max_chars:
                flush()
            buf.append(sent)
            buf_len += add
        else:
            if buf:
                flush()
            words = sent.split()
            sub = ""
            for w in words:
                trial = (sub + " " + w).strip() if sub else w
                if len(trial) <= max_chars:
                    sub = trial
                else:
                    if sub and len(sub) >= min_chars:
                        chunks.append(sub)
                    sub = w
            if sub and len(sub) >= min_chars:
                chunks.append(sub)
            buf = []
            buf_len = 0

    if buf:
        flush()
    return chunks


def chunks_range(sentences: list[str], min_chars: int, max_chars: int,
                 drop_min: int = 60, overlap: int = 0) -> list[str]:  # noqa: E501
    """
    Range-based: flush khi buffer >= min_chars VÀ câu tiếp sẽ vượt max_chars.
    Nếu buffer chưa đủ min_chars nhưng câu tiếp sẽ vượt max_chars → vẫn flush
    (tránh chunk quá lớn).
    """
    chunks = []
    buf: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        text = " ".join(buf).strip()
        if len(text) >= drop_min:
            chunks.append(text)
        keep = buf[-overlap:] if overlap and len(buf) > overlap else []
        buf = keep
        buf_len = sum(len(s) + 1 for s in buf)

    for sent in sentences:
        if len(sent) > max_chars:
            # Câu quá dài → flush buffer trước, rồi chia theo từ
            if buf:
                flush()
            words = sent.split()
            sub = ""
            for w in words:
                trial = (sub + " " + w).strip() if sub else w
                if len(trial) <= max_chars:
                    sub = trial
                else:
                    if sub and len(sub) >= drop_min:
                        chunks.append(sub)
                    sub = w
            if sub and len(sub) >= drop_min:
                chunks.append(sub)
            buf = []
            buf_len = 0
            continue

        add = len(sent) + (1 if buf else 0)
        would_exceed = buf and (buf_len + add > max_chars)
        already_enough = buf_len >= min_chars

        if would_exceed and already_enough:
            flush()

        buf.append(sent)
        buf_len += add

    if buf:
        flush()
    return chunks


def analyze(all_chunks: list[list[str]]):
    flat = [c for doc in all_chunks for c in doc]
    if not flat:
        return {}
    lens = sorted(len(c) for c in flat)
    n = len(lens)
    sent_end = re.compile(r'[.?!]\s*$')
    truncated = sum(1 for c in flat if not sent_end.search(c.strip()))
    doc_counts = [len(d) for d in all_chunks if d]
    doc_counts_s = sorted(doc_counts)
    nd = len(doc_counts_s)
    return {
        "total_chunks": n,
        "total_docs": len(doc_counts_s),
        "min_len": lens[0],
        "p25_len": lens[n // 4],
        "median_len": lens[n // 2],
        "p75_len": lens[3 * n // 4],
        "max_len": lens[-1],
        "avg_len": int(mean(lens)),
        "truncated_pct": round(truncated / n * 100, 1),
        "median_chunks_per_doc": doc_counts_s[nd // 2],
        "avg_chunks_per_doc": round(mean(doc_counts), 1),
    }


def run():
    docs = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = " ".join(l.strip() for l in d.get("text", "").split("\n") if l.strip())
            docs.append(text)

    configs = [
        ("A: Fixed MAX=300, overlap=1 (hiện tại)", lambda sents: chunks_fixed(sents, max_chars=300, overlap=1)),
        ("B: Range [150,600], overlap=0",           lambda sents: chunks_range(sents, min_chars=150, max_chars=600, overlap=0)),
        ("C: Range [150,600], overlap=1",           lambda sents: chunks_range(sents, min_chars=150, max_chars=600, overlap=1)),
        ("D: Range [150,600], overlap=2",           lambda sents: chunks_range(sents, min_chars=150, max_chars=600, overlap=2)),
        ("E: Range [100,500], overlap=0",           lambda sents: chunks_range(sents, min_chars=100, max_chars=500, overlap=0)),
        ("F: Range [100,500], overlap=1",           lambda sents: chunks_range(sents, min_chars=100, max_chars=500, overlap=1)),
    ]

    for name, fn in configs:
        all_chunks = []
        for text in docs:
            sents = split_sentences(text)
            all_chunks.append(fn(sents))
        stats = analyze(all_chunks)
        print(f"\n{'='*55}")
        print(f"Config: {name}")
        print(f"  Total chunks       : {stats['total_chunks']:,}")
        print(f"  Chunk len (chars)  : min={stats['min_len']} | p25={stats['p25_len']} | median={stats['median_len']} | p75={stats['p75_len']} | max={stats['max_len']} | avg={stats['avg_len']}")
        print(f"  Truncated          : {stats['truncated_pct']}%")
        print(f"  Chunks/doc         : median={stats['median_chunks_per_doc']} | avg={stats['avg_chunks_per_doc']}")


if __name__ == "__main__":
    run()
