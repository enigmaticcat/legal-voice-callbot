"""
Phân tích độ dài ký tự của các chunk trong nutrition_chunks.jsonl
"""
import json
from pathlib import Path
from collections import Counter

CHUNK_FILE = Path(__file__).parent / "nutrition_chunks.jsonl"

chunks = []
with open(CHUNK_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))

lens = [len(c["text"]) for c in chunks]
lens_sorted = sorted(lens)
n = len(lens_sorted)

# Percentiles
def pct(p):
    idx = int(n * p / 100)
    return lens_sorted[min(idx, n - 1)]

print(f"Tổng số chunks : {n:,}")
print(f"Min            : {min(lens)} ký tự")
print(f"Max            : {max(lens)} ký tự")
print(f"Trung bình     : {sum(lens)/n:.0f} ký tự")
print(f"Median (p50)   : {pct(50)} ký tự")
print(f"p25            : {pct(25)} ký tự")
print(f"p75            : {pct(75)} ký tự")
print(f"p90            : {pct(90)} ký tự")
print(f"p95            : {pct(95)} ký tự")
print(f"p99            : {pct(99)} ký tự")

# Phân phối theo bucket
buckets = [0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000, 99999]
labels  = ["<100", "100–200", "200–300", "300–400", "400–500",
           "500–600", "600–700", "700–800", "800–1000",
           "1000–1500", "1500–2000", ">2000"]

counts = [0] * len(labels)
for l in lens:
    for i in range(len(buckets) - 1):
        if buckets[i] <= l < buckets[i + 1]:
            counts[i] += 1
            break

print("\nPhân phối độ dài chunk:")
print(f"{'Khoảng':<14} {'Số chunk':>10} {'Tỉ lệ':>8}")
print("-" * 35)
for label, count in zip(labels, counts):
    bar = "█" * int(count / n * 50)
    print(f"{label:<14} {count:>10,}  {count/n*100:>5.1f}%  {bar}")

# Chunks vượt giới hạn an toàn (~1200 ký tự ~ 300 tokens)
over_1200 = sum(1 for l in lens if l > 1200)
over_2000 = sum(1 for l in lens if l > 2000)
print(f"\nChunk > 1200 ký tự (~300 tokens): {over_1200:,}  ({over_1200/n*100:.1f}%)")
print(f"Chunk > 2000 ký tự (~500 tokens): {over_2000:,}  ({over_2000/n*100:.1f}%)")

# Ví dụ chunk dài nhất
longest = max(chunks, key=lambda c: len(c["text"]))
print(f"\nChunk dài nhất: {len(longest['text'])} ký tự")
print(f"  doc_id : {longest['url'][:80]}")
print(f"  text   : {longest['text'][:200]}...")
