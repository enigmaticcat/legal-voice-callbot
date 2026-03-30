"""
clean_skds_corpus.py
====================
Clean skds_corpus.jsonl in-place:
  1. Strip trailing "Xem thêm video" / "Mời bạn xem tiếp video" blocks
  2. Strip trailing related-article teasers (SKĐS - ...)
  3. Drop docs with content < MIN_CHARS after cleaning
"""

import json
import re
from pathlib import Path

INPUT  = Path("/Users/nguyenthithutam/Desktop/Callbot/skds_corpus.jsonl")
OUTPUT = Path("/Users/nguyenthithutam/Desktop/Callbot/skds_corpus_clean.jsonl")
MIN_CHARS = 300

# Patterns matched from the tail — strip everything from first match onward
TAIL_PATTERNS = [
    # "Mời bạn/độc giả xem tiếp/thêm video..." (with optional \xa0 prefix)
    r"[\n\xa0]+(?:Mời\s+(?:bạn|độc\s+giả)\s+xem\s+(?:tiếp|thêm)|Xem\s+thêm)\s+video[^\n]*:?[\s\S]*$",
    # "<related title>\n\nSKĐS - <teaser>"  (related article appended)
    r"\n[^\n]{5,120}\n\nSKĐS\s*[-–]\s*[\s\S]*$",
    # "| SKĐS" suffix (with or without leading newline)
    r"\s*\|\s*SKĐS\w*\s*$",
    # Bare "\nSKĐS - <teaser>" at tail
    r"\nSKĐS\s*[-–].{0,300}$",
]

TAIL_RE = re.compile("|".join(TAIL_PATTERNS), re.IGNORECASE)


def clean_skds(doc: dict):
    content = doc["content"]

    # Apply tail-strip iteratively until stable (patterns can nest)
    for _ in range(3):
        cleaned = TAIL_RE.sub("", content).strip()
        if cleaned == content:
            break
        content = cleaned

    if len(content) < MIN_CHARS:
        return None

    doc["content"] = content
    return doc


def main():
    total = kept = dropped = 0

    with open(INPUT, encoding="utf-8") as fin, \
         open(OUTPUT, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            cleaned = clean_skds(doc)
            if cleaned is None:
                dropped += 1
                continue

            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Total read : {total}")
    print(f"Kept       : {kept}")
    print(f"Dropped    : {dropped}")
    print(f"Output     : {OUTPUT}")

    # Verify
    print("\n--- Sample tail after cleaning ---")
    with open(OUTPUT, encoding="utf-8") as f:
        docs = [json.loads(l) for l in f if l.strip()]

    import random
    for d in random.sample(docs, min(5, len(docs))):
        print(f"  [{len(d['content'])}c] {d['title'][:55]}")
        print(f"  tail: {d['content'][-150:]!r}")
        print()

    # Stats
    lengths = sorted(len(d["content"]) for d in docs)
    n = len(lengths)
    print(f"Content length — min:{lengths[0]:,}  median:{lengths[n//2]:,}  max:{lengths[-1]:,}")

    # Check residual boilerplate
    residual = sum(1 for d in docs if "SKĐS" in d["content"][-200:] or "Xem thêm video" in d["content"][-200:])
    print(f"Residual boilerplate in tail: {residual}/{n}")


if __name__ == "__main__":
    main()
