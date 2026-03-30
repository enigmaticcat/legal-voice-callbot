"""
clean_corpus.py
===============
Post-process nutrition_corpus.jsonl in-place:
  1. Strip leading navigation noise  (☰, Mục lục)
  2. Strip trailing CTA block        (Đặt lịch / HOTLINE / MyVinmec)
  3. Remove quiz pages               (trac-nghiem in URL)
  4. Drop docs with content < MIN_CHARS after cleaning
"""

import json
import re
import shutil
from pathlib import Path

INPUT  = Path("/Users/nguyenthithutam/Desktop/Callbot/nutrition_corpus.jsonl")
OUTPUT = Path("/Users/nguyenthithutam/Desktop/Callbot/nutrition_corpus_clean.jsonl")
MIN_CHARS = 300


# ─── Cleaning rules ───────────────────────────────────────────────────────────

def clean_vinmec(doc: dict) -> dict | None:
    # Drop quiz/interactive pages
    if "trac-nghiem" in doc.get("url", ""):
        return None

    content = doc["content"]

    # 1. Strip leading nav noise (☰ and/or "Mục lục" at the very top)
    content = re.sub(r"^(☰\s*\n?|Mục lục\s*\n?)+", "", content).strip()

    # 2. Strip trailing CTA block starting from "Để đặt lịch khám"
    content = re.sub(
        r"\nĐể đặt lịch khám[\s\S]*$",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()

    if len(content) < MIN_CHARS:
        return None

    doc["content"] = content
    return doc


CLEANERS = {
    "vinmec": clean_vinmec,
}


def clean_doc(doc: dict) -> dict | None:
    cleaner = CLEANERS.get(doc.get("source"))
    if cleaner:
        return cleaner(doc)
    return doc  # no cleaner defined → keep as-is


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    total = kept = dropped_quiz = dropped_short = 0

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

            url = doc.get("url", "")
            if "trac-nghiem" in url:
                dropped_quiz += 1
                continue

            cleaned = clean_doc(doc)
            if cleaned is None:
                dropped_short += 1
                continue

            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Total read:      {total}")
    print(f"Kept:            {kept}")
    print(f"Dropped (quiz):  {dropped_quiz}")
    print(f"Dropped (short): {dropped_short}")
    print(f"Output: {OUTPUT}")

    # Verify a sample
    print("\n--- Sample after cleaning (doc #0 tail) ---")
    with open(OUTPUT, encoding="utf-8") as f:
        first = json.loads(f.readline())
    print(f"Title:   {first['title']}")
    print(f"Length:  {len(first['content'])} chars")
    print(f"Head:    {first['content'][:200]!r}")
    print(f"Tail:    {first['content'][-200:]!r}")


if __name__ == "__main__":
    main()
