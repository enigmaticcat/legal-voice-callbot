# -*- coding: utf-8 -*-
"""
clean_corpus.py
===============
Làm sạch corpus_final.jsonl (stream từng dòng, ít tốn RAM):
  1. Deduplicate: giữ chunk đầu tiên khi text bị trùng
  2. Rechunk chunk > 5000 chars bằng sentence-aware sliding window:
       - Window = 200 words, stride = 150 words (overlap = 50 words)
       - Không cắt giữa câu

Chạy:
  python clean_corpus.py --dry-run
  python clean_corpus.py
"""

import argparse
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path

INPUT_FILE  = Path("data_final/corpus_final.jsonl")
BACKUP_FILE = Path("data_final/corpus_final_backup4.jsonl")
TMP_FILE    = Path("data_final/corpus_final_tmp.jsonl")

MAX_LEN      = 5000
WINDOW_WORDS = 200
STRIDE_WORDS = 150


def split_sentences(text: str) -> list[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    sentences = []
    for para in paras:
        parts = re.split(r"(?<=[.?!])\s+", para)
        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)
    return sentences


def chunk_sentences(text: str, window=WINDOW_WORDS, stride=STRIDE_WORDS) -> list[str]:
    overlap_words = window - stride
    sentences = split_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks = []
    start_idx = 0

    while start_idx < len(sentences):
        # Gom câu từ start_idx cho đến khi đủ window words
        end_idx = start_idx
        wc = 0
        while end_idx < len(sentences):
            sent_wc = len(sentences[end_idx].split())
            if wc + sent_wc > window and end_idx > start_idx:
                break
            wc += sent_wc
            end_idx += 1

        chunks.append(" ".join(sentences[start_idx:end_idx]))

        if end_idx >= len(sentences):
            break

        # Tìm start_idx mới: lùi từ end_idx về ~ overlap_words (track bằng index)
        new_start = end_idx
        ow = 0
        for j in range(end_idx - 1, start_idx - 1, -1):
            s_wc = len(sentences[j].split())
            if ow + s_wc > overlap_words:
                break
            ow += s_wc
            new_start = j

        # Đảm bảo luôn tiến về phía trước, tránh infinite loop
        start_idx = new_start if new_start > start_idx else end_idx

    return [c for c in chunks if c.strip()]


def make_chunk_id(seed: str) -> str:
    h = hashlib.md5(seed.encode()).hexdigest()
    return str(uuid.UUID(h))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seen = set()       # chỉ giữ fingerprint 300 chars (~30MB max)
    removed_dup    = 0
    rechunked_orig = 0
    added_sub      = 0
    kept           = 0
    total_in       = 0

    out = open(TMP_FILE, "w", encoding="utf-8") if not args.dry_run else None

    print("Đang xử lý corpus (stream)...")
    with open(INPUT_FILE, encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            total_in += 1
            r = json.loads(line)
            text = r.get("text", "")

            # Bước 1: Deduplicate
            key = text[:300]
            if key in seen:
                removed_dup += 1
                continue
            seen.add(key)

            # Bước 2: Rechunk large
            if len(text) > MAX_LEN:
                rechunked_orig += 1
                sub_texts = chunk_sentences(text)
                for i, sub in enumerate(sub_texts):
                    seed = f"rechunk_{r.get('chunk_id', '')}_{i}"
                    new_row = {
                        **r,
                        "chunk_id":    make_chunk_id(seed),
                        "chunk_index": r.get("chunk_index", 0) * 1000 + i,
                        "text":        sub,
                        "embed_text":  sub,
                    }
                    if out:
                        out.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                    added_sub += 1
                kept += len(sub_texts)
            else:
                if out:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1

            if total_in % 10000 == 0:
                print(f"  ...{total_in:,} dòng đã xử lý", flush=True)

    print(f"  ...{total_in:,} dòng đã xử lý (xong)", flush=True)

    if out:
        print("Đang đóng file tạm...", flush=True)
        out.close()

    net = added_sub - rechunked_orig
    total_out = kept
    print(f"\nKết quả:")
    print(f"  Đầu vào:       {total_in:,}")
    print(f"  Duplicate bỏ: -{removed_dup:,}")
    print(f"  Rechunk:       -{rechunked_orig:,} → +{added_sub:,} sub-chunks (net {net:+d})")
    print(f"  Đầu ra:        {total_out:,}")

    if args.dry_run:
        TMP_FILE.unlink(missing_ok=True)
        print("\n[DRY RUN] Không ghi file.")
        return

    print(f"\nBackup → {BACKUP_FILE} (có thể mất vài giây)...", flush=True)
    shutil.copy(INPUT_FILE, BACKUP_FILE)
    TMP_FILE.replace(INPUT_FILE)
    print(f"Done. {total_out:,} chunks → {INPUT_FILE}")


if __name__ == "__main__":
    main()
