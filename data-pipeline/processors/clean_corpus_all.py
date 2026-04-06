"""
clean_corpus_all.py
===================
Làm sạch corpus_all.jsonl trước khi chunk.

Các loại nhiễu được xử lý:
  1. Unicode noise     — \xa0 → space, \u200b/\u200c/\uFEFF → xóa
  2. Separator lines   — dòng chỉ có dấu gạch ngang (---...)
  3. URL-only lines    — dòng chỉ chứa https://...
  4. NỘI DUNG TOC      — block "NỘI DUNG\n1. ...\n2. ..." ở đầu bài
  5. Reference section — mọi nội dung SAU dấu hiệu tài liệu tham khảo
  6. Author bylines    — dòng đầu/cuối kiểu "Bài viết được cố vấn bởi ThS.BS ..."
  7. Nav/footer lines  — Đăng nhập, Trang chủ, Copyright, ...

Chạy từ thư mục gốc project:
  python data-pipeline/processors/clean_corpus_all.py
"""

import json
import re
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent.parent
INPUT_FILE  = ROOT / "corpus_all.jsonl"
OUTPUT_FILE = ROOT / "corpus_all_clean.jsonl"
MIN_CONTENT = 200   # bỏ bài quá ngắn sau khi clean


# ── Pattern definitions ───────────────────────────────────────────────────────

# Điểm cắt: mọi nội dung từ dòng này trở đi là tài liệu tham khảo
REF_CUTOFF = re.compile(
    r"\n(TÀI LIỆU THAM KHẢO|NGUỒN THAM KHẢO|Nguồn tham khảo"
    r"|Tài liệu tham khảo|REFERENCES?|References?)\b",
    re.IGNORECASE,
)

# Dòng chỉ là separator
SEPARATOR_LINE = re.compile(r"^-{5,}$")

# Dòng chỉ là URL
URL_ONLY_LINE = re.compile(r"^https?://\S+$")

# Byline author ở bất kỳ dòng nào
AUTHOR_LINE = re.compile(
    r"^(Bài viết được (cố vấn|tư vấn|kiểm duyệt)|"
    r"Nội dung video được tư vấn|"
    r"Tác giả\s*:|"
    r"Nguồn\s*:|"
    r"Theo\s+\w+\.{0,3}com|"
    r"Nguồn:\s*https?)",
    re.IGNORECASE,
)

# Nav/footer cứng
NAV_LINE = re.compile(
    r"^(Đăng nhập|Đăng ký|Trang chủ|Liên hệ|Hotline|Copyright|"
    r"Facebook|Zalo|Youtube|Instagram|Twitter|Sitemap)\b",
    re.IGNORECASE,
)


def remove_toc_block(text: str) -> str:
    """
    Xóa block NỘI DUNG / MỤC LỤC ở đầu bài.
    Pattern: "NỘI DUNG\n1. ...\n2. ...\n..." (chỉ danh sách mục, không có nội dung)
    Dừng khi gặp dòng không phải số thứ tự hoặc khoảng trắng.
    """
    toc_start = re.search(r"\nNỘI DUNG\n", text)
    if not toc_start:
        return text

    pos = toc_start.end()
    # Đọc các dòng tiếp theo, chỉ drop nếu là list (n. text)
    lines = text[pos:].split("\n")
    end_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^\d+\.", stripped) or stripped == "":
            end_idx = i + 1
        else:
            break

    # Xóa block NỘI DUNG + các dòng list
    before = text[: toc_start.start()]
    after  = "\n".join(lines[end_idx:])
    return before + ("\n" if after else "") + after


def clean_content(text: str) -> str:
    # 1. Unicode normalization
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\uFEFF", "")

    # 2. Cắt tại reference section
    m = REF_CUTOFF.search(text)
    if m:
        text = text[: m.start()]

    # 3. Xóa NỘI DUNG TOC
    text = remove_toc_block(text)

    # 4. Lọc từng dòng
    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if SEPARATOR_LINE.match(stripped):
            continue
        if URL_ONLY_LINE.match(stripped):
            continue
        if AUTHOR_LINE.match(stripped):
            continue
        if NAV_LINE.match(stripped):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 5. Chuẩn hóa khoảng trắng thừa
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def main():
    docs_in = docs_out = docs_dropped = 0
    changed = 0

    with open(INPUT_FILE, encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            docs_in += 1
            doc = json.loads(line)

            original = doc.get("content", "")
            cleaned  = clean_content(original)

            if len(cleaned) < MIN_CONTENT:
                docs_dropped += 1
                continue

            if cleaned != original:
                changed += 1

            doc["content"] = cleaned
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            docs_out += 1

    print(f"Input : {docs_in:,} docs")
    print(f"Output: {docs_out:,} docs  (dropped {docs_dropped} quá ngắn)")
    print(f"Changed: {changed:,} docs ({changed/docs_in*100:.1f}%)")
    print(f"→ {OUTPUT_FILE}")

    # Stats per source
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        out_docs = [json.loads(l) for l in f if l.strip()]
    from collections import Counter
    cnt = Counter(d["source"] for d in out_docs)
    print("\nDocs per source:")
    for src, n in cnt.most_common():
        print(f"  {src}: {n:,}")


if __name__ == "__main__":
    main()
