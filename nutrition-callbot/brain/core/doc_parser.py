"""Parse uploaded documents (PDF/txt) into text chunks for embedding."""
from __future__ import annotations

import io
import re
from typing import List

MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_CHUNKS = 50
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


class DocParseError(ValueError):
    pass


def _split_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks[:MAX_CHUNKS]


def parse_txt(content: bytes) -> List[str]:
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception as e:
        raise DocParseError(f"Không đọc được file txt: {e}")
    return _split_chunks(text)


def parse_pdf(content: bytes) -> List[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise DocParseError("Thiếu thư viện pypdf. Cài bằng: pip install pypdf")

    try:
        reader = PdfReader(io.BytesIO(content))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n".join(pages_text)
    except Exception as e:
        raise DocParseError(f"Không đọc được file PDF: {e}")

    return _split_chunks(full_text)


def parse_document(filename: str, content: bytes) -> List[str]:
    if len(content) > MAX_FILE_BYTES:
        raise DocParseError(f"File quá lớn (tối đa 5 MB)")

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        chunks = parse_pdf(content)
    elif ext in ("txt", "md"):
        chunks = parse_txt(content)
    else:
        raise DocParseError(f"Định dạng không hỗ trợ: .{ext} (chỉ nhận .pdf và .txt)")

    if not chunks:
        raise DocParseError("Không trích xuất được nội dung từ file")

    return chunks
