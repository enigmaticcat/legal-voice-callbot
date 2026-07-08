"""Parse uploaded documents (PDF/txt/md) into text chunks using LangChain."""
from __future__ import annotations

import io
import tempfile
import os
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_CHUNKS = 50
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""],
)


class DocParseError(ValueError):
    pass


def parse_txt(content: bytes) -> List[str]:
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception as e:
        raise DocParseError(f"Không đọc được file txt: {e}")
    docs = _splitter.create_documents([text])
    return [d.page_content for d in docs][:MAX_CHUNKS]


def parse_pdf(content: bytes) -> List[str]:
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        raise DocParseError("Thiếu thư viện langchain-community. Cài bằng: pip install langchain-community pypdf")

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
    except Exception as e:
        raise DocParseError(f"Không đọc được file PDF: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    docs = _splitter.split_documents(pages)
    return [d.page_content for d in docs if d.page_content.strip()][:MAX_CHUNKS]


def parse_document(filename: str, content: bytes) -> List[str]:
    if len(content) > MAX_FILE_BYTES:
        raise DocParseError("File quá lớn (tối đa 5 MB)")

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        chunks = parse_pdf(content)
    elif ext in ("txt", "md"):
        chunks = parse_txt(content)
    else:
        raise DocParseError(f"Định dạng không hỗ trợ: .{ext} (chỉ nhận .pdf, .txt, .md)")

    if not chunks:
        raise DocParseError("Không trích xuất được nội dung từ file")

    return chunks
