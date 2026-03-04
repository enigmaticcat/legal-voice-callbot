"""
Word-Safe Text Chunker — cho LLM → TTS pipeline
Buffer text từ LLM stream → tách thành chunks phù hợp cho TTS.
Không cắt giữa từ, ưu tiên dấu câu.
"""
import re
import logging
from typing import Generator

logger = logging.getLogger("brain.core.chunker")

# Dấu câu kết thúc chunk tự nhiên
PUNCTUATION = re.compile(r"[.!?;:,।।\n]")
MIN_CHUNK_SIZE = 40  # Chars tối thiểu trước khi tìm điểm cắt


async def chunk_llm_stream(text_stream, min_size: int = MIN_CHUNK_SIZE) -> Generator[str, None, None]:
    """
    Nhận async stream of text chunks từ LLM, buffer lại và yield word-safe chunks.

    Rules:
      1. Buffer cho đến khi ≥ min_size chars
      2. Tìm dấu câu cuối cùng trong buffer → cắt tại đó
      3. Nếu không có dấu câu và buffer > 2 × min_size → cắt tại khoảng trắng cuối
      4. Flush buffer còn lại khi stream kết thúc

    Args:
        text_stream: Async Iterable yield chunks từ LLM
        min_size: Kích thước tối thiểu trước khi yield

    Yields:
        str: Word-safe text chunks
    """
    buffer = ""

    async for chunk in text_stream:
        # Lấy field "text" tuỳ theo định dạng của llm generate_stream
        text = chunk.get("text", "") if isinstance(chunk, dict) else chunk
        buffer += text

        while len(buffer) >= min_size:
            # Tìm dấu câu cuối cùng trong buffer
            match = None
            for m in PUNCTUATION.finditer(buffer):
                if m.start() >= min_size // 2:  # Ít nhất nửa min_size
                    match = m

            if match:
                cut_pos = match.end()
                yield buffer[:cut_pos].strip()
                buffer = buffer[cut_pos:].lstrip()
            elif len(buffer) > min_size * 2:
                # Quá dài, cắt tại khoảng trắng cuối
                space_pos = buffer.rfind(" ", min_size // 2, min_size * 2)
                if space_pos > 0:
                    yield buffer[:space_pos].strip()
                    buffer = buffer[space_pos:].lstrip()
                else:
                    break  # Chờ thêm text
            else:
                break  # Chờ thêm text

    # Flush buffer còn lại
    if buffer.strip():
        yield buffer.strip()
