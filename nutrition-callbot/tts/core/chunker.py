"""
Word-Safe Chunker
Tách text thành chunks phù hợp cho TTS mà không cắt giữa từ.
"""
import re
import logging
from typing import List

logger = logging.getLogger("tts.core.chunker")

# Dấu câu kết thúc chunk
PUNCTUATION_PATTERN = re.compile(r"[.!?;:,]")


def chunk_text(text: str, min_size: int = 40) -> List[str]:
    """
    Tách text thành chunks cho TTS streaming.

    Rules:
      - Text < min_size chars → buffer thêm
      - Gặp dấu câu (.!?;:,) HOẶC đủ min_size chars → tạo chunk
      - Không cắt giữa từ

    Args:
        text: Text đầu vào.
        min_size: Kích thước tối thiểu 1 chunk.

    Returns:
        List of text chunks.

    Examples:
        >>> chunk_text("Xin chào. Tôi là bot.", min_size=10)
        ['Xin chào.', ' Tôi là bot.']
    """
    chunks = []
    buffer = ""

    for char in text:
        buffer += char

        # Gặp dấu câu và buffer đủ dài → tạo chunk
        if PUNCTUATION_PATTERN.match(char) and len(buffer.strip()) >= min_size:
            chunks.append(buffer)
            buffer = ""

    # Phần còn lại
    if buffer.strip():
        chunks.append(buffer)

    logger.debug(f"Chunked text into {len(chunks)} parts")
    return chunks
