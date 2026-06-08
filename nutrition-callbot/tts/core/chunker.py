"""
Word-Safe Chunker
Tách text thành chunks phù hợp cho TTS mà không cắt giữa từ.
"""
import re
import logging
from typing import List

logger = logging.getLogger("tts.core.chunker")

# Chỉ cắt tại dấu kết thúc câu hoàn chỉnh.
# Không cắt tại , ; : vì TTS model sẽ tự thêm prosody kết thúc sau mỗi chunk —
# nếu chunk kết thúc giữa mệnh đề (dấu phẩy), nghe như ngắt câu tùy tiện.
_SENT_END = re.compile(r"[.!?]")


def chunk_text(text: str, min_size: int = 40) -> List[str]:
    """
    Tách text thành chunks cho TTS streaming.

    Rules:
      - Gặp dấu kết thúc câu (.!?) VÀ buffer >= min_size → tạo chunk
      - Không cắt tại dấu phẩy/chấm phẩy để tránh prosody ngắt giữa câu
      - Phần còn lại (không có dấu câu hoặc chưa đủ min_size) → chunk cuối

    Examples:
        >>> chunk_text("Xin chào. Tôi là bot.", min_size=10)
        ['Xin chào.', ' Tôi là bot.']
    """
    chunks = []
    buffer = ""

    for char in text:
        buffer += char

        if _SENT_END.match(char) and len(buffer.strip()) >= min_size:
            chunks.append(buffer)
            buffer = ""

    if buffer.strip():
        chunks.append(buffer)

    logger.debug(f"Chunked text into {len(chunks)} parts")
    return chunks
