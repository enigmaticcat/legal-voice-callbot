import re
import logging
from typing import AsyncGenerator

logger = logging.getLogger("brain.core.chunker")

PUNCTUATION = re.compile(r"[.!?;:,।।\n]")
MIN_CHUNK_SIZE = 40  


async def chunk_llm_stream(text_stream, min_size: int = MIN_CHUNK_SIZE) -> AsyncGenerator[str, None]:
    buffer = ""

    async for chunk in text_stream:
        text = chunk.get("text", "") if isinstance(chunk, dict) else chunk
        buffer += text

        while len(buffer) >= min_size:
            match = None
            for m in PUNCTUATION.finditer(buffer):
                if m.start() >= min_size // 2:
                    match = m

            if match:
                cut_pos = match.end()
                yield buffer[:cut_pos].strip()
                buffer = buffer[cut_pos:].lstrip()
            elif len(buffer) > min_size * 2:
                space_pos = buffer.rfind(" ", min_size // 2, min_size * 2)
                if space_pos > 0:
                    yield buffer[:space_pos].strip()
                    buffer = buffer[space_pos:].lstrip()
                else:
                    break  
            else:
                break  

    if buffer.strip():
        yield buffer.strip()
