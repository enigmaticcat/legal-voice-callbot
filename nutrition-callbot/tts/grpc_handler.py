"""
TTS Service Handler
Nhận text → word-safe chunking → synthesize_stream() → yield PCM bytes
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import threading

from core.synthesizer import Synthesizer

logger = logging.getLogger("tts.grpc_handler")

_CHUNK_MAX_CHARS = 256
_NEWLINE_RE = re.compile(r"\n+")

# Giới hạn số synthesis chạy đồng thời. vieneu.infer_stream() không thread-safe
# nên serialize hoàn toàn (=1) là an toàn nhất; tăng lên nếu thư viện hỗ trợ.
_MAX_CONCURRENT = int(os.getenv("TTS_MAX_CONCURRENT", "1"))


class TTSServiceHandler:

    def __init__(self, synthesizer: Synthesizer):
        self.synthesizer = synthesizer
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

    async def speak(self, text: str, session_id: str | None = None):
        """
        Text → PCM int16 bytes streaming.
        Yield mỗi frame ngay khi infer_stream sản xuất ra, không buffer toàn chunk.
        """
        if not text or not text.strip():
            raise ValueError("TTS input text is empty")

        # LLM đôi khi sinh \n giữa đoạn — thay bằng dấu cách để TTS đọc liền mạch
        text = _NEWLINE_RE.sub(" ", text).strip()

        logger.info(f"Speak: {text[:60]}...")
        chunks = [text]

        loop = asyncio.get_running_loop()

        for chunk in chunks:
            q: asyncio.Queue = asyncio.Queue()

            def _worker(c=chunk, s=session_id):
                try:
                    for frame in self.synthesizer.synthesize_stream(c, session_id=s, max_chars=_CHUNK_MAX_CHARS):
                        loop.call_soon_threadsafe(q.put_nowait, frame)
                except Exception as e:
                    logger.exception("TTS synthesis error")
                    loop.call_soon_threadsafe(q.put_nowait, e)
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)

            async with self._semaphore:
                threading.Thread(target=_worker, daemon=True).start()

                while True:
                    frame = await q.get()
                    if frame is None:
                        break
                    if isinstance(frame, Exception):
                        raise RuntimeError("TTS synthesis failed") from frame
                    yield frame

    async def cancel(self, session_id: str):
        self.synthesizer.cancel(session_id)
