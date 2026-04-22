"""
TTS Service Handler
Nhận text → word-safe chunking → synthesize_stream() → yield PCM bytes
"""
import asyncio
import logging
import threading

from core.synthesizer import Synthesizer
from core.chunker import chunk_text

logger = logging.getLogger("tts.grpc_handler")

_CHUNK_MIN_CHARS = 60


class TTSServiceHandler:

    def __init__(self, synthesizer: Synthesizer):
        self.synthesizer = synthesizer

    async def speak(self, text: str):
        """
        Text → PCM int16 bytes streaming.
        Yield mỗi frame ngay khi infer_stream sản xuất ra, không buffer toàn chunk.
        """
        logger.info(f"Speak: {text[:60]}...")
        chunks = chunk_text(text, min_size=_CHUNK_MIN_CHARS)
        loop = asyncio.get_running_loop()

        for chunk in chunks:
            q: asyncio.Queue = asyncio.Queue()

            def _worker(c=chunk):
                try:
                    for frame in self.synthesizer.synthesize_stream(c):
                        loop.call_soon_threadsafe(q.put_nowait, frame)
                except Exception as e:
                    logger.error(f"TTS synthesis error: {e}")
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)

            threading.Thread(target=_worker, daemon=True).start()

            while True:
                frame = await q.get()
                if frame is None:
                    break
                yield frame

    async def cancel(self, session_id: str):
        self.synthesizer.cancel(session_id)
