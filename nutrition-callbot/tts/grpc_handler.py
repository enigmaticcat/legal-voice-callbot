"""
TTS Service Handler
Nhận text → word-safe chunking → synthesize_stream() → yield PCM bytes
"""
import asyncio
import logging

from core.synthesizer import Synthesizer
from core.chunker import chunk_text

logger = logging.getLogger("tts.grpc_handler")


class TTSServiceHandler:

    def __init__(self, synthesizer: Synthesizer):
        self.synthesizer = synthesizer

    async def speak(self, text: str):
        """
        Text → PCM int16 bytes streaming.
        Mỗi text chunk → chạy infer_stream trong thread → yield PCM bytes.
        """
        logger.info(f"Speak: {text[:60]}...")
        chunks = chunk_text(text, min_size=self.synthesizer.sample_rate // 100)

        for chunk in chunks:
            # infer_stream là sync iterator → chạy trong thread pool
            pcm_frames = await asyncio.to_thread(
                lambda c=chunk: list(self.synthesizer.synthesize_stream(c))
            )
            for frame in pcm_frames:
                yield frame

    async def cancel(self, session_id: str):
        self.synthesizer.cancel(session_id)
