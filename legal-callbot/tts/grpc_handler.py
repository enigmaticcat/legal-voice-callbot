"""
gRPC Handler — Speak() + Cancel() implementation
Bước 2 sẽ chuyển sang gRPC thật. Hiện tại dùng HTTP dummy.
"""
import logging

from core.synthesizer import Synthesizer
from core.chunker import chunk_text

logger = logging.getLogger("tts.grpc_handler")


class TTSServiceHandler:
    """
    Handler cho TTS gRPC Service.

    Speak():
      Nhận stream TextChunk → word-safe chunking → TTS stream → AudioChunk ra.

    Cancel():
      Dừng inference ngay lập tức, giải phóng GPU.
    """

    def __init__(self, synthesizer: Synthesizer):
        self.synthesizer = synthesizer

    async def speak(self, text: str):
        """
        Text → Audio streaming.
        TODO: Implement gRPC streaming ở Bước 2-4.
        """
        logger.info(f"Speak: {text[:50]}...")
        chunks = chunk_text(text)
        for chunk in chunks:
            for audio_frame in self.synthesizer.synthesize_stream(chunk):
                yield audio_frame

    async def cancel(self, session_id: str):
        """
        Cancel TTS cho session.
        TODO: Implement ở Bước 2.
        """
        self.synthesizer.cancel(session_id)
