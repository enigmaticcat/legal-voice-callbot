"""
gRPC Handler — StreamTranscribe() implementation
Bước 2 sẽ chuyển sang gRPC thật. Hiện tại dùng HTTP dummy.
"""
import logging

from core.transcriber import Transcriber
from core.vad import VADDetector

logger = logging.getLogger("asr.grpc_handler")


class ASRServiceHandler:
    """
    Handler cho ASR gRPC Service.

    StreamTranscribe():
      1. Nhận stream AudioChunk (PCM 16kHz)
      2. Buffer audio
      3. VAD detect pause → Whisper transcribe
      4. Stream TranscriptResult ra
    """

    def __init__(self):
        self.transcriber = Transcriber()
        self.vad = VADDetector()

    async def stream_transcribe(self, audio_stream):
        """
        Xử lý audio stream → transcript stream.
        TODO: Implement gRPC streaming ở Bước 2-3.
        """
        logger.info("StreamTranscribe called (dummy mode)")
        result = self.transcriber.transcribe(b"")
        return {
            "text": result["text"],
            "is_final": True,
            "confidence": result["confidence"],
        }
