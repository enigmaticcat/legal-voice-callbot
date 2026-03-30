"""
TTS gRPC Client
Giao tiếp với TTS Worker qua gRPC.
Bước 2 sẽ implement với proto-generated stubs.
"""
import logging

from config import settings

logger = logging.getLogger("gateway.grpc_clients.tts")


class TTSClient:
    """Client gọi TTS Worker qua gRPC."""

    def __init__(self):
        self.address = settings.tts_address
        logger.info(f"TTS client initialized → {self.address}")

    async def speak(self, text_stream):
        """
        Stream text → TTS → nhận streaming audio.
        TODO: Implement với gRPC stub ở Bước 2.
        """
        logger.debug(f"Speaking via {self.address}")
        # Dummy — trả silence
        return b""

    async def cancel(self, session_id: str):
        """
        Cancel TTS đang chạy (cho barge-in).
        TODO: Implement ở Bước 2.
        """
        logger.info(f"Cancelling TTS for session {session_id}")
