"""
ASR gRPC Client
Giao tiếp với ASR Worker qua gRPC.
Bước 2 sẽ implement với proto-generated stubs.
"""
import logging

from config import settings

logger = logging.getLogger("gateway.grpc_clients.asr")


class ASRClient:
    """Client gọi ASR Worker qua gRPC."""

    def __init__(self):
        self.address = settings.asr_address
        logger.info(f"ASR client initialized → {self.address}")

    async def transcribe(self, audio_stream):
        """
        Stream audio → ASR → nhận transcript.
        TODO: Implement với gRPC stub ở Bước 2.
        """
        logger.debug(f"Transcribing audio via {self.address}")
        # Dummy
        return {"text": "xin chào", "is_final": True}
