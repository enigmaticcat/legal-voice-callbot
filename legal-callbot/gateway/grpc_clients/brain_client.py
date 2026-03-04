"""
Brain gRPC Client
Giao tiếp với Brain Worker qua gRPC.
Bước 2 sẽ implement với proto-generated stubs.
"""
import logging

from config import settings

logger = logging.getLogger("gateway.grpc_clients.brain")


class BrainClient:
    """Client gọi Brain Worker qua gRPC."""

    def __init__(self):
        self.address = settings.brain_address
        logger.info(f"Brain client initialized → {self.address}")

    async def think(self, query: str, session_id: str):
        """
        Gửi câu hỏi → Brain → nhận streaming text.
        TODO: Implement với gRPC stub ở Bước 2.
        """
        logger.debug(f"Thinking via {self.address}: {query[:50]}")
        # Dummy
        return {"text": "Tôi là bot pháp luật", "is_final": True}
