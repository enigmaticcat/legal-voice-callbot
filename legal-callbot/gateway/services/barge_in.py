"""
Barge-in Service
Xử lý khi người dùng cắt lời bot (nói xen vào lúc TTS đang phát).
Bước 6 sẽ implement logic cancel TTS stream.
"""
import logging

logger = logging.getLogger("gateway.services.barge_in")


class BargeInHandler:
    """
    Xử lý barge-in (cắt lời bot).

    Khi VAD detect client nói trong lúc TTS đang phát:
      1. Cancel TTS stream (gRPC Cancel)
      2. Ngưng gửi audio cho client
      3. Bắt đầu ASR stream mới
    """

    async def handle(self, session_id: str):
        """
        Xử lý sự kiện barge-in.
        TODO: Implement ở Bước 6.
        """
        logger.info(f"[{session_id}] Barge-in detected — cancelling TTS")
        # Placeholder
        pass
