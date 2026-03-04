"""
Orchestrator Service
Điều phối pipeline: ASR → Brain → TTS theo kiểu stream-to-stream.
Bước 6 sẽ implement đầy đủ logic pipeline chồng lấp.
"""
import logging

logger = logging.getLogger("gateway.services.orchestrator")


class Orchestrator:
    """
    Điều phối luồng đàm thoại giữa các AI workers.

    Pipeline:
      1. Nhận audio từ client
      2. Stream audio → ASR → nhận transcript
      3. Gửi transcript → Brain → nhận text response (streaming)
      4. Gửi text → TTS → nhận audio response (streaming)
      5. Stream audio về client

    LLM và TTS chạy pipeline chồng lấp:
      TTS bắt đầu ngay khi Brain gửi text chunk đầu tiên.
    """

    def __init__(self, asr_client, brain_client, tts_client):
        self.asr = asr_client
        self.brain = brain_client
        self.tts = tts_client

    async def process_audio(self, session_id: str, audio_data: bytes):
        """
        Xử lý audio đầu vào qua toàn bộ pipeline.
        TODO: Implement ở Bước 6.
        """
        logger.info(f"[{session_id}] Processing audio ({len(audio_data)} bytes)")
        # Placeholder — sẽ implement pipeline thật
        return {"text": "Dummy response", "audio": b""}
