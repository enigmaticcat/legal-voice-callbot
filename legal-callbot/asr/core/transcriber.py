"""
Transcriber — Faster-Whisper Inference Wrapper
Nhận audio PCM → trả text tiếng Việt.
Bước 3 sẽ tích hợp faster-whisper thật.
"""
import logging

logger = logging.getLogger("asr.core.transcriber")


class Transcriber:
    """
    Faster-Whisper ASR engine.

    Config tối ưu latency:
      - beam_size = 1 (Greedy decoding)
      - language = "vi" (skip language detection)
      - vad_filter = False (dùng Silero VAD riêng)
    """

    def __init__(self, model_name: str = "large-v3", language: str = "vi"):
        self.model_name = model_name
        self.language = language
        self._model = None  # Lazy load
        logger.info(f"Transcriber initialized (model: {model_name}, lang: {language})")

    def load_model(self):
        """
        Load Whisper model vào GPU.
        TODO: Implement ở Bước 3.
        """
        logger.info(f"Loading Whisper model: {self.model_name}...")
        # Placeholder

    def transcribe(self, audio_pcm: bytes, sample_rate: int = 16000) -> dict:
        """
        Transcribe audio PCM → text.
        TODO: Implement ở Bước 3.

        Returns:
            {"text": str, "confidence": float}
        """
        logger.debug(f"Transcribing {len(audio_pcm)} bytes audio")
        # Dummy response
        return {
            "text": "xin chào",
            "confidence": 0.95,
        }
