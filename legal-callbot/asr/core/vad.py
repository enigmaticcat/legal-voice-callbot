"""
Voice Activity Detection (VAD) — Silero VAD Wrapper
Phát hiện khi người dùng ngưng nói (silence) để trigger transcription.
Bước 3 sẽ tích hợp Silero VAD thật.
"""
import logging

logger = logging.getLogger("asr.core.vad")


class VADDetector:
    """
    Voice Activity Detection sử dụng Silero VAD.

    Chạy trên CPU (nhẹ, không tranh VRAM với Whisper).
    Khi phát hiện im lặng > threshold → gửi is_final = True.
    """

    def __init__(self, silence_threshold_ms: int = 300):
        self.silence_threshold_ms = silence_threshold_ms
        logger.info(f"VAD initialized (threshold: {silence_threshold_ms}ms)")

    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        Kiểm tra chunk audio có chứa giọng nói không.
        TODO: Implement Silero VAD ở Bước 3.
        """
        # Dummy — luôn trả True
        return True

    def is_end_of_utterance(self, audio_chunk: bytes) -> bool:
        """
        Kiểm tra đã dứt lời chưa (silence > threshold).
        TODO: Implement ở Bước 3.
        """
        # Dummy
        return False
