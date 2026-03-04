"""
Audio Utilities
Xử lý PCM: resampling, normalization, format conversion.
"""
import struct
import logging

logger = logging.getLogger("asr.core.audio_utils")


def pcm_to_float(pcm_bytes: bytes) -> list:
    """Chuyển PCM 16-bit signed → float [-1.0, 1.0]."""
    samples = struct.unpack(f"<{len(pcm_bytes) // 2}h", pcm_bytes)
    return [s / 32768.0 for s in samples]


def validate_audio(audio_bytes: bytes, expected_sample_rate: int = 16000) -> bool:
    """Kiểm tra audio data hợp lệ."""
    if len(audio_bytes) < 2:
        logger.warning("Audio too short")
        return False
    if len(audio_bytes) % 2 != 0:
        logger.warning("Audio length not even (invalid 16-bit PCM)")
        return False
    return True
