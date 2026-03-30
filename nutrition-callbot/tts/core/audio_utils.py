"""
Audio Utilities — TTS
WAV encoding, PCM conversion, format helpers.
"""
import struct
import logging

logger = logging.getLogger("tts.core.audio_utils")


def generate_silence_wav(duration_ms: int = 500, sample_rate: int = 24000) -> bytes:
    """Sinh file WAV silence (all zeros)."""
    num_samples = int(sample_rate * duration_ms / 1000)
    data_size = num_samples * 2  # 16-bit = 2 bytes/sample
    file_size = 36 + data_size

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', file_size, b'WAVE',
        b'fmt ', 16,
        1,                  # PCM
        1,                  # mono
        sample_rate,
        sample_rate * 2,    # byte rate
        2,                  # block align
        16,                 # bits/sample
        b'data', data_size,
    )
    return header + (b'\x00' * data_size)


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap raw PCM data trong WAV header."""
    data_size = len(pcm_data)
    file_size = 36 + data_size

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', file_size, b'WAVE',
        b'fmt ', 16,
        1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', data_size,
    )
    return header + pcm_data
