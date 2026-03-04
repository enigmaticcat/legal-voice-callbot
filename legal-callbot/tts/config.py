"""
Legal CallBot — TTS Worker Configuration
"""
import os
from dataclasses import dataclass


@dataclass
class TTSConfig:
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    port: int = int(os.getenv("TTS_PORT", "50053"))

    # VieNeu-TTS settings (Bước 4)
    backbone_repo: str = "pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf"
    codec_repo: str = "neuphonic/distill-neucodec"
    streaming_frames_per_chunk: int = 15  # TTFC ~300ms
    streaming_lookforward: int = 5
    sample_rate: int = 24000

    # Word-Safe Chunking
    min_chunk_size: int = 40


config = TTSConfig()
