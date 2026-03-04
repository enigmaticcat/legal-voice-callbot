"""
Legal CallBot — ASR Worker Configuration
"""
import os
from dataclasses import dataclass


@dataclass
class ASRConfig:
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    port: int = int(os.getenv("ASR_PORT", "50051"))

    # Whisper settings (Bước 3)
    whisper_model: str = os.getenv("WHISPER_MODEL", "large-v3")
    whisper_language: str = "vi"
    whisper_beam_size: int = 1

    # VAD settings (Bước 3)
    vad_silence_threshold_ms: int = 300


config = ASRConfig()
