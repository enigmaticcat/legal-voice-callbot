"""
Legal CallBot — Gateway Configuration
Tập trung quản lý environment variables và settings.
"""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # --- Server ---
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # --- gRPC Worker Hosts ---
    asr_host: str = os.getenv("ASR_HOST", "asr")
    asr_port: int = int(os.getenv("ASR_PORT", "50051"))
    brain_host: str = os.getenv("BRAIN_HOST", "brain")
    brain_port: int = int(os.getenv("BRAIN_PORT", "50052"))
    tts_host: str = os.getenv("TTS_HOST", "tts")
    tts_port: int = int(os.getenv("TTS_PORT", "50053"))

    # --- Session ---
    session_timeout_seconds: int = int(os.getenv("SESSION_TIMEOUT", "300"))

    @property
    def asr_address(self) -> str:
        return f"{self.asr_host}:{self.asr_port}"

    @property
    def brain_address(self) -> str:
        return f"{self.brain_host}:{self.brain_port}"

    @property
    def tts_address(self) -> str:
        return f"{self.tts_host}:{self.tts_port}"

    @property
    def asr_http_url(self) -> str:
        return f"http://{self.asr_host}:{self.asr_port}"

    @property
    def brain_http_url(self) -> str:
        return f"http://{self.brain_host}:{self.brain_port}"

    @property
    def tts_http_url(self) -> str:
        return f"http://{self.tts_host}:{self.tts_port}"


settings = Settings()
