"""
Nutrition CallBot — Brain Worker Configuration
Tự động load .env file từ project root.
"""
import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


def _resolve_env_path(raw_value: str) -> str:
    value = (raw_value or "").strip()
    if not value:
        return ""
    p = Path(value)
    if p.is_absolute():
        return str(p)
    # Resolve relative paths from nutrition-callbot/.env location for portable deploys.
    return str((_env_path.parent / p).resolve())



@dataclass
class BrainConfig:
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    port: int = int(os.getenv("BRAIN_PORT", "50052"))

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.3  
    qdrant_host: str = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_path: str = _resolve_env_path(os.getenv("QDRANT_PATH", ""))
    qdrant_snapshot_path: str = _resolve_env_path(os.getenv("QDRANT_SNAPSHOT_PATH", ""))
    qdrant_snapshot_force_restore: bool = os.getenv("QDRANT_SNAPSHOT_FORCE_RESTORE", "false").strip().lower() in {"1", "true", "yes", "on"}
    qdrant_snapshot_timeout_s: int = int(os.getenv("QDRANT_SNAPSHOT_TIMEOUT_S", "600"))
    qdrant_snapshot_priority: str = os.getenv("QDRANT_SNAPSHOT_PRIORITY", "snapshot")
    qdrant_collection: str = "nutrition_articles"
    min_chunk_size: int = 40
    tts_url: str = os.getenv("TTS_URL", "http://localhost:50053")


config = BrainConfig()
