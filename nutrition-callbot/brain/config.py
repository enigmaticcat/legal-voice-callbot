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
    qdrant_path: str = os.getenv("QDRANT_PATH", "") # Thêm biến đọc từ Path để gọi Qdrant Snapshot Local
    qdrant_collection: str = "nutrition_articles"
    min_chunk_size: int = 40


config = BrainConfig()
