"""
Legal CallBot — Brain Worker Configuration
Tự động load .env file từ project root.
"""
import os
from pathlib import Path
from dataclasses import dataclass

# Load .env file
from dotenv import load_dotenv

# Tìm .env ở thư mục cha (legal-callbot/.env)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)



@dataclass
class BrainConfig:
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    port: int = int(os.getenv("BRAIN_PORT", "50052"))

    # LLM settings (Bước 5)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.3  # Thấp hơn cho pháp lý

    # RAG settings (Bước 8-9)
    qdrant_host: str = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = "phap_dien_khoan" # changed to children collection

    # Chunking
    min_chunk_size: int = 40


config = BrainConfig()
