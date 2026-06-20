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

    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://llm:8000/v1")
    llm_model: str = os.getenv("LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    llm_api_key: str = os.getenv("LLM_API_KEY", "local")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379")
    retrieval_cache_enabled: bool = os.getenv("RETRIEVAL_CACHE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    retrieval_cache_required: bool = os.getenv("RETRIEVAL_CACHE_REQUIRED", "false").strip().lower() in {"1", "true", "yes", "on"}
    retrieval_cache_ttl_seconds: int = int(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS", "86400"))
    corpus_version: str = os.getenv("CORPUS_VERSION", "nutrition-v1")
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
    min_chunk_size: int = int(os.getenv("BRAIN_MIN_CHUNK_SIZE", "40"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding")
    reranker_model: str = os.getenv("RERANKER_MODEL", "thanhtantran/Vietnamese_Reranker")
    rag_fetch_k: int = int(os.getenv("RAG_FETCH_K", "15"))
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    rag_use_hyde: bool = os.getenv("RAG_USE_HYDE", "false").strip().lower() in {"1", "true", "yes", "on"}
    llm_max_output_tokens: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1500"))
    tts_url: str = os.getenv("TTS_URL", "http://localhost:50053")


config = BrainConfig()
