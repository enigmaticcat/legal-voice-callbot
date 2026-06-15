"""
Nutrition CallBot — TTS Worker Configuration
"""
import os
from dataclasses import dataclass


@dataclass
class TTSConfig:
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    port: int = int(os.getenv("TTS_PORT", "50053"))

    # Backend:
    # - local: bundled VieNeu-TTS GGUF/PyTorch in this service
    # - lmdeploy: local codec + remote LMDeploy OpenAI-compatible server
    backend: str = os.getenv("TTS_BACKEND", "local").strip().lower()
    lmdeploy_api_base: str = os.getenv("TTS_LMDEPLOY_API_BASE", "http://tts-lmdeploy:23333/v1").rstrip("/")
    lmdeploy_model: str = os.getenv("TTS_LMDEPLOY_MODEL", "pnnbao-ump/VieNeu-TTS")

    # VieNeu-TTS settings (Bước 4)
    backbone_repo: str = os.getenv("TTS_BACKBONE_REPO", "pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf")
    codec_repo: str = "neuphonic/distill-neucodec"
    streaming_frames_per_chunk: int = 15  # TTFC ~300ms
    streaming_lookforward: int = 5
    sample_rate: int = 24000

    # Word-Safe Chunking
    min_chunk_size: int = 40

    # Device control (optional): if empty, auto-detect from torch.cuda.is_available()
    tts_device: str = os.getenv("TTS_DEVICE", "").strip().lower()
    backbone_device: str = os.getenv("TTS_BACKBONE_DEVICE", "").strip().lower()
    codec_device: str = os.getenv("TTS_CODEC_DEVICE", "").strip().lower()
    require_cuda: bool = os.getenv("TTS_REQUIRE_CUDA", "false").strip().lower() in {"1", "true", "yes", "on"}


config = TTSConfig()
