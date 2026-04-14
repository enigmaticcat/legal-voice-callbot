"""
Legal CallBot — ASR Worker Configuration
"""
import os
from dataclasses import dataclass


@dataclass
class ASRConfig:
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    port: int = int(os.getenv("ASR_PORT", "50051"))

    encoder_path: str = os.getenv("ASR_ENCODER_PATH", os.path.join(os.path.dirname(__file__), "data", "encoder-epoch-31-avg-11-chunk-16-left-128.fp16.onnx"))
    decoder_path: str = os.getenv("ASR_DECODER_PATH", os.path.join(os.path.dirname(__file__), "data", "decoder-epoch-31-avg-11-chunk-16-left-128.fp16.onnx"))
    joiner_path: str = os.getenv("ASR_JOINER_PATH", os.path.join(os.path.dirname(__file__), "data", "joiner-epoch-31-avg-11-chunk-16-left-128.fp16.onnx"))
    tokens_path: str = os.getenv("ASR_TOKENS_PATH", os.path.join(os.path.dirname(__file__), "data", "config.json"))
    
    sample_rate: int = 16000
    provider: str = os.getenv("ASR_PROVIDER", "cpu")  # "cpu" hoặc "cuda"


config = ASRConfig()

