"""
Legal CallBot — ASR Worker Configuration
"""
import os
from dataclasses import dataclass


@dataclass
class ASRConfig:
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    port: int = int(os.getenv("ASR_PORT", "50051"))

    # Sherpa-Onnx Settings
    encoder_path: str = os.path.join(os.path.dirname(__file__), "data", "encoder-epoch-31-avg-11-chunk-16-left-128.fp16.onnx")
    decoder_path: str = os.path.join(os.path.dirname(__file__), "data", "decoder-epoch-31-avg-11-chunk-16-left-128.fp16.onnx")
    joiner_path: str = os.path.join(os.path.dirname(__file__), "data", "joiner-epoch-31-avg-11-chunk-16-left-128.fp16.onnx")

    tokens_path: str = os.path.join(os.path.dirname(__file__), "data", "config.json")
    
    sample_rate: int = 16000


config = ASRConfig()

