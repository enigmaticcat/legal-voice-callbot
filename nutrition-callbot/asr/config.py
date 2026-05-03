"""
ASR Service Configuration
"""
import os
from dataclasses import dataclass

_data_dir = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class ASRConfig:
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    port: int = int(os.getenv("ASR_PORT", "50051"))

    encoder_path: str = os.getenv(
        "ASR_ENCODER_PATH",
        os.path.join(_data_dir, "encoder-epoch-20-avg-10.onnx"),
    )
    decoder_path: str = os.getenv(
        "ASR_DECODER_PATH",
        os.path.join(_data_dir, "decoder-epoch-20-avg-10.onnx"),
    )
    joiner_path: str = os.getenv(
        "ASR_JOINER_PATH",
        os.path.join(_data_dir, "joiner-epoch-20-avg-10.onnx"),
    )
    tokens_path: str = os.getenv(
        "ASR_TOKENS_PATH",
        os.path.join(_data_dir, "config.json"),
    )

    sample_rate: int = 16_000
    num_threads: int = int(os.getenv("ASR_NUM_THREADS", "4"))
    provider: str = os.getenv("ASR_PROVIDER", "cpu")   # "cpu" hoặc "cuda"
    require_cuda: bool = os.getenv("ASR_REQUIRE_CUDA", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }


config = ASRConfig()
