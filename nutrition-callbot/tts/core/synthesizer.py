"""
Synthesizer — VieNeu-TTS Inference Wrapper
Nhận text → sinh audio streaming.
Bước 4 sẽ tích hợp VieNeu-TTS GGUF thật.
"""
import logging
from typing import Iterator

logger = logging.getLogger("tts.core.synthesizer")


class Synthesizer:
    """
    VieNeu-TTS Streaming Engine.

    Config:
      - backbone: VieNeu-TTS-0.3B GGUF (GPU)
      - codec: NeuCodec Distilled (GPU)
      - streaming_frames_per_chunk: 15 → TTFC ~300ms
    """

    def __init__(self, backbone_repo: str, codec_repo: str):
        self.backbone_repo = backbone_repo
        self.codec_repo = codec_repo
        self._model = None  # Lazy load
        logger.info(f"Synthesizer initialized (backbone: {backbone_repo})")

    def load_model(self):
        """
        Load TTS model vào GPU.
        TODO: Implement ở Bước 4.
        """
        logger.info("Loading VieNeu-TTS model...")
        # Placeholder

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """
        Sinh audio streaming từ text.
        TODO: Implement ở Bước 4.

        Yields:
            bytes — PCM audio chunks (24kHz, 16-bit mono)
        """
        logger.debug(f"Synthesizing: {text[:50]}...")
        # Dummy — trả silence
        yield b"\x00" * 4800  # 100ms silence at 24kHz

    def cancel(self, session_id: str):
        """
        Hủy inference đang chạy (cho barge-in).
        TODO: Implement ở Bước 4.
        """
        logger.info(f"Cancelling synthesis for session {session_id}")
