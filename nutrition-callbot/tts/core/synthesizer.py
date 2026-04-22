"""
Synthesizer — VieNeu-TTS Inference
Dùng đúng API từ vieneu library (vieneu/core.py):
  tts = Vieneu(backbone_repo, backbone_device, codec_repo, codec_device)
  voice = tts.get_preset_voice()
  for audio_chunk in tts.infer_stream(text, voice=voice): ...  # np.ndarray float32
"""
import logging
import numpy as np
from typing import Iterator

from config import config

logger = logging.getLogger("tts.core.synthesizer")

SAMPLE_RATE = 24_000


class Synthesizer:

    def __init__(self, backbone_repo: str, codec_repo: str):
        self.backbone_repo = backbone_repo
        self.codec_repo = codec_repo
        self.sample_rate = SAMPLE_RATE
        self._tts = None
        self._voice = None
        logger.info(f"Synthesizer init: {backbone_repo}")

    def load_model(self):
        from vieneu import Vieneu

        def _normalize_backbone_device(value: str) -> str:
            v = (value or "").strip().lower()
            if v in {"gpu", "cuda"}:
                return "gpu"
            if v in {"cpu", ""}:
                return "cpu"
            return v

        def _normalize_codec_device(value: str) -> str:
            v = (value or "").strip().lower()
            if v in {"gpu", "cuda"}:
                return "cuda"
            if v in {"cpu", ""}:
                return "cpu"
            return v

        has_cuda = False
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            pass

        if config.backbone_device:
            backbone_device = _normalize_backbone_device(config.backbone_device)
        elif config.tts_device:
            backbone_device = _normalize_backbone_device(config.tts_device)
        else:
            backbone_device = "gpu" if has_cuda else "cpu"

        if config.codec_device:
            codec_device = _normalize_codec_device(config.codec_device)
        elif config.tts_device:
            codec_device = _normalize_codec_device(config.tts_device)
        else:
            codec_device = "cuda" if has_cuda else "cpu"

        logger.info(f"Loading VieNeu-TTS (backbone={backbone_device}, codec={codec_device})...")
        self._tts = Vieneu(
            backbone_repo=self.backbone_repo,
            backbone_device=backbone_device,
            codec_repo=self.codec_repo,
            codec_device=codec_device,
        )
        self._voice = self._tts.get_preset_voice()
        logger.info("VieNeu-TTS loaded.")

    def synthesize_stream(self, text: str, max_chars: int = 256) -> Iterator[bytes]:
        """
        Yields:
            bytes — PCM int16, 24kHz mono
                    mỗi chunk là 1 np.ndarray float32 từ infer_stream() → convert sang int16 bytes
        """
        if self._tts is None:
            self.load_model()

        for audio_chunk in self._tts.infer_stream(
            text=text,
            voice=self._voice,
            max_chars=max_chars,
            temperature=1.0,
            top_k=50,
        ):
            # audio_chunk: np.ndarray float32, shape (N,)
            audio_i16 = (audio_chunk * 32767).clip(-32768, 32767).astype(np.int16)
            yield audio_i16.tobytes()

    def cancel(self, session_id: str):
        logger.info(f"Cancel: {session_id}")

    def close(self):
        if self._tts is not None:
            self._tts.close()
            self._tts = None
            logger.info("VieNeu-TTS closed.")
