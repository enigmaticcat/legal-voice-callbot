"""
Transcriber — Sherpa-Onnx Offline Transducer
Model: ZipFormer-RNNT (hynt/Zipformer-30M-RNNT-6000h)
Nhận toàn bộ audio PCM một lần → trả transcript tiếng Việt.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("asr.core.transcriber")

SAMPLE_RATE = 16_000


@dataclass
class _BufferedOfflineStream:
    """Compatibility stream for callers that still use the old streaming API."""

    sample_rate: int = SAMPLE_RATE
    audio: bytearray = field(default_factory=bytearray)
    started_at: float = field(default_factory=time.time)
    first_text_at: Optional[float] = None
    last_text: str = ""


class Transcriber:

    def __init__(self):
        from config import config
        self.config = config
        self._recognizer = None
        self._load()

    def _load(self):
        import sherpa_onnx

        provider = self.config.provider.lower().strip()
        if provider == "gpu":
            provider = "cuda"

        logger.info(
            "Loading Sherpa-Onnx OfflineRecognizer (provider=%s, threads=%d)...",
            provider,
            self.config.num_threads,
        )

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=self.config.encoder_path,
            decoder=self.config.decoder_path,
            joiner=self.config.joiner_path,
            tokens=self.config.tokens_path,
            num_threads=self.config.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            provider=provider,
            debug=False,
        )
        logger.info("Sherpa-Onnx OfflineRecognizer loaded (provider=%s).", provider)

    def transcribe(self, audio_pcm: bytes, sample_rate: int = SAMPLE_RATE) -> dict:
        """Transcribe a complete audio clip (PCM int16, 16kHz mono)."""
        if not audio_pcm:
            return {"text": "", "confidence": 0.0}

        samples = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        # Pad 0.5s silence so the encoder flushes the last spoken frames
        samples = np.concatenate([samples, np.zeros(sample_rate // 2, dtype=np.float32)])

        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._recognizer.decode_streams([stream])

        text = stream.result.text.strip()
        logger.info("Transcribed (%d bytes → %d chars): %s", len(audio_pcm), len(text), text[:80])
        return {"text": text, "confidence": 0.95}

    def create_stream(self) -> _BufferedOfflineStream:
        """
        Return a buffered stream compatible with older call sites.

        The configured model is an offline recognizer, so this object accumulates
        PCM chunks and decodes the accumulated utterance on each accept call.
        Finalization is handled by callers when the stream ends.
        """
        return _BufferedOfflineStream(sample_rate=SAMPLE_RATE)

    def accept_wave(self, stream: _BufferedOfflineStream, audio_pcm: bytes) -> str:
        """Accept PCM bytes and return the latest transcript hypothesis."""
        if not audio_pcm:
            return stream.last_text

        stream.audio.extend(audio_pcm)
        result = self.transcribe(bytes(stream.audio), sample_rate=stream.sample_rate)
        text = result["text"]
        if text:
            stream.last_text = text
            if stream.first_text_at is None:
                stream.first_text_at = time.time()
        return stream.last_text

    def accept_wave_with_ttft(
        self,
        stream: _BufferedOfflineStream,
        audio_pcm: bytes,
    ):
        """Accept PCM bytes and return transcript plus time-to-first-text."""
        text = self.accept_wave(stream, audio_pcm)
        ttft = None
        if stream.first_text_at is not None:
            ttft = stream.first_text_at - stream.started_at
        return text, ttft

    def is_endpoint(self, stream: _BufferedOfflineStream) -> bool:
        """Offline compatibility mode does not perform endpoint detection."""
        return False
