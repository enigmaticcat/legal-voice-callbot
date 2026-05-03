"""
Transcriber — Sherpa-Onnx Offline Transducer
Model: ZipFormer-RNNT (hynt/Zipformer-30M-RNNT-6000h)
Nhận toàn bộ audio PCM một lần → trả transcript tiếng Việt.
"""
import logging

import numpy as np

logger = logging.getLogger("asr.core.transcriber")

SAMPLE_RATE = 16_000


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

        if self.config.require_cuda:
            self._assert_cuda_available()

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

    @staticmethod
    def _assert_cuda_available():
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
        except Exception as e:
            raise RuntimeError(f"Cannot inspect ONNX Runtime providers: {e}")
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(
                "ASR_REQUIRE_CUDA=true but CUDAExecutionProvider is unavailable. "
                f"providers={providers}"
            )

    def transcribe(self, audio_pcm: bytes, sample_rate: int = SAMPLE_RATE) -> dict:
        """Transcribe a complete audio clip (PCM int16, 16kHz mono)."""
        if not audio_pcm:
            return {"text": "", "confidence": 0.0}

        samples = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0

        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._recognizer.decode_streams([stream])

        text = stream.result.text.strip()
        logger.info("Transcribed (%d bytes → %d chars): %s", len(audio_pcm), len(text), text[:80])
        return {"text": text, "confidence": 0.95}
