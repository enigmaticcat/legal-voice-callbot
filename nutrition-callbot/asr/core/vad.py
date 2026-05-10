"""
Voice Activity Detection — Silero VAD via sherpa-onnx VadModel.

Phân tích từng window 512 samples (32ms @ 16kHz), theo dõi trạng thái
SILENCE → SPEECH → TRAILING_SILENCE để cắt ra các đoạn lời nói hoàn chỉnh.
"""
import logging

import numpy as np

logger = logging.getLogger("asr.core.vad")

_WINDOW_SIZE = 512  # 32ms at 16kHz


class VADDetector:
    """Silero VAD wrapper với state machine phát hiện speech segment.

    Nhận PCM int16 bytes theo từng chunk bất kỳ kích thước, trả về
    list các numpy float32 array mỗi khi phát hiện xong một đoạn lời.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        min_silence_ms: int = 500,
        min_speech_ms: int = 250,
        sample_rate: int = 16_000,
    ):
        import sherpa_onnx

        s = sherpa_onnx.SileroVadModelConfig()
        s.model = model_path
        s.threshold = threshold
        s.min_silence_duration = min_silence_ms / 1000.0
        s.min_speech_duration = min_speech_ms / 1000.0
        s.window_size = _WINDOW_SIZE

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad = s
        cfg.sample_rate = sample_rate

        self._vad = sherpa_onnx.VadModel.create(cfg)
        self._sample_rate = sample_rate
        # Silence threshold in samples (derived from model config)
        self._silence_needed: int = self._vad.min_silence_duration_samples
        self._speech_needed: int = self._vad.min_speech_duration_samples

        # Internal state
        self._remainder = np.zeros(0, dtype=np.float32)
        self._speech_frames: list[np.ndarray] = []
        self._in_speech = False
        self._silence_samples = 0  # accumulated silence samples after speech

        logger.info(
            "Silero VAD loaded (threshold=%.2f, silence=%dms, speech=%dms)",
            threshold, min_silence_ms, min_speech_ms,
        )

    # ── Public API ────────────────────────────────────────────────────

    def accept_chunk(self, pcm_bytes: bytes) -> list[np.ndarray]:
        """Process PCM int16 bytes; return completed speech segments (float32)."""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        buf = np.concatenate([self._remainder, samples]) if self._remainder.size else samples

        completed: list[np.ndarray] = []

        while buf.size >= _WINDOW_SIZE:
            window = buf[:_WINDOW_SIZE]
            buf = buf[_WINDOW_SIZE:]
            self._process_window(window, completed)

        self._remainder = buf
        return completed

    def flush(self) -> list[np.ndarray]:
        """Force-emit any buffered speech (call when stream ends)."""
        completed: list[np.ndarray] = []
        if self._in_speech and self._speech_frames:
            completed.append(np.concatenate(self._speech_frames))
            self._speech_frames = []
            self._in_speech = False
            self._silence_samples = 0
        self._remainder = np.zeros(0, dtype=np.float32)
        return completed

    def reset(self):
        """Reset all state (new session)."""
        self._vad.reset()
        self._remainder = np.zeros(0, dtype=np.float32)
        self._speech_frames = []
        self._in_speech = False
        self._silence_samples = 0

    # ── Internal ──────────────────────────────────────────────────────

    def _process_window(self, window: np.ndarray, completed: list[np.ndarray]):
        is_speech = self._vad.is_speech(window.tolist())

        if is_speech:
            self._in_speech = True
            self._silence_samples = 0
            self._speech_frames.append(window)
        else:
            if self._in_speech:
                # Trailing silence — still buffer it (natural pause)
                self._silence_samples += _WINDOW_SIZE
                self._speech_frames.append(window)

                if self._silence_samples >= self._silence_needed:
                    segment = np.concatenate(self._speech_frames)
                    completed.append(segment)
                    self._speech_frames = []
                    self._in_speech = False
                    self._silence_samples = 0
