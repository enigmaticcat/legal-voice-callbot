"""
Voice Activity Detection — Silero VAD via silero-vad (PyTorch).

Phân tích từng window 512 samples (32ms @ 16kHz), theo dõi trạng thái
SILENCE → SPEECH → TRAILING_SILENCE để cắt ra các đoạn lời nói hoàn chỉnh.
"""
import logging

import numpy as np
import torch
from silero_vad import load_silero_vad
from silero_vad.utils_vad import VADIterator

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
        del model_path

        self._sample_rate = sample_rate

        self._min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self._device = torch.device("cpu")
        self._model = load_silero_vad()
        self._model.to(self._device)
        self._iterator = VADIterator(
            self._model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=0,
        )

        # Internal state
        self._remainder = np.zeros(0, dtype=np.float32)
        self._speech_frames: list[np.ndarray] = []
        self._in_speech = False

        logger.info(
            "Silero VAD loaded (threshold=%.2f, silence=%dms, speech=%dms)",
            threshold, min_silence_ms, min_speech_ms,
        )

    # ── Public API ────────────────────────────────────────────────────

    def accept_chunk(self, pcm_bytes: bytes) -> tuple[list[np.ndarray], bool]:
        """Process PCM int16 bytes.

        Returns:
            (completed_segments, speech_just_started)
            speech_just_started=True nếu chunk này chứa thời điểm bắt đầu nói.
        """
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        buf = np.concatenate([self._remainder, samples]) if self._remainder.size else samples

        completed: list[np.ndarray] = []
        speech_started = False

        while buf.size >= _WINDOW_SIZE:
            window = buf[:_WINDOW_SIZE]
            buf = buf[_WINDOW_SIZE:]
            if self._process_window(window, completed):
                speech_started = True

        self._remainder = buf
        return completed, speech_started

    def flush(self) -> list[np.ndarray]:
        """Force-emit any buffered speech (call when stream ends)."""
        completed: list[np.ndarray] = []
        if self._in_speech and self._speech_frames:
            segment = np.concatenate(self._speech_frames)
            if segment.size >= self._min_speech_samples:
                completed.append(segment)
            self._speech_frames = []
            self._in_speech = False
        self._remainder = np.zeros(0, dtype=np.float32)
        return completed

    def reset(self):
        """Reset all state (new session)."""
        self._iterator.reset_states()
        self._remainder = np.zeros(0, dtype=np.float32)
        self._speech_frames = []
        self._in_speech = False

    # ── Internal ──────────────────────────────────────────────────────

    def _process_window(self, window: np.ndarray, completed: list[np.ndarray]) -> bool:
        """Return True nếu window này chứa thời điểm speech start."""
        event = self._iterator(window, return_seconds=False)
        speech_started = False

        if event and "start" in event:
            self._in_speech = True
            self._speech_frames = []
            speech_started = True

        if self._in_speech:
            self._speech_frames.append(window)

        if event and "end" in event and self._in_speech:
            segment = np.concatenate(self._speech_frames)
            if segment.size >= self._min_speech_samples:
                completed.append(segment)
            self._speech_frames = []
            self._in_speech = False

        return speech_started
