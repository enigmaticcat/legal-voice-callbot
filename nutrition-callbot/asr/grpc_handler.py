"""
gRPC Handler — StreamTranscribe() implementation
Nhận stream AudioChunk (PCM 16kHz) → yield TranscriptChunk.

Mỗi session duy trì một sherpa-onnx online stream riêng.
Khi recognizer phát hiện endpoint (dứt lời) → yield is_final=True và reset stream.
"""
import logging
from typing import AsyncIterable, AsyncGenerator

from core.transcriber import Transcriber

logger = logging.getLogger("asr.grpc_handler")


class ASRServiceHandler:
    """
    Handler cho ASR gRPC Service: StreamTranscribe()

    Luồng xử lý:
      1. Nhận AudioChunk { data: bytes (PCM 16kHz), session_id: str }
      2. Đẩy audio vào sherpa-onnx online stream của session
      3. Khi text thay đổi → yield partial TranscriptChunk
      4. Khi recognizer báo endpoint → yield is_final=True, reset stream
      5. Khi client đóng stream → flush kết quả cuối nếu còn
    """

    def __init__(self):
        self.transcriber = Transcriber()
        # session_id → sherpa online stream
        self._streams: dict = {}

    def _get_or_create_stream(self, session_id: str):
        if session_id not in self._streams:
            self._streams[session_id] = self.transcriber.create_stream()
            logger.info(f"[{session_id}] New ASR stream created")
        return self._streams[session_id]

    def _reset_stream(self, session_id: str):
        """Xoá stream cũ để reset cho utterance tiếp theo."""
        self._streams.pop(session_id, None)

    async def stream_transcribe(
        self,
        audio_stream: AsyncIterable,
    ) -> AsyncGenerator[dict, None]:
        """
        audio_stream yields dict/object với:
            .data       : bytes — PCM int16 little-endian, mono, 16000 Hz
            .session_id : str

        Yields dict:
            text        : str
            is_final    : bool
            confidence  : float (0.0–1.0, sherpa không có score nên dùng 0.95)
        """
        session_id = None
        last_text = ""

        async for chunk in audio_stream:
            # Hỗ trợ cả dict lẫn object (proto message)
            if isinstance(chunk, dict):
                data = chunk["data"]
                session_id = chunk.get("session_id", "default")
            else:
                data = chunk.data
                session_id = chunk.session_id or "default"

            if not data:
                continue

            stream = self._get_or_create_stream(session_id)

            # Đẩy PCM vào sherpa và decode
            text = self.transcriber.accept_wave(stream, data)

            # Yield partial nếu text mới xuất hiện hoặc thay đổi
            if text and text != last_text:
                last_text = text
                yield {
                    "text": text,
                    "is_final": False,
                    "confidence": 0.95,
                }
                logger.debug(f"[{session_id}] Partial: {text!r}")

            # Sherpa phát hiện endpoint (dứt lời)
            if self.transcriber.is_endpoint(stream):
                final_text = last_text or text
                if final_text:
                    logger.info(f"[{session_id}] Endpoint → final: {final_text!r}")
                    yield {
                        "text": final_text,
                        "is_final": True,
                        "confidence": 0.95,
                    }
                # Reset stream để chuẩn bị cho utterance tiếp theo
                self._reset_stream(session_id)
                last_text = ""

        # Client đóng stream — flush kết quả còn lại nếu chưa emit final
        if session_id and last_text:
            logger.info(f"[{session_id}] Stream ended → flush final: {last_text!r}")
            yield {
                "text": last_text,
                "is_final": True,
                "confidence": 0.95,
            }
            self._reset_stream(session_id)

    def close_session(self, session_id: str):
        """Dọn dẹp khi session kết thúc (gọi từ gRPC server on disconnect)."""
        self._reset_stream(session_id)
        logger.info(f"[{session_id}] Session closed, stream released")
