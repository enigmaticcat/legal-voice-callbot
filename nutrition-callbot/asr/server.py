from __future__ import annotations

import json
import logging
import asyncio
from time import perf_counter
from datetime import datetime

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
import uvicorn

from config import config
from grpc_handler import ASRServiceHandler
from observability.metrics import ActiveRequest, MetricsRegistry

logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("asr")
metrics = MetricsRegistry("asr")

handler = ASRServiceHandler()
app = FastAPI()


@app.get("/metrics")
async def prometheus_metrics():
    return Response(metrics.render(), media_type="text/plain; version=0.0.4")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "asr",
        "timestamp": datetime.utcnow().isoformat(),
        "provider": config.provider,
    }


@app.post("/transcribe")
async def transcribe(request: Request):
    audio_pcm = await request.body()
    started = perf_counter()
    with ActiveRequest(metrics, "callbot_asr_active_requests", endpoint="transcribe"):
        try:
            result = await asyncio.to_thread(handler.transcriber.transcribe, audio_pcm)
            metrics.inc("callbot_asr_requests_total", endpoint="transcribe", status="ok")
        except Exception:
            metrics.inc("callbot_asr_requests_total", endpoint="transcribe", status="error")
            raise
        finally:
            metrics.observe(
                "callbot_asr_transcribe_duration_seconds",
                perf_counter() - started,
                endpoint="transcribe",
            )
    return {"text": result["text"], "is_final": True, "confidence": result["confidence"]}


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """Buffer all audio chunks, transcribe in one shot when 'end' is received."""
    await websocket.accept()
    metrics.add_gauge("callbot_asr_ws_sessions", 1, endpoint="ws_transcribe")
    audio_chunks: list[bytes] = []
    logger.info("ASR WebSocket session started")

    try:
        while True:
            message = await websocket.receive()

            if message.get("bytes"):
                audio_chunks.append(message["bytes"])

            elif message.get("text"):
                data = json.loads(message["text"])
                if data.get("type") == "end":
                    audio_data = b"".join(audio_chunks)
                    audio_chunks.clear()
                    result = await asyncio.to_thread(
                        handler.transcriber.transcribe, audio_data
                    )
                    logger.info("ASR final: %s", result["text"][:80])
                    await websocket.send_json({"text": result["text"], "is_final": True})
                    return

    except WebSocketDisconnect:
        logger.info("ASR WebSocket session disconnected")
    finally:
        metrics.add_gauge("callbot_asr_ws_sessions", -1, endpoint="ws_transcribe")


@app.get("/vad/status")
async def vad_status():
    return {
        "available": handler._vad_available,
        "model_path": config.vad_model_path,
        "threshold": config.vad_threshold,
        "min_silence_ms": config.vad_min_silence_ms,
        "min_speech_ms": config.vad_min_speech_ms,
    }


@app.websocket("/ws/transcribe/vad")
async def ws_transcribe_vad(websocket: WebSocket):
    """
    Streaming VAD endpoint.
    Client gửi:
      - binary: PCM int16 LE 16kHz mono (chunk bất kỳ kích thước)
      - JSON {"type": "end"}: flush và đóng session

    Server trả về JSON khi phát hiện xong một đoạn lời:
      {"text": "...", "is_final": true, "confidence": 0.95}
    """
    await websocket.accept()
    metrics.add_gauge("callbot_asr_ws_sessions", 1, endpoint="ws_transcribe_vad")

    if not handler._vad_available:
        await websocket.send_json({
            "error": "VAD model not available",
            "code": "NO_VAD",
        })
        await websocket.close()
        return

    vad = handler.create_vad_session()
    logger.info("VAD streaming session started")

    async def _transcribe_segment(samples: np.ndarray) -> dict | None:
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        started = perf_counter()
        with ActiveRequest(metrics, "callbot_asr_active_requests", endpoint="vad_segment"):
            try:
                result = await asyncio.to_thread(handler.transcriber.transcribe, pcm)
                metrics.inc("callbot_asr_requests_total", endpoint="vad_segment", status="ok")
            except Exception:
                metrics.inc("callbot_asr_requests_total", endpoint="vad_segment", status="error")
                raise
            finally:
                metrics.observe(
                    "callbot_asr_transcribe_duration_seconds",
                    perf_counter() - started,
                    endpoint="vad_segment",
                )
        if result["text"]:
            return {"text": result["text"], "is_final": True, "confidence": result["confidence"]}
        return None

    # Timeout: nếu không nhận được chunk nào trong 30s thì đóng session.
    # Tránh coroutine treo khi gateway disconnect không clean (crash, network drop).
    _RECEIVE_TIMEOUT_S = 30

    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), timeout=_RECEIVE_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                logger.warning("VAD session timed out after %ds inactivity, closing", _RECEIVE_TIMEOUT_S)
                break

            if message.get("type") == "websocket.disconnect":
                break

            if message.get("bytes"):
                segments, speech_started = await asyncio.to_thread(vad.accept_chunk, message["bytes"])
                if speech_started:
                    await websocket.send_json({"type": "speech_start"})
                for seg in segments:
                    resp = await _transcribe_segment(seg)
                    if resp:
                        logger.info("VAD→ASR: %s", resp["text"][:80])
                        await websocket.send_json(resp)

            elif message.get("text"):
                data = json.loads(message["text"])
                if data.get("type") == "end":
                    segments = await asyncio.to_thread(vad.flush)
                    for seg in segments:
                        resp = await _transcribe_segment(seg)
                        if resp:
                            logger.info("VAD flush: %s", resp["text"][:80])
                            await websocket.send_json(resp)
                    vad.reset()
                    return

    except WebSocketDisconnect:
        logger.info("VAD WebSocket session disconnected")
    finally:
        vad.reset()
        metrics.add_gauge("callbot_asr_ws_sessions", -1, endpoint="ws_transcribe_vad")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)
