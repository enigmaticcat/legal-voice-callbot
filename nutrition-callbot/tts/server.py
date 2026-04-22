"""
Nutrition CallBot — TTS Worker (FastAPI)
Endpoints:
  GET  /health
  POST /speak        → WAV bytes + X-TTFB-ms / X-RTF / X-Duration-s headers
  POST /speak/stream → raw PCM int16 stream (24kHz mono)
"""
import io
import time
import wave
import logging
import os
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from core.synthesizer import Synthesizer
from grpc_handler import TTSServiceHandler

logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("tts")

synthesizer = Synthesizer(
    backbone_repo=config.backbone_repo,
    codec_repo=config.codec_repo,
)
handler = TTSServiceHandler(synthesizer=synthesizer)

app = FastAPI(title="TTS Worker", version="0.3.0")


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "tts"}


@app.get("/")
async def root():
    return {"status": "ok", "service": "tts", "version": "0.3.0"}


@app.post("/speak")
async def speak(request: Request):
    """Synthesize text → full WAV. Headers carry TTFB and RTF."""
    body = await request.json()
    text = (body.get("text", "") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    t0 = time.perf_counter()
    pcm_chunks = []
    first_byte_ms = None

    async for chunk in handler.speak(text):
        if first_byte_ms is None:
            first_byte_ms = (time.perf_counter() - t0) * 1000
        pcm_chunks.append(chunk)

    pcm = b"".join(pcm_chunks)
    if not pcm:
        raise HTTPException(status_code=502, detail="TTS returned empty audio")

    synth_s = time.perf_counter() - t0
    duration_s = len(pcm) / 2 / synthesizer.sample_rate  # int16 = 2 bytes/sample

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(synthesizer.sample_rate)
        w.writeframes(pcm)

    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={
            "X-TTFB-ms":    str(round(first_byte_ms or 0, 1)),
            "X-RTF":        str(round(synth_s / duration_s, 3) if duration_s > 0 else "0"),
            "X-Duration-s": str(round(duration_s, 2)),
        },
    )


@app.post("/speak/stream")
async def speak_stream(request: Request):
    """Stream raw PCM int16 chunks as they are synthesized."""
    body = await request.json()
    text = (body.get("text", "") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    async def generate():
        has_audio = False
        async for chunk in handler.speak(text):
            has_audio = True
            yield chunk
        if not has_audio:
            raise RuntimeError("TTS returned empty audio stream")

    return StreamingResponse(
        generate(),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(synthesizer.sample_rate)},
    )


if __name__ == "__main__":
    logger.info(f"TTS Worker starting on port {config.port}")
    uvicorn.run(app, host="0.0.0.0", port=config.port)
