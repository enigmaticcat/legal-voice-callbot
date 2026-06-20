"""
Nutrition CallBot — TTS Worker (FastAPI)
Endpoints:
  GET  /health
    POST /speak        → WAV bytes + X-TTFB-ms / X-RTF / X-Duration-s headers
    POST /speak/stream → raw PCM int16 stream (24kHz mono)
    POST /cancel       → cancel synthesis for session_id
"""
import io
import time
import wave
import logging
import os
import sys
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from core.synthesizer import Synthesizer, SynthesisCancelled
from core.tts_cache import TTSCache
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
tts_cache = TTSCache(
    redis_url=config.redis_url,
    enabled=config.cache_enabled,
    required=config.cache_required,
    ttl_seconds=config.cache_ttl_seconds,
    max_bytes=config.cache_max_bytes,
    version=config.cache_version,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await tts_cache.connect()
    if config.preload_model:
        logger.info("Preloading VieNeu-TTS before accepting requests...")
        await asyncio.to_thread(synthesizer.load_model)
    yield
    await tts_cache.close()
    await asyncio.to_thread(synthesizer.close)


app = FastAPI(title="TTS Worker", version="0.3.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "tts",
        "model_loaded": synthesizer.is_loaded,
        "cache_connected": tts_cache.connected,
    }


@app.get("/cache/stats")
async def cache_stats():
    return await tts_cache.stats()


@app.delete("/cache")
async def clear_cache():
    return {"deleted_keys": await tts_cache.clear()}


@app.get("/")
async def root():
    return {"status": "ok", "service": "tts", "version": "0.3.0"}


@app.post("/speak")
async def speak(request: Request):
    """Synthesize text → full WAV. Headers carry TTFB and RTF."""
    body = await request.json()
    text = (body.get("text", "") or "").strip()
    session_id = (body.get("session_id", "") or "").strip() or "default"
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    t0 = time.perf_counter()
    pcm_chunks = []
    first_byte_ms = None

    async for chunk in handler.speak(text, session_id=session_id):
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
    session_id = (body.get("session_id", "") or "").strip() or "default"
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    cache_key = tts_cache.build_key(
        text=text,
        backbone_repo=config.backbone_repo,
        codec_repo=config.codec_repo,
        sample_rate=synthesizer.sample_rate,
    )
    cached_pcm, cache_meta = await tts_cache.get(cache_key)
    if cached_pcm is not None:
        async def generate_cached():
            for offset in range(0, len(cached_pcm), 4800):
                yield cached_pcm[offset:offset + 4800]

        return StreamingResponse(
            generate_cached(),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": str(synthesizer.sample_rate),
                "X-TTS-Cache": "HIT",
                "X-TTS-Cache-Key": cache_meta["key"],
                "X-TTS-Estimated-Saved-ms": str(cache_meta.get("estimated_saved_ms", 0)),
            },
        )

    started = time.perf_counter()
    stream = handler.speak(text, session_id=session_id)

    try:
        first_chunk = await anext(stream)
    except StopAsyncIteration:
        raise HTTPException(status_code=502, detail="TTS returned empty audio stream")
    except Exception as e:
        logger.exception("TTS stream preflight failed")
        raise HTTPException(status_code=502, detail=f"TTS stream failed: {e}")

    async def generate():
        pcm_parts = [first_chunk]
        completed = False
        try:
            yield first_chunk
            async for chunk in stream:
                pcm_parts.append(chunk)
                yield chunk
            completed = True
        except SynthesisCancelled:
            logger.info("TTS stream cancelled; skip cache write for session=%s", session_id)
            return
        except Exception:
            logger.exception("TTS stream interrupted after first chunk")
            return
        finally:
            if completed:
                compute_ms = (time.perf_counter() - started) * 1000
                await tts_cache.set(cache_key, b"".join(pcm_parts), compute_ms)

    return StreamingResponse(
        generate(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(synthesizer.sample_rate),
            "X-TTS-Cache": "MISS",
            "X-TTS-Cache-Key": cache_meta["key"],
        },
    )


@app.post("/cancel")
async def cancel(request: Request):
    body = await request.json()
    session_id = (body.get("session_id", "") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    await handler.cancel(session_id)
    return {"status": "ok", "session_id": session_id}


if __name__ == "__main__":
    logger.info(f"TTS Worker starting on port {config.port}")
    uvicorn.run(app, host="0.0.0.0", port=config.port)
