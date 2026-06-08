import json
import logging
import asyncio
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from brain.config import config
from brain.core.llm import LLMClient
from brain.core.rag import RAGPipeline
from brain.core.chunker import chunk_llm_stream
from brain.grpc_handler import BrainServiceHandler

logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("brain")

llm = LLMClient(api_key=config.gemini_api_key, model=config.gemini_model)
qdrant_url = config.qdrant_url or f"http://{config.qdrant_host}:{config.qdrant_port}"
rag = RAGPipeline(
    qdrant_url=qdrant_url,
    qdrant_api_key=config.qdrant_api_key,
    collection=config.qdrant_collection,
    qdrant_path=config.qdrant_path or None,
    qdrant_snapshot_path=config.qdrant_snapshot_path or None,
    qdrant_snapshot_force_restore=config.qdrant_snapshot_force_restore,
    qdrant_snapshot_timeout_s=config.qdrant_snapshot_timeout_s,
    qdrant_snapshot_priority=config.qdrant_snapshot_priority,
    llm_client=llm,
)
handler = BrainServiceHandler(
    llm=llm, rag=rag,
    rag_fetch_k=config.rag_fetch_k,
    rag_top_k=config.rag_top_k,
    min_chunk_size=config.min_chunk_size,
    use_hyde=config.rag_use_hyde,
)

app = FastAPI(title="Brain Worker", version="0.2.0")


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "brain"}


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "brain",
        "version": "0.2.0",
        "mode": "gemini-streaming",
    }


@app.post("/think")
async def think(request: Request):
    body = await request.json()
    query = body.get("query", "")
    session_id = body.get("session_id", "test-session")
    history = body.get("conversation_history", [])

    full_text = []
    timing = {}
    contexts = []
    async for chunk in handler.think(query, session_id, history):
        if chunk["text"]:
            full_text.append(chunk["text"])
        if "timing" in chunk:
            timing.update(chunk["timing"])
        if "contexts" in chunk:
            contexts = chunk["contexts"]

    return {
        "text": " ".join(full_text),
        "timing": timing,
        "contexts": contexts,
    }


@app.post("/summarize")
async def summarize(request: Request):
    body = await request.json()
    old_summary: str = body.get("summary", "")
    turns: list = body.get("turns", [])

    parts = []
    if old_summary:
        parts.append(f"Tóm tắt trước: {old_summary}")
    for t in turns:
        role_label = "Người dùng" if t.get("role") == "user" else "Tư vấn viên"
        parts.append(f"{role_label}: {t.get('text', '')}")

    prompt = (
        "Tóm tắt ngắn gọn (tối đa 3 câu) hội thoại dưới đây. "
        "Giữ lại thông tin quan trọng về sức khỏe, tình trạng của người dùng "
        "và các chủ đề dinh dưỡng đã được thảo luận:\n\n"
        + "\n".join(parts)
        + "\n\nTóm tắt:"
    )
    summary = await llm.generate(prompt, temperature=0)
    return {"summary": summary.strip()}


@app.post("/think/stream")
async def think_stream(request: Request):
    body = await request.json()
    query = body.get("query", "")
    session_id = body.get("session_id", "test-session")
    history = body.get("conversation_history", [])
    summary = body.get("conversation_summary", "")

    async def generate():
        started = time.time()
        try:
            async for chunk in handler.think(query, session_id, history, summary):
                yield json.dumps(chunk, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.exception("[%s] think/stream failed", session_id)
            error_chunk = {
                "text": "Xin lỗi, Brain đang gặp lỗi xử lý. Vui lòng thử lại.",
                "is_final": True,
                "error": True,
                "error_type": type(e).__name__,
                "timing": {"total_ms": round((time.time() - started) * 1000, 1)},
            }
            yield json.dumps(error_chunk, ensure_ascii=False) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )


@app.post("/pipeline/audio-stream")
async def pipeline_audio_stream(request: Request):
    """
    LLM streaming → sentence chunker → TTS streaming → PCM audio.

    Flow:
      1. Gọi RAG + LLM như /think/stream
      2. Mỗi khi chunker tích đủ 1 câu → POST sang TTS /speak/stream ngay
      3. Yield PCM bytes từ TTS liên tục về client
      4. Cuối cùng yield NDJSON final chunk (is_final=True) với timing + contexts

    Response:
      Content-Type: audio/pcm
      X-Sample-Rate: 24000
      body: raw PCM int16 chunks xen kẽ, kết thúc bằng final metadata line
    """
    body = await request.json()
    query = body.get("query", "")
    session_id = body.get("session_id", "pipeline")
    history = body.get("conversation_history", [])

    async def generate():
        final_meta = {}

        async def _llm_chunks():
            nonlocal final_meta
            async for chunk in handler.think(query, session_id, history):
                if chunk.get("is_final"):
                    final_meta = chunk
                else:
                    yield chunk

        async with httpx.AsyncClient(timeout=None) as client:
            async for sentence in chunk_llm_stream(_llm_chunks()):
                if not sentence.strip():
                    continue
                # Stream PCM từ TTS ngay khi có 1 câu hoàn chỉnh
                async with client.stream(
                    "POST",
                    f"{config.tts_url}/speak/stream",
                    json={"text": sentence},
                ) as tts_resp:
                    async for pcm_chunk in tts_resp.aiter_bytes(chunk_size=4800):
                        yield pcm_chunk

        # Gửi final metadata cuối cùng dưới dạng newline-delimited JSON
        yield b"\n" + json.dumps(final_meta, ensure_ascii=False).encode() + b"\n"

    return StreamingResponse(
        generate(),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": "24000"},
    )


if __name__ == "__main__":
    logger.info(f"Brain Worker starting on port {config.port}")
    uvicorn.run(app, host="0.0.0.0", port=config.port)
