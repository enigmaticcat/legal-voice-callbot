import json
import logging
import asyncio
import time

import httpx
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from brain.config import config
from brain.core.llm import LLMClient
from brain.core.rag import RAGPipeline
from brain.core.retrieval_cache import RetrievalCache
from brain.core.chunker import chunk_llm_stream
from brain.core.doc_parser import parse_document, DocParseError
from brain.grpc_handler import BrainServiceHandler

logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("brain")

llm = LLMClient(
    api_key=config.llm_api_key,
    model=config.llm_model,
    base_url=config.llm_base_url,
)
retrieval_cache = RetrievalCache(
    redis_url=config.redis_url,
    enabled=config.retrieval_cache_enabled,
    required=config.retrieval_cache_required,
    ttl_seconds=config.retrieval_cache_ttl_seconds,
    corpus_version=config.corpus_version,
)
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
    retrieval_cache=retrieval_cache,
    semantic_cache_enabled=config.semantic_cache_enabled,
    semantic_cache_collection=config.semantic_cache_collection,
    semantic_cache_threshold=config.semantic_cache_threshold,
    semantic_cache_ttl_seconds=config.semantic_cache_ttl_seconds,
)
handler = BrainServiceHandler(
    llm=llm, rag=rag,
    rag_fetch_k=config.rag_fetch_k,
    rag_top_k=config.rag_top_k,
    min_chunk_size=config.min_chunk_size,
    use_hyde=config.rag_use_hyde,
)

app = FastAPI(title="Brain Worker", version="0.2.0")


@app.on_event("startup")
async def startup():
    await retrieval_cache.connect()


@app.on_event("shutdown")
async def shutdown():
    await retrieval_cache.close()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "brain",
        "retrieval_cache_connected": retrieval_cache.connected,
    }


@app.get("/cache/stats")
async def cache_stats():
    return {
        "exact": await retrieval_cache.stats(),
        "semantic": await rag.semantic_cache_stats(),
    }


@app.delete("/cache")
async def clear_cache():
    return {
        "exact_deleted_keys": await retrieval_cache.clear(),
        "semantic_recreated": await rag.clear_semantic_cache(),
    }


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "brain",
        "version": "0.2.0",
        "mode": "local-qwen-streaming",
        "model": config.llm_model,
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
    retrieval_cache_meta = {}
    async for chunk in handler.think(query, session_id, history):
        if chunk["text"]:
            full_text.append(chunk["text"])
        if "timing" in chunk:
            timing.update(chunk["timing"])
        if "contexts" in chunk:
            contexts = chunk["contexts"]
        if "retrieval_cache" in chunk:
            retrieval_cache_meta = chunk["retrieval_cache"]

    return {
        "text": " ".join(full_text),
        "timing": timing,
        "contexts": contexts,
        "retrieval_cache": retrieval_cache_meta,
    }


@app.post("/documents/upload")
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        return JSONResponse(status_code=400, content={"error": "File quá lớn (tối đa 5 MB)"})
    try:
        chunks = parse_document(file.filename or "upload", content)
    except DocParseError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    try:
        n = await rag.upsert_user_docs(
            session_id=session_id,
            filename=file.filename or "upload",
            chunks=chunks,
        )
    except Exception as e:
        logger.exception("[%s] upsert_user_docs failed", session_id)
        return JSONResponse(status_code=500, content={"error": "Không thể lưu tài liệu."})

    return {"chunks": n, "filename": file.filename, "session_id": session_id}


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
