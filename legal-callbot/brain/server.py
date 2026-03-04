"""
Legal CallBot — Brain Worker
HTTP server cho testing (Bước 1). Bước 2 sẽ chuyển sang gRPC.
FastAPI thay vì BaseHTTPRequestHandler để hỗ trợ async streaming.
"""
import json
import logging
import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from config import config
from core.llm import LLMClient
from core.rag import RAGPipeline
from grpc_handler import BrainServiceHandler

# ─── Logging ─────────────────────────────────────────────
logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("brain")

# ─── Initialize Components ──────────────────────────────
llm = LLMClient(api_key=config.gemini_api_key, model=config.gemini_model)
rag = RAGPipeline(
    qdrant_url=config.qdrant_url,
    qdrant_api_key=config.qdrant_api_key,
    collection=config.qdrant_collection,
)
handler = BrainServiceHandler(llm=llm, rag=rag)

# ─── FastAPI App ─────────────────────────────────────────
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
    """
    Non-streaming endpoint — trả full response.
    Dùng cho test đơn giản.
    """
    body = await request.json()
    query = body.get("query", "")
    session_id = body.get("session_id", "test-session")
    history = body.get("conversation_history", [])

    full_text = []
    timing = {}
    async for chunk in handler.think(query, session_id, history):
        if chunk["text"]:
            full_text.append(chunk["text"])
        if "timing" in chunk:
            timing.update(chunk["timing"])

    return {
        "text": " ".join(full_text),
        "timing": timing,
    }


@app.post("/think/stream")
async def think_stream(request: Request):
    """
    Streaming endpoint — trả từng chunk qua SSE.
    Dùng cho production pipeline.
    """
    body = await request.json()
    query = body.get("query", "")
    session_id = body.get("session_id", "test-session")
    history = body.get("conversation_history", [])

    async def generate():
        async for chunk in handler.think(query, session_id, history):
            yield json.dumps(chunk, ensure_ascii=False) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
    )


if __name__ == "__main__":
    logger.info(f"🧠 Brain Worker starting on port {config.port}")
    uvicorn.run(app, host="0.0.0.0", port=config.port)
