import json
import logging
import asyncio
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
import uvicorn

from config import config
from grpc_handler import ASRServiceHandler

logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("asr")

handler = ASRServiceHandler()
app = FastAPI()


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
    result = await asyncio.to_thread(handler.transcriber.transcribe, audio_pcm)
    return {"text": result["text"], "is_final": True, "confidence": result["confidence"]}


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """Buffer all audio chunks, transcribe in one shot when 'end' is received."""
    await websocket.accept()
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.port)
