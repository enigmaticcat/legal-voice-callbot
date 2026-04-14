"""
WebSocket Voice Route
WS /ws/voice — Entry point cho cuộc gọi tư vấn.
Bước 6 sẽ implement orchestrator pipeline thật (ASR → Brain → TTS).
"""
import logging
import json
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.orchestrator import Orchestrator

router = APIRouter(tags=["voice"])
logger = logging.getLogger("gateway.routes.websocket")
orchestrator = Orchestrator()


@router.websocket("/ws/voice")
async def voice_chat(websocket: WebSocket):
    """
    WebSocket endpoint cho cuộc gọi voice.

    Pipeline (sẽ implement ở Bước 6):
      Client Audio → gRPC ASR → Text → gRPC Brain → Text → gRPC TTS → Audio → Client
    """
    session_id = str(uuid4())
    await websocket.accept()
    logger.info(f"[{session_id}] Client connected")

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # Ưu tiên binary cho voice payload.
            if message.get("bytes") is not None:
                audio_data = message["bytes"]
                async for event in orchestrator.process_audio(session_id, audio_data):
                    if event.get("type") == "audio_chunk":
                        await websocket.send_bytes(event["audio"])
                    else:
                        await websocket.send_json(event)
                continue

            text_data = message.get("text")
            if text_data is None:
                continue

            # Hỗ trợ 2 mode:
            # 1) plain text: "toi nen an gi"
            # 2) json text:  {"type":"text","text":"..."}
            query = text_data
            try:
                payload = json.loads(text_data)
                if isinstance(payload, dict) and payload.get("type") == "text":
                    query = str(payload.get("text", "")).strip()
            except json.JSONDecodeError:
                pass

            if not query:
                await websocket.send_json({
                    "type": "error",
                    "session_id": session_id,
                    "message": "Query rỗng.",
                    "code": "EMPTY_QUERY",
                })
                continue

            logger.debug(f"[{session_id}] Received text query: {query[:100]}")
            async for event in orchestrator.process_text(session_id, query):
                if event.get("type") == "audio_chunk":
                    await websocket.send_bytes(event["audio"])
                else:
                    await websocket.send_json(event)

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Client disconnected")
