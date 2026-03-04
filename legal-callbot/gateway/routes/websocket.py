"""
WebSocket Voice Route
WS /ws/voice — Entry point cho cuộc gọi tư vấn.
Bước 6 sẽ implement orchestrator pipeline thật (ASR → Brain → TTS).
"""
import logging
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["voice"])
logger = logging.getLogger("gateway.routes.websocket")


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
            # Bước 1: Nhận message từ client (dummy echo)
            data = await websocket.receive_text()
            logger.debug(f"[{session_id}] Received: {data[:100]}")

            # Dummy response — sẽ thay bằng orchestrator ở Bước 6
            await websocket.send_json({
                "type": "transcript",
                "session_id": session_id,
                "text": f"[Echo] {data}",
            })

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Client disconnected")
