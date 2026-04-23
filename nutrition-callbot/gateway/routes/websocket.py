import json
import logging
from uuid import uuid4

import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import settings
from services.orchestrator import Orchestrator

router = APIRouter(tags=["voice"])
logger = logging.getLogger("gateway.routes.websocket")
orchestrator = Orchestrator()


@router.websocket("/ws/voice")
async def voice_chat(websocket: WebSocket):
    session_id = str(uuid4())
    await websocket.accept()
    logger.info("[%s] Client connected", session_id)

    conversation_history = []
    asr_ws = None

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # ── Binary: audio chunk đang stream từ mic ────────────────
            if message.get("bytes") is not None:
                audio_chunk = message["bytes"]
                if asr_ws is None:
                    asr_ws = await websockets.connect(settings.asr_ws_url)
                    logger.info("[%s] ASR stream opened", session_id)
                await asr_ws.send(audio_chunk)
                continue

            text_data = message.get("text")
            if not text_data:
                continue

            try:
                payload = json.loads(text_data)
            except json.JSONDecodeError:
                payload = {"type": "text", "text": text_data}

            msg_type = payload.get("type", "text")

            # ── end_speech: user bấm dừng nói ────────────────────────
            if msg_type == "end_speech":
                if asr_ws is None:
                    await websocket.send_json({
                        "type": "error",
                        "session_id": session_id,
                        "code": "NO_AUDIO",
                        "message": "Không có audio.",
                    })
                    continue

                await asr_ws.send(json.dumps({"type": "end"}))
                asr_result = json.loads(await asr_ws.recv())
                await asr_ws.close()
                asr_ws = None

                transcript = (asr_result or {}).get("text", "").strip()
                logger.info("[%s] ASR transcript: %s", session_id, transcript[:80])

                await websocket.send_json({
                    "type": "transcript",
                    "session_id": session_id,
                    "text": transcript,
                    "is_final": True,
                })

                if not transcript:
                    await websocket.send_json({
                        "type": "error",
                        "session_id": session_id,
                        "code": "ASR_EMPTY",
                        "message": "ASR không nhận diện được nội dung.",
                    })
                    continue

                async for event in orchestrator.process_text(
                    session_id, transcript, conversation_history
                ):
                    if event.get("type") == "transcript":
                        continue
                    if event.get("type") == "audio_chunk":
                        await websocket.send_bytes(event["audio"])
                    else:
                        await websocket.send_json(event)

            # ── text: query text trực tiếp ────────────────────────────
            elif msg_type == "text":
                query = str(payload.get("text", "")).strip()
                if not query:
                    continue
                logger.debug("[%s] Text query: %s", session_id, query[:80])
                async for event in orchestrator.process_text(
                    session_id, query, conversation_history
                ):
                    if event.get("type") == "audio_chunk":
                        await websocket.send_bytes(event["audio"])
                    else:
                        await websocket.send_json(event)

    except WebSocketDisconnect:
        logger.info("[%s] Client disconnected", session_id)
    finally:
        if asr_ws:
            try:
                await asr_ws.close()
            except Exception:
                pass
