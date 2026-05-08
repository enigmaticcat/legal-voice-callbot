import json
import logging
from uuid import uuid4

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import settings
from services.orchestrator import Orchestrator
import services.session_memory as session_memory

router = APIRouter(tags=["voice"])
logger = logging.getLogger("gateway.routes.websocket")
orchestrator = Orchestrator()

_ASR_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)


@router.websocket("/ws/voice")
async def voice_chat(websocket: WebSocket):
    session_id = str(uuid4())
    await websocket.accept()
    logger.info("[%s] Client connected", session_id)

    mem = session_memory.get()
    fallback_history: list = []  # dùng khi Redis không khả dụng
    audio_buffer: list[bytes] = []

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # ── Binary: audio chunk đang stream từ mic ────────────────
            if message.get("bytes") is not None:
                audio_buffer.append(message["bytes"])
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
                if not audio_buffer:
                    await websocket.send_json({
                        "type": "error",
                        "session_id": session_id,
                        "code": "NO_AUDIO",
                        "message": "Không có audio.",
                    })
                    continue

                audio_data = b"".join(audio_buffer)
                audio_buffer.clear()
                logger.info("[%s] ASR batch: %d bytes", session_id, len(audio_data))

                try:
                    async with httpx.AsyncClient(timeout=_ASR_TIMEOUT) as client:
                        response = await client.post(
                            f"{settings.asr_http_url}/transcribe",
                            content=audio_data,
                            headers={"Content-Type": "application/octet-stream"},
                        )
                        response.raise_for_status()
                        asr_result = response.json()
                except Exception as e:
                    logger.exception("[%s] ASR HTTP failed", session_id)
                    await websocket.send_json({
                        "type": "error",
                        "session_id": session_id,
                        "code": "ASR_ERROR",
                        "message": "ASR lỗi.",
                        "detail": str(e),
                    })
                    continue

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

                if mem:
                    ctx = await mem.get_context(session_id)
                else:
                    ctx = {"summary": "", "turns": fallback_history}

                bot_text = ""
                async for event in orchestrator.process_text(
                    session_id, transcript,
                    conversation_history=ctx["turns"],
                    conversation_summary=ctx["summary"],
                ):
                    if event.get("type") == "transcript":
                        continue
                    if event.get("type") == "bot_response" and not event.get("is_final"):
                        bot_text += event.get("text", "")
                    if event.get("type") == "audio_chunk":
                        await websocket.send_bytes(event["audio"])
                    else:
                        await websocket.send_json(event)

                if bot_text:
                    if mem:
                        await mem.append_turn(session_id, "user", transcript)
                        await mem.append_turn(session_id, "assistant", bot_text)
                    else:
                        fallback_history.extend([
                            {"role": "user", "text": transcript},
                            {"role": "assistant", "text": bot_text},
                        ])
                        fallback_history[:] = fallback_history[-6:]

            # ── text: query text trực tiếp ────────────────────────────
            elif msg_type == "text":
                query = str(payload.get("text", "")).strip()
                if not query:
                    continue
                logger.debug("[%s] Text query: %s", session_id, query[:80])

                if mem:
                    ctx = await mem.get_context(session_id)
                else:
                    ctx = {"summary": "", "turns": fallback_history}

                bot_text = ""
                async for event in orchestrator.process_text(
                    session_id, query,
                    conversation_history=ctx["turns"],
                    conversation_summary=ctx["summary"],
                ):
                    if event.get("type") == "bot_response" and not event.get("is_final"):
                        bot_text += event.get("text", "")
                    if event.get("type") == "audio_chunk":
                        await websocket.send_bytes(event["audio"])
                    else:
                        await websocket.send_json(event)

                if bot_text:
                    if mem:
                        await mem.append_turn(session_id, "user", query)
                        await mem.append_turn(session_id, "assistant", bot_text)
                    else:
                        fallback_history.extend([
                            {"role": "user", "text": query},
                            {"role": "assistant", "text": bot_text},
                        ])
                        fallback_history[:] = fallback_history[-6:]

    except WebSocketDisconnect:
        logger.info("[%s] Client disconnected", session_id)
