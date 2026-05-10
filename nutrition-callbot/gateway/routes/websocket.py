import asyncio
import json
import logging
from uuid import uuid4

import httpx
import websockets
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
    fallback_history: list = []
    audio_buffer: list[bytes] = []

    # VAD mode state
    vad_active = False
    vad_audio_q: asyncio.Queue | None = None
    vad_task: asyncio.Task | None = None

    async def _get_ctx():
        if mem:
            return await mem.get_context(session_id)
        return {"summary": "", "turns": fallback_history}

    async def _save_turn(user_text: str, bot_text: str):
        if bot_text:
            if mem:
                await mem.append_turn(session_id, "user", user_text)
                await mem.append_turn(session_id, "assistant", bot_text)
            else:
                fallback_history.extend([
                    {"role": "user", "text": user_text},
                    {"role": "assistant", "text": bot_text},
                ])
                fallback_history[:] = fallback_history[-6:]

    async def _process_transcript(transcript: str):
        """Run Brain→TTS pipeline và stream events về client."""
        ctx = await _get_ctx()
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
        await _save_turn(transcript, bot_text)

    # ── VAD worker: kết nối WebSocket tới ASR, forward audio ─────────
    async def _vad_worker(audio_q: asyncio.Queue):
        try:
            async with websockets.connect(
                settings.asr_vad_ws_url,
                ping_interval=20,
                ping_timeout=60,
            ) as asr_ws:
                logger.info("[%s] VAD worker connected to ASR", session_id)
                await websocket.send_json({
                    "type": "vad_ready",
                    "session_id": session_id,
                })

                async def _sender():
                    while True:
                        chunk = await audio_q.get()
                        if chunk is None:
                            await asr_ws.send(json.dumps({"type": "end"}))
                            break
                        await asr_ws.send(chunk)

                sender_task = asyncio.create_task(_sender())
                try:
                    while True:
                        try:
                            raw = await asyncio.wait_for(asr_ws.recv(), timeout=30.0)
                        except asyncio.TimeoutError:
                            if sender_task.done():
                                break
                            continue

                        data = json.loads(raw)

                        if data.get("error"):
                            await websocket.send_json({
                                "type": "error",
                                "session_id": session_id,
                                "code": data.get("code", "VAD_ERROR"),
                                "message": data["error"],
                            })
                            break

                        if data.get("is_final") and data.get("text"):
                            transcript = data["text"]
                            logger.info("[%s] VAD transcript: %s", session_id, transcript[:80])
                            await websocket.send_json({
                                "type": "transcript",
                                "session_id": session_id,
                                "text": transcript,
                                "is_final": True,
                            })
                            await _process_transcript(transcript)

                        if sender_task.done():
                            break
                finally:
                    sender_task.cancel()
                    with asyncio.suppress(asyncio.CancelledError, Exception):
                        await sender_task

        except Exception as e:
            logger.exception("[%s] VAD worker error", session_id)
            with asyncio.suppress(Exception):
                await websocket.send_json({
                    "type": "error",
                    "session_id": session_id,
                    "code": "VAD_WORKER_ERROR",
                    "message": str(e),
                })
        finally:
            logger.info("[%s] VAD worker stopped", session_id)

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # ── Binary: audio chunk đang stream từ mic ─────────────────
            if message.get("bytes") is not None:
                chunk = message["bytes"]
                if vad_active and vad_audio_q is not None:
                    # VAD mode: forward thẳng tới VAD worker
                    await vad_audio_q.put(chunk)
                else:
                    # PTT mode: buffer cho đến khi end_speech
                    audio_buffer.append(chunk)
                continue

            text_data = message.get("text")
            if not text_data:
                continue

            try:
                payload = json.loads(text_data)
            except json.JSONDecodeError:
                payload = {"type": "text", "text": text_data}

            msg_type = payload.get("type", "text")

            # ── start_vad: bật chế độ tự động phát hiện giọng nói ──────
            if msg_type == "start_vad":
                if vad_active:
                    await websocket.send_json({
                        "type": "error",
                        "session_id": session_id,
                        "code": "VAD_ALREADY_ACTIVE",
                        "message": "VAD đang chạy.",
                    })
                    continue
                vad_active = True
                audio_buffer.clear()
                vad_audio_q = asyncio.Queue()
                vad_task = asyncio.create_task(_vad_worker(vad_audio_q))
                logger.info("[%s] VAD mode started", session_id)

            # ── stop_vad: tắt VAD mode ────────────────────────────────
            elif msg_type == "stop_vad":
                if vad_active and vad_audio_q is not None:
                    await vad_audio_q.put(None)
                    vad_active = False
                    vad_audio_q = None
                    logger.info("[%s] VAD mode stopped", session_id)
                    await websocket.send_json({
                        "type": "vad_stopped",
                        "session_id": session_id,
                    })

            # ── end_speech: PTT mode ─────────────────────────────────
            elif msg_type == "end_speech":
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

                await _process_transcript(transcript)

            # ── text: query text trực tiếp ────────────────────────────
            elif msg_type == "text":
                query = str(payload.get("text", "")).strip()
                if not query:
                    continue
                logger.debug("[%s] Text query: %s", session_id, query[:80])

                ctx = await _get_ctx()
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

                await _save_turn(query, bot_text)

    except WebSocketDisconnect:
        logger.info("[%s] Client disconnected", session_id)
    finally:
        # Cleanup VAD worker khi session kết thúc
        if vad_active and vad_audio_q is not None:
            await vad_audio_q.put(None)
        if vad_task is not None:
            vad_task.cancel()
            with asyncio.suppress(asyncio.CancelledError, Exception):
                await vad_task
