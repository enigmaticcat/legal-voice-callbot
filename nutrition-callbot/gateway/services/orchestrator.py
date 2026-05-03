"""
Orchestrator Service
HTTP-only pipeline: ASR -> Brain -> TTS.
"""
import asyncio
import contextlib
import json
import logging
from typing import AsyncGenerator

import httpx

from config import settings

logger = logging.getLogger("gateway.services.orchestrator")


class Orchestrator:
    """Coordinate HTTP streaming pipeline across ASR/Brain/TTS."""

    def __init__(self):
        self.asr_url = settings.asr_http_url
        self.brain_url = settings.brain_http_url
        self.tts_url = settings.tts_http_url
        self.http_timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0)

    @staticmethod
    def _ready_for_tts(buffer: str, min_chars: int = 40) -> bool:
        if len(buffer.strip()) >= min_chars:
            return True
        return any(p in buffer for p in [".", "?", "!", "\n"])

    @staticmethod
    def _clean_for_tts(text: str) -> str:
        import re
        # Strip markdown formatting
        text = re.sub(r'\*{1,3}([^*]+?)\*{1,3}', r'\1', text)       # bold/italic
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)   # headings
        text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)    # bullet list
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)    # numbered list
        text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)                # inline/block code
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)        # links → label only
        text = re.sub(r'_{1,2}([^_]+?)_{1,2}', r'\1', text)          # underscore italic/bold
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    async def _asr_transcribe(self, session_id: str, audio_data: bytes) -> dict:
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            response = await client.post(
                f"{self.asr_url}/transcribe",
                content=audio_data,
                headers={"Content-Type": "application/octet-stream"},
            )
            response.raise_for_status()
            result = response.json()
            logger.info("[%s] ASR done: %s", session_id, (result.get("text") or "")[:120])
            return result

    async def _brain_stream(
        self,
        session_id: str,
        query: str,
        conversation_history: list | None = None,
    ) -> AsyncGenerator[dict, None]:
        payload = {
            "query": query,
            "session_id": session_id,
            "conversation_history": conversation_history or [],
        }

        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            async with client.stream("POST", f"{self.brain_url}/think/stream", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("[%s] Invalid NDJSON chunk from brain", session_id)

    async def _tts_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        saw_audio = False
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            async with client.stream("POST", f"{self.tts_url}/speak/stream", json={"text": text}) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=4800):
                    if chunk:
                        saw_audio = True
                        yield chunk
        if not saw_audio:
            raise RuntimeError("TTS returned empty audio stream")

    async def process_text(
        self,
        session_id: str,
        query: str,
        conversation_history: list | None = None,
    ) -> AsyncGenerator[dict, None]:
        """HTTP pipeline text -> Brain stream -> TTS stream.

        Brain and TTS run in parallel via asyncio.Queue:
        - brain_producer streams text chunks into tts_queue
        - tts_consumer reads tts_queue and synthesises audio concurrently
        - both push events into event_queue for the main loop to yield
        """
        logger.info("[%s] Processing text query", session_id)
        yield {
            "type": "transcript",
            "session_id": session_id,
            "text": query,
            "is_final": True,
        }

        # str = text segment for TTS, None = sentinel (done)
        tts_queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        # dict = event to yield, None = sentinel (all done)
        event_queue: asyncio.Queue = asyncio.Queue()
        saw_any_brain_chunk = False

        async def brain_producer():
            nonlocal saw_any_brain_chunk
            buffer = ""
            try:
                async for brain_chunk in self._brain_stream(session_id, query, conversation_history):
                    text = (brain_chunk or {}).get("text", "")
                    is_final = bool((brain_chunk or {}).get("is_final", False))

                    if text:
                        saw_any_brain_chunk = True
                        await event_queue.put({
                            "type": "bot_response",
                            "session_id": session_id,
                            "text": text,
                            "is_final": False,
                        })
                        buffer += text

                        if self._ready_for_tts(buffer):
                            await tts_queue.put(self._clean_for_tts(buffer))
                            buffer = ""

                    if is_final:
                        break

                if buffer:
                    await tts_queue.put(self._clean_for_tts(buffer))

            except Exception as e:
                logger.exception("[%s] Brain streaming failed", session_id)
                await event_queue.put({
                    "type": "error",
                    "session_id": session_id,
                    "message": "Brain stream loi.",
                    "code": "BRAIN_STREAM_ERROR",
                    "detail": str(e),
                })
            finally:
                await tts_queue.put(None)

        async def tts_consumer():
            sent_audio_start = False
            try:
                while True:
                    segment = await tts_queue.get()
                    if segment is None:
                        break
                    if not sent_audio_start:
                        sent_audio_start = True
                        await event_queue.put({
                            "type": "audio_start",
                            "session_id": session_id,
                            "sample_rate": 24000,
                        })
                    try:
                        async for pcm_chunk in self._tts_stream(segment):
                            await event_queue.put({
                                "type": "audio_chunk",
                                "session_id": session_id,
                                "audio": pcm_chunk,
                            })
                    except Exception as e:
                        logger.exception("[%s] TTS streaming failed", session_id)
                        await event_queue.put({
                            "type": "error",
                            "session_id": session_id,
                            "message": "TTS khong tao duoc audio.",
                            "code": "TTS_STREAM_ERROR",
                            "detail": str(e),
                        })
                        # drain remaining segments so brain_producer unblocks
                        while True:
                            item = await tts_queue.get()
                            if item is None:
                                break
                        return
            finally:
                await event_queue.put(None)

        producer_task = asyncio.create_task(brain_producer())
        consumer_task = asyncio.create_task(tts_consumer())

        try:
            while True:
                event = await event_queue.get()
                if event is None:
                    break
                yield event
        finally:
            producer_task.cancel()
            consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(producer_task, consumer_task, return_exceptions=True)

        if not saw_any_brain_chunk:
            yield {
                "type": "error",
                "session_id": session_id,
                "message": "Brain khong tra ve noi dung.",
                "code": "BRAIN_EMPTY",
            }
            return

        yield {
            "type": "bot_response",
            "session_id": session_id,
            "text": "",
            "is_final": True,
        }

    async def process_audio(
        self,
        session_id: str,
        audio_data: bytes,
        conversation_history: list | None = None,
    ) -> AsyncGenerator[dict, None]:
        """HTTP pipeline audio -> ASR -> Brain -> TTS."""
        if not audio_data:
            yield {
                "type": "error",
                "session_id": session_id,
                "message": "Audio rong.",
                "code": "EMPTY_AUDIO",
            }
            return

        logger.info("[%s] Processing audio (%d bytes)", session_id, len(audio_data))
        asr_result = await self._asr_transcribe(session_id, audio_data)
        transcript = (asr_result or {}).get("text", "").strip()

        yield {
            "type": "transcript",
            "session_id": session_id,
            "text": transcript,
            "is_final": bool((asr_result or {}).get("is_final", True)),
        }

        if not transcript:
            yield {
                "type": "error",
                "session_id": session_id,
                "message": "ASR khong nhan dien duoc noi dung.",
                "code": "ASR_EMPTY",
            }
            return

        async for event in self.process_text(
            session_id=session_id,
            query=transcript,
            conversation_history=conversation_history,
        ):
            if event.get("type") == "transcript":
                continue
            yield event
