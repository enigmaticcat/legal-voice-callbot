"""
Redis-backed session memory cho conversation history.

Mỗi session lưu:
  session:{id}:turns    — List[{role, text}] — tối đa MAX_RAW_TURNS turns gần nhất
  session:{id}:summary  — string — tóm tắt tích lũy từ các turns cũ hơn

Khi turns > MAX_RAW_TURNS: giữ 2 turns gần nhất, compress phần còn lại vào summary
bằng cách gọi Brain /summarize (async, không block response hiện tại).
"""
import asyncio
import json
import logging
from typing import Optional

import httpx

logger = logging.getLogger("gateway.services.session_memory")

DEFAULT_MAX_RAW_TURNS = 3
DEFAULT_SESSION_TTL = 1800  # 30 phút

_instance: Optional["SessionMemory"] = None


class SessionMemory:

    def __init__(self, redis, brain_url: str, max_raw_turns: int, session_ttl: int):
        self.redis = redis
        self.brain_url = brain_url
        self.max_raw_turns = max_raw_turns
        self.session_ttl = session_ttl

    async def get_context(self, session_id: str) -> dict:
        """Trả về {summary: str, turns: list} cho session này."""
        raw_turns, raw_summary = await asyncio.gather(
            self.redis.get(f"session:{session_id}:turns"),
            self.redis.get(f"session:{session_id}:summary"),
        )
        return {
            "summary": raw_summary.decode() if raw_summary else "",
            "turns": json.loads(raw_turns) if raw_turns else [],
        }

    async def append_turn(self, session_id: str, role: str, content: str):
        """Append turn, trigger summarization async nếu vượt ngưỡng."""
        key_turns = f"session:{session_id}:turns"
        key_summary = f"session:{session_id}:summary"

        raw = await self.redis.get(key_turns)
        turns = json.loads(raw) if raw else []
        turns.append({"role": role, "text": content})

        if len(turns) > self.max_raw_turns:
            to_compress = turns[:-2]
            turns = turns[-2:]
            asyncio.create_task(
                self._compress(session_id, key_summary, to_compress)
            )

        await asyncio.gather(
            self.redis.setex(key_turns, self.session_ttl, json.dumps(turns, ensure_ascii=False)),
            self.redis.expire(key_summary, self.session_ttl),
        )

    async def _compress(self, session_id: str, key_summary: str, turns: list):
        """Gọi Brain /summarize để nén turns cũ vào running summary."""
        try:
            raw = await self.redis.get(key_summary)
            old_summary = raw.decode() if raw else ""

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.brain_url}/summarize",
                    json={"summary": old_summary, "turns": turns},
                )
                resp.raise_for_status()
                new_summary = resp.json().get("summary", "")

            await self.redis.setex(key_summary, self.session_ttl, new_summary)
            logger.info("[%s] Summary updated (%d chars)", session_id, len(new_summary))
        except Exception:
            logger.exception("[%s] Summarization failed, keeping old summary", session_id)


async def init(
    redis_url: str,
    brain_url: str,
    max_raw_turns: int = DEFAULT_MAX_RAW_TURNS,
    session_ttl: int = DEFAULT_SESSION_TTL,
) -> SessionMemory:
    global _instance
    import redis.asyncio as aioredis
    client = aioredis.from_url(redis_url, decode_responses=False)
    await client.ping()
    _instance = SessionMemory(client, brain_url, max_raw_turns, session_ttl)
    logger.info("SessionMemory connected to Redis: %s", redis_url)
    return _instance


async def close():
    global _instance
    if _instance:
        await _instance.redis.aclose()
        _instance = None


def get() -> Optional[SessionMemory]:
    return _instance
