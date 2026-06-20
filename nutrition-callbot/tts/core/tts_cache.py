from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

logger = logging.getLogger("tts.core.tts_cache")


class TTSCache:
    def __init__(
        self,
        redis_url: str,
        enabled: bool = True,
        required: bool = False,
        ttl_seconds: int = 604800,
        max_bytes: int = 8_000_000,
        version: str = "v1",
    ):
        self.redis_url = redis_url
        self.enabled = enabled
        self.required = required
        self.ttl_seconds = ttl_seconds
        self.max_bytes = max_bytes
        self.version = version
        self._redis = None
        self._prefix = "cache:tts"

    @property
    def connected(self) -> bool:
        return self._redis is not None

    async def connect(self) -> None:
        if not self.enabled:
            return
        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(self.redis_url, decode_responses=False)
            await self._redis.ping()
            logger.info("TTS cache connected: %s", self.redis_url)
        except Exception:
            self._redis = None
            if self.required:
                raise
            logger.warning("TTS cache unavailable; continuing without cache", exc_info=True)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    def build_key(
        self,
        text: str,
        backbone_repo: str,
        codec_repo: str,
        sample_rate: int,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str:
        normalized = " ".join((text or "").split())
        material = "\n".join([
            normalized,
            backbone_repo,
            codec_repo,
            str(sample_rate),
            str(temperature),
            str(top_k),
            self.version,
        ])
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return f"{self._prefix}:{self.version}:{digest}"

    async def get(self, key: str) -> tuple[bytes | None, dict[str, Any]]:
        if self._redis is None:
            return None, {"status": "disabled", "key": key[-12:]}
        started = time.perf_counter()
        try:
            values = await self._redis.hmget(key, "pcm", "compute_ms")
            lookup_ms = (time.perf_counter() - started) * 1000
            pcm, compute_ms_raw = values
            if pcm is None:
                await self._redis.hincrby(f"{self._prefix}:stats", "misses", 1)
                return None, {
                    "status": "miss",
                    "key": key[-12:],
                    "lookup_ms": round(lookup_ms, 2),
                }

            compute_ms = float(compute_ms_raw or 0)
            stats_key = f"{self._prefix}:stats"
            await self._redis.hincrby(stats_key, "hits", 1)
            await self._redis.hincrby(stats_key, "bytes_served", len(pcm))
            if compute_ms > 0:
                await self._redis.hincrbyfloat(
                    stats_key, "estimated_saved_ms", compute_ms
                )
            logger.info(
                "TTS cache HIT key=%s bytes=%d saved_ms=%.1f",
                key[-12:],
                len(pcm),
                compute_ms,
            )
            return pcm, {
                "status": "hit",
                "key": key[-12:],
                "lookup_ms": round(lookup_ms, 2),
                "estimated_saved_ms": round(compute_ms, 1),
            }
        except Exception:
            await self._record_error()
            logger.warning("TTS cache read failed", exc_info=True)
            return None, {"status": "error", "key": key[-12:]}

    async def set(self, key: str, pcm: bytes, compute_ms: float) -> bool:
        if self._redis is None or not pcm or len(pcm) > self.max_bytes:
            return False
        try:
            await self._redis.hset(
                key,
                mapping={
                    "pcm": pcm,
                    "compute_ms": str(round(compute_ms, 1)),
                    "created_at": str(int(time.time())),
                },
            )
            await self._redis.expire(key, self.ttl_seconds)
            stats_key = f"{self._prefix}:stats"
            await self._redis.hincrby(stats_key, "writes", 1)
            await self._redis.hincrby(stats_key, "bytes_written", len(pcm))
            logger.info(
                "TTS cache WRITE key=%s bytes=%d ttl=%ds",
                key[-12:],
                len(pcm),
                self.ttl_seconds,
            )
            return True
        except Exception:
            await self._record_error()
            logger.warning("TTS cache write failed", exc_info=True)
            return False

    async def stats(self) -> dict[str, Any]:
        if self._redis is None:
            return {"enabled": self.enabled, "connected": False}
        raw = await self._redis.hgetall(f"{self._prefix}:stats")
        values = {
            key.decode(): value.decode()
            for key, value in raw.items()
        }
        hits = int(float(values.get("hits", 0)))
        misses = int(float(values.get("misses", 0)))
        total = hits + misses
        return {
            "enabled": self.enabled,
            "connected": True,
            "version": self.version,
            "ttl_seconds": self.ttl_seconds,
            "max_bytes": self.max_bytes,
            "hits": hits,
            "misses": misses,
            "writes": int(float(values.get("writes", 0))),
            "errors": int(float(values.get("errors", 0))),
            "bytes_written": int(float(values.get("bytes_written", 0))),
            "bytes_served": int(float(values.get("bytes_served", 0))),
            "estimated_saved_ms": round(
                float(values.get("estimated_saved_ms", 0)), 1
            ),
            "hit_rate": round(hits / total, 4) if total else 0.0,
        }

    async def clear(self) -> int:
        if self._redis is None:
            return 0
        deleted = 0
        async for key in self._redis.scan_iter(match=f"{self._prefix}:*"):
            deleted += await self._redis.delete(key)
        return deleted

    async def _record_error(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.hincrby(f"{self._prefix}:stats", "errors", 1)
            except Exception:
                pass
