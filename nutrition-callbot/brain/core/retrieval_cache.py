from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger("brain.core.retrieval_cache")


class RetrievalCache:
    def __init__(
        self,
        redis_url: str,
        enabled: bool = True,
        required: bool = False,
        ttl_seconds: int = 86400,
        corpus_version: str = "v1",
    ):
        self.redis_url = redis_url
        self.enabled = enabled
        self.required = required
        self.ttl_seconds = ttl_seconds
        self.corpus_version = corpus_version
        self._redis = None
        self._prefix = "cache:retrieval"

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
            logger.info("Retrieval cache connected: %s", self.redis_url)
        except Exception:
            self._redis = None
            if self.required:
                raise
            logger.warning("Retrieval cache unavailable; continuing without cache", exc_info=True)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    def build_key(
        self,
        query: str,
        embedding_model: str,
        reranker_model: str,
        fetch_k: int,
        top_k: int,
        use_hyde: bool,
    ) -> str:
        payload = {
            "query": " ".join((query or "").lower().split()),
            "embedding_model": embedding_model,
            "reranker_model": reranker_model,
            "fetch_k": fetch_k,
            "top_k": top_k,
            "use_hyde": use_hyde,
            "corpus_version": self.corpus_version,
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"{self._prefix}:{self.corpus_version}:{digest}"

    async def get(self, key: str) -> tuple[list[dict] | None, dict[str, Any]]:
        if self._redis is None:
            return None, {"status": "disabled", "key": key[-12:]}

        started = time.perf_counter()
        try:
            raw = await self._redis.get(key)
            lookup_ms = (time.perf_counter() - started) * 1000
            if raw is None:
                await self._redis.hincrby(f"{self._prefix}:stats", "misses", 1)
                return None, {
                    "status": "miss",
                    "key": key[-12:],
                    "lookup_ms": round(lookup_ms, 2),
                }

            payload = json.loads(raw)
            await self._redis.hincrby(f"{self._prefix}:stats", "hits", 1)
            saved_ms = float(payload.get("compute_ms", 0))
            if saved_ms > 0:
                await self._redis.hincrbyfloat(
                    f"{self._prefix}:stats", "estimated_saved_ms", saved_ms
                )
            logger.info(
                "Retrieval cache HIT key=%s docs=%d saved_ms=%.1f",
                key[-12:],
                len(payload.get("documents", [])),
                saved_ms,
            )
            return payload.get("documents", []), {
                "status": "hit",
                "key": key[-12:],
                "lookup_ms": round(lookup_ms, 2),
                "estimated_saved_ms": round(saved_ms, 1),
            }
        except Exception:
            await self._record_error()
            logger.warning("Retrieval cache read failed", exc_info=True)
            return None, {"status": "error", "key": key[-12:]}

    async def set(self, key: str, documents: list[dict], compute_ms: float) -> None:
        if self._redis is None:
            return
        payload = {
            "documents": documents,
            "compute_ms": round(compute_ms, 1),
            "created_at": int(time.time()),
            "corpus_version": self.corpus_version,
        }
        try:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            await self._redis.setex(key, self.ttl_seconds, encoded)
            stats_key = f"{self._prefix}:stats"
            await self._redis.hincrby(stats_key, "writes", 1)
            await self._redis.hincrby(stats_key, "bytes_written", len(encoded))
            logger.info(
                "Retrieval cache WRITE key=%s docs=%d ttl=%ds",
                key[-12:],
                len(documents),
                self.ttl_seconds,
            )
        except Exception:
            await self._record_error()
            logger.warning("Retrieval cache write failed", exc_info=True)

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
            "corpus_version": self.corpus_version,
            "ttl_seconds": self.ttl_seconds,
            "hits": hits,
            "misses": misses,
            "writes": int(float(values.get("writes", 0))),
            "errors": int(float(values.get("errors", 0))),
            "bytes_written": int(float(values.get("bytes_written", 0))),
            "estimated_saved_ms": round(
                float(values.get("estimated_saved_ms", 0)), 1
            ),
            "semantic_hits": int(float(values.get("semantic_hits", 0))),
            "semantic_misses": int(float(values.get("semantic_misses", 0))),
            "semantic_writes": int(float(values.get("semantic_writes", 0))),
            "semantic_estimated_saved_ms": round(
                float(values.get("semantic_estimated_saved_ms", 0)), 1
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

    async def record_semantic(
        self,
        status: str,
        estimated_saved_ms: float = 0,
    ) -> None:
        if self._redis is None:
            return
        try:
            stats_key = f"{self._prefix}:stats"
            await self._redis.hincrby(stats_key, f"semantic_{status}", 1)
            if status == "hits" and estimated_saved_ms > 0:
                await self._redis.hincrbyfloat(
                    stats_key,
                    "semantic_estimated_saved_ms",
                    estimated_saved_ms,
                )
        except Exception:
            await self._record_error()

    async def _record_error(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.hincrby(f"{self._prefix}:stats", "errors", 1)
            except Exception:
                pass
