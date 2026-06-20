# Redis Cache Verification

The system exposes evidence for both caches:

- Brain response field `retrieval_cache.status`: `miss`, then `hit`.
- TTS response header `X-TTS-Cache`: `MISS`, then `HIT`.
- Persistent counters at `GET /cache/stats`.
- Redis keys under `cache:retrieval:*` and `cache:tts:*`.

After starting the stack, run:

```bash
python3 scripts/verify_cache.py
```

Expected result:

```text
retrieval.first.status = miss
retrieval.second.status = hit
tts.first.status = MISS
tts.second.status = HIT
```

The JSON output also includes `request_ms`, `rag_ms`, and
`estimated_saved_ms`, allowing a cold-cache and warm-cache latency comparison.

Inspect cumulative statistics:

```bash
curl -s http://localhost:50052/cache/stats | python3 -m json.tool
curl -s http://localhost:50053/cache/stats | python3 -m json.tool
```

Inspect Redis directly:

```bash
docker compose exec redis redis-cli HGETALL cache:retrieval:stats
docker compose exec redis redis-cli HGETALL cache:tts:stats
docker compose exec redis redis-cli --scan --pattern 'cache:retrieval:*'
docker compose exec redis redis-cli --scan --pattern 'cache:tts:*'
```

For an experiment, clear both caches before each cold/warm comparison:

```bash
curl -X DELETE http://localhost:50052/cache
curl -X DELETE http://localhost:50053/cache
```

The retrieval cache reports estimated time saved from the original embedding,
Qdrant search and reranking duration. The TTS cache reports estimated synthesis
time saved and the number of PCM bytes served from Redis.
