"""
Sinh 1000 WAV clips cho MOS study (250 queries × 4 conditions).

Yêu cầu: Docker stack đang chạy
  cd nutrition-callbot
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

Chạy:
  python scripts/gen_tts_mos_clips.py

Output:
  evaluation/mos_wav/
    syn_xxx_0001_chunk_20.wav
    syn_xxx_0001_chunk_40.wav
    syn_xxx_0001_chunk_80.wav
    syn_xxx_0001_full.wav
    ...
  evaluation/mos_latency.jsonl
"""

import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf

# ── Paths & config ─────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[1]
SYNTHETIC_QA  = ROOT / "evaluation" / "synthetic_qa.jsonl"
OUT_WAV_DIR  = ROOT / "evaluation" / "mos_wav"
LATENCY_PATH = ROOT / "evaluation" / "mos_latency.jsonl"

LLM_BASE_URL = "http://localhost:8080/v1"          # vLLM (docker-compose port)
LLM_MODEL    = "Qwen/Qwen3-4B-Instruct-2507"
TTS_URL      = "http://localhost:50053"
BRAIN_URL    = "http://localhost:50052"

N_QUERIES    = 250
SEED         = 42
SAMPLE_RATE  = 24_000

CONDITIONS = [
    ("chunk_20", 20),
    ("chunk_40", 40),
    ("chunk_80", 80),
    ("full",     None),
]

SAMPLE_COUNTS = {
    "benhvienthucuc": 78,
    "viendinhduong":  72,
    "suckhoedoisong": 51,
    "vinmec":         49,
}

# ── System prompt (từ brain/core/prompt.py) ────────────────────────────────────
SYSTEM_PROMPT = """Bạn là chuyên gia tư vấn dinh dưỡng qua giọng nói. Tuân thủ:

1. **Dựa vào tài liệu**: Trả lời dựa trên thông tin dinh dưỡng được cung cấp. Không trích dẫn tên nguồn hay URL trong câu trả lời.
2. **Phong cách bác sĩ**: Bắt đầu bằng "Chào bạn,", tư vấn như chuyên gia dinh dưỡng.
3. **Ngắn gọn, dễ nghe**: Câu trả lời sẽ được đọc thành giọng nói — tối đa 150 từ, dùng câu ngắn, không dùng bullet points hay danh sách. Sau mỗi dấu chấm hoặc dấu phẩy phải có dấu cách.
4. **Trung thực**: Nếu không có thông tin → nói rõ "Tôi không có thông tin về vấn đề này".
5. **Disclaimer**: Kết thúc bằng "Để được tư vấn chính xác, bạn nên gặp bác sĩ dinh dưỡng."
"""

# ── Chunker (copy từ tts/core/chunker.py) ─────────────────────────────────────
_SENT_END   = re.compile(r"[.!?]")
_NEWLINE_RE = re.compile(r"\n+")


def chunk_text(text: str, min_size: int) -> list[str]:
    chunks, buffer = [], ""
    for char in text:
        buffer += char
        if _SENT_END.match(char) and len(buffer.strip()) >= min_size:
            chunks.append(buffer)
            buffer = ""
    if buffer.strip():
        chunks.append(buffer)
    return chunks


# ── LLM streaming (gọi vLLM OpenAI-compatible) ────────────────────────────────
async def llm_stream(session: aiohttp.ClientSession, query: str):
    """Yield text tokens từ vLLM."""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ],
        "temperature": 0.3,
        "max_tokens":  400,
        "stream":      True,
        "extra_body":  {"chat_template_kwargs": {"enable_thinking": False}},
    }
    async with session.post(
        f"{LLM_BASE_URL}/chat/completions",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                text = chunk["choices"][0]["delta"].get("content", "")
                if text:
                    yield text
            except (json.JSONDecodeError, KeyError):
                continue


# ── TTS streaming (gọi /speak/stream) ─────────────────────────────────────────
async def tts_speak(session: aiohttp.ClientSession, text: str, session_id: str) -> tuple[bytes, float]:
    """
    Gọi /speak/stream → thu thập toàn bộ PCM bytes + đo TTFA.
    Trả về (pcm_bytes, ttfa_ms).
    """
    t0 = time.perf_counter()
    ttfa_ms = None
    chunks = []

    async with session.post(
        f"{TTS_URL}/speak/stream",
        json={"text": text, "session_id": session_id},
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        resp.raise_for_status()
        async for chunk in resp.content.iter_chunked(4096):
            if ttfa_ms is None:
                ttfa_ms = (time.perf_counter() - t0) * 1000
            chunks.append(chunk)

    return b"".join(chunks), round(ttfa_ms or 0, 1)


# ── Pipeline: LLM stream → buffer → TTS per chunk ─────────────────────────────
async def run_pipeline(session: aiohttp.ClientSession, query: str,
                        min_size: int | None, session_id: str) -> dict:
    """
    Với min_size=None: thu full LLM response → 1 lần TTS.
    Với min_size=N:    buffer token → flush khi đủ min_size + .!? → TTS ngay.

    Trả về dict với: full_text, pcm_all, ttft_ms, ttfa_ms, total_ms, n_chunks.
    """
    start = time.perf_counter()
    ttft_ms = None
    ttfa_ms = None
    pcm_all = b""
    n_chunks = 0
    full_parts = []
    buffer = ""

    async for token in llm_stream(session, query):
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - start) * 1000
        full_parts.append(token)
        buffer += token

        if min_size is not None and _SENT_END.search(buffer) and len(buffer.strip()) >= min_size:
            chunk_text_str = _NEWLINE_RE.sub(" ", buffer).strip()
            buffer = ""
            t_before_tts = time.perf_counter()
            pcm, tfa = await tts_speak(session, chunk_text_str, session_id)
            if ttfa_ms is None:
                ttfa_ms = (t_before_tts - start) * 1000 + tfa
            pcm_all += pcm
            n_chunks += 1

    full_text = _NEWLINE_RE.sub(" ", "".join(full_parts)).strip()

    # Flush phần còn lại (hoặc toàn bộ nếu full)
    remaining = buffer.strip() if min_size is not None else full_text
    if remaining:
        pcm, tfa = await tts_speak(session, remaining, session_id + "_tail")
        if ttfa_ms is None:
            ttfa_ms = ttft_ms + tfa if ttft_ms else tfa
        pcm_all += pcm
        n_chunks += 1

    total_ms = (time.perf_counter() - start) * 1000

    return {
        "full_text": full_text,
        "pcm_all":   pcm_all,
        "ttft_ms":   round(ttft_ms, 1) if ttft_ms else None,
        "ttfa_ms":   round(ttfa_ms, 1) if ttfa_ms else None,
        "total_ms":  round(total_ms, 1),
        "n_chunks":  n_chunks,
    }


# ── Save WAV ───────────────────────────────────────────────────────────────────
def save_wav(pcm_bytes: bytes, path: Path) -> float:
    if not pcm_bytes:
        return 0.0
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    sf.write(str(path), audio, SAMPLE_RATE)
    return len(audio) / SAMPLE_RATE


# ── Load & stratified sample queries ──────────────────────────────────────────
def load_queries() -> list[dict]:
    with open(SYNTHETIC_QA, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    by_source: dict[str, list] = {}
    for r in records:
        by_source.setdefault(r["source"], []).append(r)

    random.seed(SEED)
    selected = []
    for src, n in SAMPLE_COUNTS.items():
        selected.extend(random.sample(by_source[src], n))
    random.shuffle(selected)
    assert len(selected) == N_QUERIES
    return selected


# ── Health check ───────────────────────────────────────────────────────────────
async def health_check(session: aiohttp.ClientSession):
    errors = []
    for name, url in [("TTS", f"{TTS_URL}/health"), ("LLM", f"{LLM_BASE_URL}/models")]:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                r.raise_for_status()
                print(f"  {name}: OK")
        except Exception as e:
            errors.append(f"{name} not reachable at {url}: {e}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────────────────
async def main():
    queries = load_queries()
    print(f"Loaded {len(queries)} queries")

    # Setup dirs
    OUT_WAV_DIR.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        print("\nChecking services...")
        await health_check(session)

        all_results = []

        for cond_name, min_size in CONDITIONS:
            print(f"\n{'='*50}")
            print(f"Condition: {cond_name}  (min_size={min_size})")
            print(f"{'='*50}")
            cond_dir = OUT_WAV_DIR / cond_name

            for q_idx, q in enumerate(queries):
                query_id   = q["id"]
                question   = q["question"]
                session_id = f"{cond_name}_{query_id}"
                wav_name   = f"{query_id}_{cond_name}.wav"
                wav_path   = OUT_WAV_DIR / wav_name

                try:
                    result = await run_pipeline(session, question, min_size, session_id)
                    duration_s = save_wav(result["pcm_all"], wav_path)

                    row = {
                        "condition":  cond_name,
                        "query_id":   query_id,
                        "question":   question,
                        "n_chunks":   result["n_chunks"],
                        "ttft_ms":    result["ttft_ms"],
                        "ttfa_ms":    result["ttfa_ms"],
                        "total_ms":   result["total_ms"],
                        "duration_s": round(duration_s, 2),
                        "wav_path":   str(wav_path),
                    }
                    all_results.append(row)

                    print(
                        f"  [{q_idx+1:03d}/{N_QUERIES}] {query_id}"
                        f"  chunks={result['n_chunks']}"
                        f"  TTFT={result['ttft_ms']}ms"
                        f"  TTFA={result['ttfa_ms']}ms"
                        f"  Total={result['total_ms']:.0f}ms"
                        f"  dur={duration_s:.1f}s"
                    )

                except Exception as e:
                    print(f"  [{q_idx+1:03d}] ERROR {query_id}: {e}")

    # Save latency JSONL
    with open(LATENCY_PATH, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nLatency saved: {LATENCY_PATH}")

    # Summary
    from collections import Counter
    total_ok = sum(1 for r in all_results if Path(r["wav_path"]).exists())
    print(f"WAV files: {total_ok} / {len(all_results)}")
    counts = Counter(r["condition"] for r in all_results)
    for c, _ in CONDITIONS:
        print(f"  {c}: {counts.get(c, 0)} clips")
    print(f"Output dir: {OUT_WAV_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
