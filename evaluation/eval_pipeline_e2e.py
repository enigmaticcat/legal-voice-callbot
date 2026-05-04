"""
Đánh giá latency end-to-end toàn bộ pipeline ASR → Brain → TTS.

Metric chính:
  speech_end_to_first_audio_ms = asr_ms + brain_first_flush_ms + tts_first_chunk_ms
  (= thời gian từ khi người dùng dừng nói đến khi nghe tiếng đầu tiên)

Usage:
  # Chạy thử 5 file từ eval_1
  python eval_pipeline_e2e.py --eval-dirs eval_1 --sample 5

  # Chạy toàn bộ tất cả eval sets, concurrency 2
  python eval_pipeline_e2e.py --concurrency 2

  # Chỉ định URL khác (Lightning AI port-forward)
  python eval_pipeline_e2e.py --asr-url http://localhost:50051 --brain-url http://localhost:50052 --tts-url http://localhost:50053
"""

import argparse
import asyncio
import json
import logging
import re
import time
import wave
from datetime import datetime
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("eval_e2e")

WAV_ROOT    = Path(__file__).parent.parent / "wav_16k"
RESULTS_DIR = Path(__file__).parent / "results"

_TTS_MIN_CHARS = 40
_TTS_PUNCTS    = {".", "?", "!", "\n"}


def _ready_for_tts(buf: str) -> bool:
    return len(buf.strip()) >= _TTS_MIN_CHARS or any(p in buf for p in _TTS_PUNCTS)


def _clean_for_tts(text: str) -> str:
    text = re.sub(r'\*{1,3}([^*]+?)\*{1,3}', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\n+', ' ', text)
    return re.sub(r' {2,}', ' ', text).strip()


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


def collect_wav_files(wav_root: Path, eval_dirs: list[str]) -> list[tuple[str, Path]]:
    """Trả về list (eval_set, wav_path)."""
    items = []
    for ed in eval_dirs:
        d = wav_root / ed
        if not d.is_dir():
            logger.warning("Không tìm thấy thư mục: %s", d)
            continue
        for wav in sorted(d.glob("*.wav")):
            items.append((ed, wav))
    return items


async def run_one(
    client: httpx.AsyncClient,
    eval_set: str,
    wav_path: Path,
    asr_url: str,
    brain_url: str,
    tts_url: str,
) -> dict:
    file_id = wav_path.stem
    wall_start = time.perf_counter()

    try:
        audio_bytes = wav_path.read_bytes()
        duration_s  = _wav_duration(wav_path)

        # ── 1) ASR ───────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        resp = await client.post(
            f"{asr_url}/transcribe",
            content=audio_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()
        asr_ms   = (time.perf_counter() - t0) * 1000
        transcript = resp.json().get("text", "").strip()

        # ── 2) Brain stream ──────────────────────────────────────────────────
        payload = {"query": transcript, "session_id": f"e2e-{file_id}", "conversation_history": []}
        brain_parts        = []
        brain_flushes      = []
        brain_timing       = {}
        brain_first_flush_ms = None
        buf = ""

        t0 = time.perf_counter()
        async with client.stream("POST", f"{brain_url}/think/stream", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = data.get("text", "")
                if text:
                    brain_parts.append(text)
                    buf += text
                    if _ready_for_tts(buf):
                        if brain_first_flush_ms is None:
                            brain_first_flush_ms = (time.perf_counter() - t0) * 1000
                        brain_flushes.append(buf)
                        buf = ""
                if data.get("timing"):
                    brain_timing.update(data["timing"])
                if data.get("is_final"):
                    break

        brain_total_ms = (time.perf_counter() - t0) * 1000
        if buf.strip():
            if brain_first_flush_ms is None:
                brain_first_flush_ms = brain_total_ms
            brain_flushes.append(buf)

        # ── 3) TTS ──────────────────────────────────────────────────────────
        tts_first_chunk_ms = None
        total_pcm_bytes    = 0

        t0 = time.perf_counter()
        for flush_text in brain_flushes:
            async with client.stream("POST", f"{tts_url}/speak/stream", json={"text": _clean_for_tts(flush_text)}) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(chunk_size=4800):
                    if chunk:
                        if tts_first_chunk_ms is None:
                            tts_first_chunk_ms = (time.perf_counter() - t0) * 1000
                        total_pcm_bytes += len(chunk)

        tts_total_ms = (time.perf_counter() - t0) * 1000

        wall_ms = (time.perf_counter() - wall_start) * 1000
        ttfb    = round(asr_ms + (brain_first_flush_ms or brain_total_ms) + (tts_first_chunk_ms or 0), 1)

        return {
            "id":           file_id,
            "eval_set":     eval_set,
            "audio_duration_s": round(duration_s, 3),
            "transcript":   transcript,
            "asr_ms":       round(asr_ms, 1),
            "brain_first_flush_ms": round(brain_first_flush_ms or brain_total_ms, 1),
            "brain_total_ms":  round(brain_total_ms, 1),
            "brain_timing":    brain_timing,
            "tts_first_chunk_ms": round(tts_first_chunk_ms or 0, 1),
            "tts_total_ms":    round(tts_total_ms, 1),
            "speech_end_to_first_audio_ms": ttfb,
            "wall_ms":       round(wall_ms, 1),
            "success":       True,
            "error":         None,
        }

    except Exception as e:
        wall_ms = (time.perf_counter() - wall_start) * 1000
        logger.error("[%s] FAILED: %s", file_id, e)
        return {
            "id":       wav_path.stem,
            "eval_set": eval_set,
            "success":  False,
            "error":    str(e),
            "wall_ms":  round(wall_ms, 1),
        }


async def run_all(
    items: list[tuple[str, Path]],
    asr_url: str,
    brain_url: str,
    tts_url: str,
    concurrency: int,
) -> list[dict]:
    sem     = asyncio.Semaphore(concurrency)
    results = [None] * len(items)

    async def _one(idx: int, eval_set: str, wav_path: Path):
        async with sem:
            timeout = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=30.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await run_one(client, eval_set, wav_path, asr_url, brain_url, tts_url)
            status = "OK" if r["success"] else "ERR"
            ttfb   = r.get("speech_end_to_first_audio_ms", "-")
            asr    = r.get("asr_ms", "-")
            brain  = r.get("brain_first_flush_ms", "-")
            tts    = r.get("tts_first_chunk_ms", "-")
            logger.info(
                "[%3d/%d] %s | %s | TTFB=%sms (asr=%s brain=%s tts=%s)",
                idx + 1, len(items), status, wav_path.stem, ttfb, asr, brain, tts,
            )
            results[idx] = r

    await asyncio.gather(*[_one(i, es, wp) for i, (es, wp) in enumerate(items)])
    return results


def _stat(values: list[float]) -> dict:
    if not values:
        return {}
    values = sorted(values)
    n = len(values)
    return {
        "mean":   round(sum(values) / n, 1),
        "median": round(values[n // 2], 1),
        "p90":    round(values[int(n * 0.9)], 1),
        "p95":    round(values[int(n * 0.95)], 1),
        "min":    round(values[0], 1),
        "max":    round(values[-1], 1),
    }


def compute_metrics(results: list[dict]) -> dict:
    ok  = [r for r in results if r.get("success")]
    err = [r for r in results if not r.get("success")]

    return {
        "total":         len(results),
        "success":       len(ok),
        "error":         len(err),
        "success_rate":  round(100 * len(ok) / len(results), 1) if results else 0,
        "speech_end_to_first_audio_ms": _stat([r["speech_end_to_first_audio_ms"] for r in ok]),
        "asr_ms":                 _stat([r["asr_ms"] for r in ok]),
        "brain_first_flush_ms":   _stat([r["brain_first_flush_ms"] for r in ok]),
        "brain_total_ms":         _stat([r["brain_total_ms"] for r in ok]),
        "tts_first_chunk_ms":     _stat([r["tts_first_chunk_ms"] for r in ok if r.get("tts_first_chunk_ms", 0) > 0]),
        "tts_total_ms":           _stat([r["tts_total_ms"] for r in ok]),
        "by_eval_set": {
            es: _stat([r["speech_end_to_first_audio_ms"] for r in ok if r["eval_set"] == es])
            for es in sorted({r["eval_set"] for r in ok})
        },
        "errors": [{"id": r["id"], "error": r.get("error")} for r in err],
    }


def print_summary(metrics: dict):
    print("\n" + "=" * 65)
    print("KẾT QUẢ ĐÁNH GIÁ PIPELINE END-TO-END (ASR → Brain → TTS)")
    print("=" * 65)
    print(f"  Tổng file   : {metrics['total']}")
    print(f"  Thành công  : {metrics['success']} ({metrics['success_rate']}%)")
    print(f"  Lỗi         : {metrics['error']}")

    def row(label, stat, unit="ms"):
        if not stat:
            return
        print(f"  {label:<28}: mean={stat['mean']}{unit}  p90={stat['p90']}{unit}  max={stat['max']}{unit}")

    print("\n  ── KEY METRIC (speech_end → first_audio) ──────────────────")
    row("TTFB (end-to-end)", metrics["speech_end_to_first_audio_ms"])
    print("    (= ASR + Brain first flush + TTS first chunk)")

    print("\n  ── Breakdown ──────────────────────────────────────────────")
    row("ASR", metrics["asr_ms"])
    row("Brain first flush", metrics["brain_first_flush_ms"])
    row("Brain total", metrics["brain_total_ms"])
    row("TTS first chunk", metrics["tts_first_chunk_ms"])
    row("TTS total", metrics["tts_total_ms"])

    if metrics["by_eval_set"]:
        print("\n  ── TTFB theo eval set ─────────────────────────────────────")
        for es, stat in metrics["by_eval_set"].items():
            if stat:
                print(f"  {es:<28}: mean={stat['mean']}ms  p90={stat['p90']}ms")

    if metrics["errors"]:
        print(f"\n  ── Lỗi ({len(metrics['errors'])} file) ──────────────────────────────")
        for e in metrics["errors"][:5]:
            print(f"  [{e['id']}] {e['error']}")
    print("=" * 65 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Eval pipeline ASR→Brain→TTS trên wav_16k")
    parser.add_argument("--asr-url",   default="http://localhost:50051")
    parser.add_argument("--brain-url", default="http://localhost:50052")
    parser.add_argument("--tts-url",   default="http://localhost:50053")
    parser.add_argument("--eval-dirs", nargs="+", default=["eval_1", "eval_2", "eval_3", "eval_4", "eval_5"],
                        help="Danh sách eval dirs trong wav_16k (mặc định: tất cả)")
    parser.add_argument("--sample",      type=int, default=None, help="Giới hạn số file (mặc định: tất cả)")
    parser.add_argument("--concurrency", type=int, default=1,    help="Số request song song (mặc định: 1)")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    # Health check
    async with httpx.AsyncClient(timeout=10) as client:
        for name, url in [("ASR", args.asr_url), ("Brain", args.brain_url), ("TTS", args.tts_url)]:
            try:
                r = await client.get(f"{url}/health")
                r.raise_for_status()
                logger.info("%s health OK: %s", name, url)
            except Exception as e:
                logger.error("%s không phản hồi tại %s: %s", name, url, e)
                return

    items = collect_wav_files(WAV_ROOT, args.eval_dirs)
    if not items:
        logger.error("Không tìm thấy file WAV nào trong %s", WAV_ROOT)
        return

    if args.sample:
        items = items[: args.sample]

    logger.info("Đánh giá %d file | concurrency=%d", len(items), args.concurrency)

    t_start = time.time()
    results = await run_all(items, args.asr_url, args.brain_url, args.tts_url, args.concurrency)
    elapsed = time.time() - t_start
    logger.info("Hoàn tất trong %.1fs", elapsed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.sample:
        tag += f"_n{args.sample}"

    results_path = out_dir / f"e2e_results_{tag}.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics = compute_metrics(results)
    metrics_path = out_dir / f"e2e_metrics_{tag}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print_summary(metrics)
    print(f"File kết quả : {results_path}")
    print(f"File metrics : {metrics_path}")


if __name__ == "__main__":
    asyncio.run(main())
