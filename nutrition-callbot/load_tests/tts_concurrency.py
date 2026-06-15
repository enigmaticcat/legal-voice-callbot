from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Iterable
from urllib.parse import urlparse


DEFAULT_TEXTS = [
    "Chào bạn, tôi có thể hỗ trợ bạn về dinh dưỡng.",
    "Người trưởng thành nên duy trì chế độ ăn cân bằng, tăng rau xanh, trái cây, ngũ cốc nguyên hạt và hạn chế đồ uống nhiều đường.",
    "Với người có nguy cơ thừa cân, việc kiểm soát khẩu phần, ưu tiên thực phẩm giàu chất xơ và duy trì vận động đều đặn thường quan trọng hơn việc kiêng khem cực đoan.",
]


@dataclass
class Sample:
    concurrency: int
    index: int
    ok: bool
    status: int | None
    ttfa_ms: float | None
    total_ms: float | None
    bytes_received: int
    audio_duration_s: float | None
    rtf: float | None
    error: str | None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    weight = pos - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def summarize(samples: list[Sample]) -> dict[str, object]:
    ok_samples = [sample for sample in samples if sample.ok]
    ttfa = [sample.ttfa_ms for sample in ok_samples if sample.ttfa_ms is not None]
    total = [sample.total_ms for sample in ok_samples if sample.total_ms is not None]
    rtf = [sample.rtf for sample in ok_samples if sample.rtf is not None]
    return {
        "requests": len(samples),
        "ok": len(ok_samples),
        "errors": len(samples) - len(ok_samples),
        "error_rate": round((len(samples) - len(ok_samples)) / len(samples), 4) if samples else 0,
        "ttfa_ms": stats(ttfa),
        "total_ms": stats(total),
        "rtf": stats(rtf),
    }


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "mean": round(statistics.mean(values), 2),
        "p50": round(percentile(values, 0.50) or 0, 2),
        "p90": round(percentile(values, 0.90) or 0, 2),
        "p95": round(percentile(values, 0.95) or 0, 2),
        "p99": round(percentile(values, 0.99) or 0, 2),
        "max": round(max(values), 2),
    }


def post_stream(url: str, text: str, session_id: str, timeout_s: float, index: int, concurrency: int) -> Sample:
    parsed = urlparse(url)
    body = json.dumps({"text": text, "session_id": session_id}, ensure_ascii=False).encode("utf-8")
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    port = parsed.port
    host = parsed.hostname or "localhost"
    path = parsed.path or "/speak/stream"
    if parsed.query:
        path += f"?{parsed.query}"

    started = time.perf_counter()
    first_byte_at: float | None = None
    bytes_received = 0
    status: int | None = None

    try:
        conn = conn_cls(host, port=port, timeout=timeout_s)
        conn.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        response = conn.getresponse()
        status = response.status
        while True:
            chunk = response.read(4096)
            if not chunk:
                break
            if first_byte_at is None:
                first_byte_at = time.perf_counter()
            bytes_received += len(chunk)
        finished = time.perf_counter()
        conn.close()

        ok = 200 <= status < 300 and bytes_received > 0 and first_byte_at is not None
        total_s = finished - started
        audio_duration_s = bytes_received / 2 / 24000 if bytes_received else None
        return Sample(
            concurrency=concurrency,
            index=index,
            ok=ok,
            status=status,
            ttfa_ms=((first_byte_at - started) * 1000) if first_byte_at else None,
            total_ms=total_s * 1000,
            bytes_received=bytes_received,
            audio_duration_s=audio_duration_s,
            rtf=(total_s / audio_duration_s) if audio_duration_s else None,
            error=None if ok else "empty_or_non_2xx_response",
        )
    except Exception as exc:
        return Sample(
            concurrency=concurrency,
            index=index,
            ok=False,
            status=status,
            ttfa_ms=None,
            total_ms=(time.perf_counter() - started) * 1000,
            bytes_received=bytes_received,
            audio_duration_s=None,
            rtf=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_level(url: str, concurrency: int, requests: int, timeout_s: float, texts: list[str]) -> list[Sample]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                post_stream,
                url,
                texts[index % len(texts)],
                f"tts-load-{concurrency}-{index}",
                timeout_s,
                index,
                concurrency,
            )
            for index in range(requests)
        ]
        return [future.result() for future in concurrent.futures.as_completed(futures)]


def parse_levels(raw: str) -> list[int]:
    levels = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            levels.append(int(item))
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(description="TTS concurrent TTFA benchmark.")
    parser.add_argument("--url", default="http://localhost:50053/speak/stream")
    parser.add_argument("--concurrency", default="1,2,4,8,10,16")
    parser.add_argument("--requests-per-level", type=int, default=30)
    parser.add_argument("--timeout-s", type=float, default=120)
    parser.add_argument("--output", default="tts_concurrency_results.json")
    parser.add_argument("--text", action="append", help="Custom text. Can be passed multiple times.")
    args = parser.parse_args()

    texts = args.text or DEFAULT_TEXTS
    levels = parse_levels(args.concurrency)
    all_samples: list[Sample] = []
    summaries = {}

    for level in levels:
        print(f"\n== concurrency={level} requests={args.requests_per_level} ==")
        samples = run_level(args.url, level, args.requests_per_level, args.timeout_s, texts)
        summary = summarize(samples)
        summaries[str(level)] = summary
        all_samples.extend(samples)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    output = {
        "url": args.url,
        "levels": levels,
        "requests_per_level": args.requests_per_level,
        "summaries": summaries,
        "samples": [asdict(sample) for sample in all_samples],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
