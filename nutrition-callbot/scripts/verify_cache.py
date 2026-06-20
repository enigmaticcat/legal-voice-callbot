#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def request_json(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))
    result["_request_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


def request_pcm(url: str, text: str) -> dict:
    payload = json.dumps(
        {"text": text, "session_id": "cache-verification"},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        pcm = response.read()
        return {
            "status": response.headers.get("X-TTS-Cache"),
            "key": response.headers.get("X-TTS-Cache-Key"),
            "estimated_saved_ms": response.headers.get("X-TTS-Estimated-Saved-ms"),
            "bytes": len(pcm),
            "request_ms": round((time.perf_counter() - started) * 1000, 1),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify retrieval and TTS Redis caches with repeated requests."
    )
    parser.add_argument("--brain-url", default="http://localhost:50052")
    parser.add_argument("--tts-url", default="http://localhost:50053")
    parser.add_argument(
        "--query",
        default="Người bị tiểu đường nên lựa chọn thực phẩm như thế nào?",
    )
    parser.add_argument(
        "--semantic-query",
        default="Bệnh nhân đái tháo đường nên lựa chọn thực phẩm nào?",
    )
    parser.add_argument(
        "--tts-text",
        default="Chào bạn, để được tư vấn chính xác, bạn nên gặp bác sĩ dinh dưỡng.",
    )
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()

    if not args.no_clear:
        request_json(f"{args.brain_url}/cache", method="DELETE")
        request_json(f"{args.tts_url}/cache", method="DELETE")

    brain_payload = {
        "query": args.query,
        "session_id": "cache-verification",
        "conversation_history": [],
    }
    brain_first = request_json(
        f"{args.brain_url}/think", method="POST", payload=brain_payload
    )
    brain_second = request_json(
        f"{args.brain_url}/think", method="POST", payload=brain_payload
    )
    semantic_payload = {
        **brain_payload,
        "query": args.semantic_query,
        "session_id": "semantic-cache-verification",
    }
    brain_semantic = request_json(
        f"{args.brain_url}/think", method="POST", payload=semantic_payload
    )

    tts_first = request_pcm(f"{args.tts_url}/speak/stream", args.tts_text)
    tts_second = request_pcm(f"{args.tts_url}/speak/stream", args.tts_text)

    result = {
        "retrieval": {
            "first": {
                **brain_first.get("retrieval_cache", {}),
                "request_ms": brain_first.get("_request_ms"),
                "rag_ms": brain_first.get("timing", {}).get("rag_ms"),
            },
            "second": {
                **brain_second.get("retrieval_cache", {}),
                "request_ms": brain_second.get("_request_ms"),
                "rag_ms": brain_second.get("timing", {}).get("rag_ms"),
            },
            "semantic": {
                **brain_semantic.get("retrieval_cache", {}),
                "request_ms": brain_semantic.get("_request_ms"),
                "rag_ms": brain_semantic.get("timing", {}).get("rag_ms"),
            },
            "stats": request_json(f"{args.brain_url}/cache/stats"),
        },
        "tts": {
            "first": tts_first,
            "second": tts_second,
            "stats": request_json(f"{args.tts_url}/cache/stats"),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    retrieval_ok = (
        result["retrieval"]["first"].get("status") == "miss"
        and result["retrieval"]["second"].get("status") == "hit"
    )
    semantic_ok = (
        result["retrieval"]["semantic"].get("status") == "semantic_hit"
    )
    tts_ok = (
        result["tts"]["first"].get("status") == "MISS"
        and result["tts"]["second"].get("status") == "HIT"
    )
    if not retrieval_ok or not semantic_ok or not tts_ok:
        raise SystemExit("Cache verification failed")


if __name__ == "__main__":
    main()
