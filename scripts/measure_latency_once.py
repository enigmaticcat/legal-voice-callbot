"""
Đo latency end-to-end: speech_end → first_audio_byte

Pipeline:
  WAV file → ASR HTTP POST → Brain stream → TTS stream → output.wav

Metric quan trọng:
  speech_end_to_first_audio_ms = asr_ms + brain_first_flush_ms + tts_first_chunk_ms
"""
import json
import re
import time
import wave
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).parent.parent
AUDIO_PATH = _REPO_ROOT / "wav_16k" / "eval_1" / "thucuc_s1_003.wav"
OUT_WAV    = _REPO_ROOT / "latency_final_audio.wav"

ASR_URL   = "http://localhost:50051/transcribe"
BRAIN_URL = "http://localhost:50052/think/stream"
TTS_URL   = "http://localhost:50053/speak/stream"

_TTS_MIN_CHARS = 40
_TTS_PUNCTS = {".", "?", "!", "\n"}


def _ready_for_tts(buf: str) -> bool:
    return len(buf.strip()) >= _TTS_MIN_CHARS or any(p in buf for p in _TTS_PUNCTS)


def _clean_for_tts(text: str) -> str:
    return re.sub(r" {2,}", " ", re.sub(r"\n+", " ", text)).strip()


def main() -> None:
    audio_bytes = AUDIO_PATH.read_bytes()
    with wave.open(str(AUDIO_PATH), "rb") as f:
        duration_s = f.getnframes() / f.getframerate()

    result = {
        "input_audio": AUDIO_PATH.name,
        "audio_duration_s": round(duration_s, 3),
    }

    with httpx.Client(timeout=120) as client:

        # ── 1) ASR ────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        resp = client.post(
            ASR_URL,
            content=audio_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        resp.raise_for_status()
        asr_ms = (time.perf_counter() - t0) * 1000
        transcript = resp.json().get("text", "")

        result["asr_ms"] = round(asr_ms, 1)
        result["transcript"] = transcript

        # ── 2) Brain ──────────────────────────────────────────────────────────
        payload = {"query": transcript, "session_id": "latency-check", "conversation_history": []}
        brain_parts, brain_flushes, brain_timing = [], [], {}
        brain_first_flush_ms = None
        buf = ""

        t0 = time.perf_counter()
        with client.stream("POST", BRAIN_URL, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line if isinstance(line, str) else line.decode())
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

        result["brain_first_flush_ms"] = round(brain_first_flush_ms or 0, 1)
        result["brain_total_ms"]       = round(brain_total_ms, 1)
        result["brain_timing"]         = brain_timing

        # ── 3) TTS ────────────────────────────────────────────────────────────
        all_pcm = bytearray()
        tts_first_chunk_ms = None
        sample_rate = 24000

        t0 = time.perf_counter()
        for flush_text in brain_flushes:
            with client.stream("POST", TTS_URL, json={"text": _clean_for_tts(flush_text)}) as resp:
                resp.raise_for_status()
                if not all_pcm:
                    sr = resp.headers.get("X-Sample-Rate")
                    if sr:
                        sample_rate = int(sr)
                for chunk in resp.iter_bytes(chunk_size=4800):
                    if chunk:
                        if tts_first_chunk_ms is None:
                            tts_first_chunk_ms = (time.perf_counter() - t0) * 1000
                        all_pcm.extend(chunk)

        tts_total_ms = (time.perf_counter() - t0) * 1000
        pcm = bytes(all_pcm)
        if not pcm:
            raise RuntimeError("TTS returned empty audio")

        out_duration_s = len(pcm) / 2 / sample_rate
        with wave.open(str(OUT_WAV), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
            w.writeframes(pcm)

        result["tts_first_chunk_ms"]    = round(tts_first_chunk_ms or 0, 1)
        result["tts_total_ms"]          = round(tts_total_ms, 1)
        result["tts_rtf"]               = round((tts_total_ms / 1000) / out_duration_s, 3)
        result["output_audio_duration_s"] = round(out_duration_s, 2)
        result["output_audio"]          = str(OUT_WAV)

        # ── KEY METRIC ────────────────────────────────────────────────────────
        result["speech_end_to_first_audio_ms"] = round(
            asr_ms + (brain_first_flush_ms or brain_total_ms) + (tts_first_chunk_ms or 0), 1
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
