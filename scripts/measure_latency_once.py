import json
import re
import time
import wave
import asyncio
from pathlib import Path

import httpx
import websockets


def _clean_for_tts(text: str) -> str:
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

_REPO_ROOT = Path(__file__).parent.parent
AUDIO_PATH = _REPO_ROOT / 'wav_16k' / 'eval_1' / 'thucuc_s1_003.wav'
OUT_WAV = _REPO_ROOT / 'latency_final_audio.wav'

ASR_WS_URL  = 'ws://localhost:50051/ws/transcribe'
BRAIN_URL   = 'http://localhost:50052/think/stream'
TTS_STREAM_URL = 'http://localhost:50053/speak/stream'

CHUNK_BYTES = 3200   # 100ms × 16kHz × 2 bytes (PCM Int16)

_TTS_MIN_CHARS = 40
_TTS_PUNCTS = {'.', '?', '!', '\n'}


def _ready_for_tts(buf: str) -> bool:
    if len(buf.strip()) >= _TTS_MIN_CHARS:
        return True
    return any(p in buf for p in _TTS_PUNCTS)


async def _asr_stream(audio_bytes: bytes) -> tuple[str, float, float]:
    """
    Mô phỏng streaming ASR real-time:
      - Feed từng chunk 100ms với delay 100ms để ASR xử lý song song (giống user đang nói)
      - Gửi "end" và đo thời gian finalize (phần duy nhất đóng góp vào TTFB)
    Trả về (transcript, feed_ms, finalize_ms)
    """
    async with websockets.connect(ASR_WS_URL) as ws:
        t_feed_start = time.perf_counter()
        for i in range(0, len(audio_bytes), CHUNK_BYTES):
            chunk = audio_bytes[i:i + CHUNK_BYTES]
            await ws.send(chunk)
            await asyncio.sleep(0.1)  # giả lập real-time: 100ms/chunk
        feed_ms = (time.perf_counter() - t_feed_start) * 1000

        t_finalize = time.perf_counter()
        await ws.send(json.dumps({'type': 'end'}))
        result = json.loads(await ws.recv())
        finalize_ms = (time.perf_counter() - t_finalize) * 1000

    return result.get('text', ''), feed_ms, finalize_ms


def main() -> None:
    audio_bytes = AUDIO_PATH.read_bytes()
    with wave.open(str(AUDIO_PATH), 'rb') as f:
        asr_in_rate = f.getframerate()
        asr_in_channels = f.getnchannels()
        asr_in_width = f.getsampwidth()
        asr_in_frames = f.getnframes()
    asr_in_duration_s = asr_in_frames / asr_in_rate if asr_in_rate else 0

    result = {
        'input_audio': str(AUDIO_PATH),
        'output_audio': str(OUT_WAV),
        'asr_input_sample_rate': asr_in_rate,
        'asr_input_channels': asr_in_channels,
        'asr_input_sample_width': asr_in_width,
        'asr_input_duration_s': round(asr_in_duration_s, 3),
        'asr_input_bytes': len(audio_bytes),
    }

    # ── 1) ASR streaming ──────────────────────────────────────────────────────
    transcript, feed_ms, finalize_ms = asyncio.run(_asr_stream(audio_bytes))

    # feed_ms = thời gian push hết audio vào model (song song với user nói trong thực tế)
    # finalize_ms = thời gian ASR flush buffer cuối sau khi user dừng → đây là phần
    #               đóng góp thực sự vào TTFB trong streaming mode
    result['asr_feed_ms']      = round(feed_ms, 1)
    result['asr_finalize_ms']  = round(finalize_ms, 1)
    result['asr_rtf_feed']     = round((feed_ms / 1000) / asr_in_duration_s, 3) if asr_in_duration_s > 0 else None
    result['transcript_preview'] = transcript[:220]

    with httpx.Client(timeout=300) as client:
        # ── 2) Brain stream ───────────────────────────────────────────────────
        payload = {
            'query': transcript,
            'session_id': 'latency-check',
            'conversation_history': [],
        }

        brain_text_parts: list[str] = []
        brain_flushes: list[str] = []
        brain_first_chunk_ms = None
        brain_first_flush_ms = None
        brain_timing: dict = {}
        buf = ''

        brain_start = time.perf_counter()
        with client.stream('POST', BRAIN_URL, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode('utf-8', errors='ignore')
                data = json.loads(line)

                if brain_first_chunk_ms is None:
                    brain_first_chunk_ms = (time.perf_counter() - brain_start) * 1000

                text = data.get('text', '')
                if text:
                    brain_text_parts.append(text)
                    buf += text
                    if _ready_for_tts(buf):
                        if brain_first_flush_ms is None:
                            brain_first_flush_ms = (time.perf_counter() - brain_start) * 1000
                        brain_flushes.append(buf)
                        buf = ''

                if data.get('timing'):
                    brain_timing.update(data['timing'])

                if data.get('is_final'):
                    break

        brain_total_ms = (time.perf_counter() - brain_start) * 1000
        if buf.strip():
            if brain_first_flush_ms is None:
                brain_first_flush_ms = brain_total_ms
            brain_flushes.append(buf)

        brain_text = ''.join(brain_text_parts)
        result['brain_first_chunk_ms'] = round(brain_first_chunk_ms or 0, 1)
        result['brain_first_flush_ms'] = round(brain_first_flush_ms or 0, 1)
        result['brain_total_ms']       = round(brain_total_ms, 1)
        result['brain_timing']         = brain_timing
        result['brain_text']           = brain_text
        result['brain_text_chars']     = len(brain_text)
        result['brain_flush_count']    = len(brain_flushes)

        # ── 3) TTS stream ─────────────────────────────────────────────────────
        sample_rate = 24000
        all_pcm = bytearray()
        tts_first_chunk_ms = None

        tts_start = time.perf_counter()
        for flush_text in brain_flushes:
            with client.stream('POST', TTS_STREAM_URL, json={'text': _clean_for_tts(flush_text)}) as resp:
                resp.raise_for_status()
                sr_header = resp.headers.get('X-Sample-Rate')
                if sr_header and not all_pcm:
                    sample_rate = int(sr_header)
                for chunk in resp.iter_bytes(chunk_size=4800):
                    if chunk:
                        if tts_first_chunk_ms is None:
                            tts_first_chunk_ms = (time.perf_counter() - tts_start) * 1000
                        all_pcm.extend(chunk)

        tts_total_ms = (time.perf_counter() - tts_start) * 1000
        pcm = bytes(all_pcm)
        if not pcm:
            raise RuntimeError('TTS returned empty audio stream')

        duration_s = len(pcm) / 2 / sample_rate
        with wave.open(str(OUT_WAV), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm)

        result['tts_mode']             = 'real_stream'
        result['tts_first_chunk_ms']   = round(tts_first_chunk_ms or 0, 1)
        result['tts_total_ms']         = round(tts_total_ms, 1)
        result['tts_output_sample_rate'] = sample_rate
        result['tts_text_chars']       = len(brain_text)
        result['tts_rtf']              = round((tts_total_ms / 1000) / duration_s, 3) if duration_s > 0 else None
        result['audio_duration_s']     = round(duration_s, 2)
        result['audio_bytes']          = len(pcm)

        # ── 4) Metrics tổng hợp ───────────────────────────────────────────────
        # speech_end_to_first_audio_ms: từ lúc người nói XONG → byte audio đầu tiên
        # Trong streaming mode, ASR xử lý song song khi user đang nói.
        # Chỉ còn asr_finalize_ms (flush buffer cuối) + brain + tts đóng góp vào TTFB.
        speech_end_to_first_audio_ms = (
            finalize_ms
            + (brain_first_flush_ms or brain_total_ms)
            + (tts_first_chunk_ms or 0)
        )
        result['speech_end_to_first_audio_ms'] = round(speech_end_to_first_audio_ms, 1)
        result['pipeline_total_ms'] = round(feed_ms + brain_total_ms + tts_total_ms, 1)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
