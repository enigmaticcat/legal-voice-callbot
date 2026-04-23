import json
import re
import time
import wave
from pathlib import Path

import httpx


def _clean_for_tts(text: str) -> str:
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

_REPO_ROOT = Path(__file__).parent.parent
AUDIO_PATH = _REPO_ROOT / 'wav_16k' / 'eval_1' / 'thucuc_s1_003.wav'
OUT_WAV = _REPO_ROOT / 'latency_final_audio.wav'

ASR_URL = 'http://localhost:50051/transcribe'
BRAIN_URL = 'http://localhost:50052/think/stream'
TTS_STREAM_URL = 'http://localhost:50053/speak/stream'

# Phải khớp với orchestrator._ready_for_tts
_TTS_MIN_CHARS = 40
_TTS_PUNCTS = {'.', '?', '!', '\n'}


def _ready_for_tts(buf: str) -> bool:
    if len(buf.strip()) >= _TTS_MIN_CHARS:
        return True
    return any(p in buf for p in _TTS_PUNCTS)


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

    with httpx.Client(timeout=300) as client:
        # ── 1) ASR ────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        asr_resp = client.post(
            ASR_URL,
            content=audio_bytes,
            headers={'Content-Type': 'application/octet-stream'},
        )
        asr_resp.raise_for_status()
        asr_ms = (time.perf_counter() - t0) * 1000
        transcript = asr_resp.json().get('text', '')

        result['asr_ms'] = round(asr_ms, 1)
        result['asr_rtf'] = round((asr_ms / 1000) / asr_in_duration_s, 3) if asr_in_duration_s > 0 else None
        result['transcript_preview'] = transcript[:220]

        # ── 2) Brain stream ───────────────────────────────────────────────────
        # Ghi lại các "flush point": mỗi khi buffer đủ điều kiện gửi TTS.
        payload = {
            'query': transcript,
            'session_id': 'latency-check',
            'conversation_history': [],
        }

        brain_text_parts: list[str] = []
        brain_flushes: list[str] = []   # mỗi phần text sẽ được gửi TTS riêng
        brain_first_chunk_ms = None
        brain_first_flush_ms = None     # thời điểm flush đầu tiên (= khi TTS bắt đầu thật)
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
        # Phần còn dư chưa flush
        if buf.strip():
            if brain_first_flush_ms is None:
                brain_first_flush_ms = brain_total_ms
            brain_flushes.append(buf)

        brain_text = ''.join(brain_text_parts)
        result['brain_first_chunk_ms'] = round(brain_first_chunk_ms or 0, 1)
        result['brain_first_flush_ms'] = round(brain_first_flush_ms or 0, 1)
        result['brain_total_ms'] = round(brain_total_ms, 1)
        result['brain_timing'] = brain_timing
        result['brain_text'] = brain_text
        result['brain_text_chars'] = len(brain_text)
        result['brain_flush_count'] = len(brain_flushes)

        # ── 3) TTS real stream ────────────────────────────────────────────────
        # Phát lại các flush theo đúng thứ tự orchestrator sẽ làm.
        # Script chạy tuần tự nên đây là lower-bound: thực tế brain vẫn tiếp tục
        # stream song song khi TTS đang xử lý flush đầu tiên.
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

        result['tts_mode'] = 'real_stream'
        result['tts_first_chunk_ms'] = round(tts_first_chunk_ms or 0, 1)
        result['tts_total_ms'] = round(tts_total_ms, 1)
        result['tts_output_sample_rate'] = sample_rate
        result['tts_text_chars'] = len(brain_text)
        result['tts_rtf'] = round((tts_total_ms / 1000) / duration_s, 3) if duration_s > 0 else None
        result['audio_duration_s'] = round(duration_s, 2)
        result['audio_bytes'] = len(pcm)

        # ── 4) Metrics tổng hợp ───────────────────────────────────────────────
        # pipeline_ttfb_ms: từ lúc người nói xong → tiếng đầu tiên cất ra.
        # = ASR + (brain đến lúc flush đầu) + (TTS đến khi có PCM chunk đầu tiên)
        # Lưu ý: đây là đo tuần tự (brain xong rồi mới gọi TTS) nên brain_first_flush_ms
        # là upper-bound; thực tế song song sẽ chỉ còn ASR + brain_first_flush + TTS_first_chunk.
        pipeline_ttfb_ms = asr_ms + (brain_first_flush_ms or brain_total_ms) + (tts_first_chunk_ms or 0)
        result['pipeline_ttfb_ms'] = round(pipeline_ttfb_ms, 1)
        result['pipeline_total_ms'] = round(asr_ms + brain_total_ms + tts_total_ms, 1)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
