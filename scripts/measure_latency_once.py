import json
import io
import os
import time
import wave
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).parent.parent
AUDIO_PATH = _REPO_ROOT / 'wav_16k' / 'eval_1' / 'thucuc_s1_003.wav'
OUT_PCM = _REPO_ROOT / 'latency_final_audio.pcm'
OUT_WAV = _REPO_ROOT / 'latency_final_audio.wav'

ASR_URL = 'http://localhost:50051/transcribe'
BRAIN_URL = 'http://localhost:50052/think/stream'
TTS_URL = 'http://localhost:50053/speak'

FAKE_STREAMING_AUDIO = os.getenv('FAKE_STREAMING_AUDIO', 'true').strip().lower() in {'1', 'true', 'yes', 'on'}
FAKE_STREAM_CHUNK_BYTES = int(os.getenv('FAKE_STREAM_CHUNK_BYTES', '4800'))


def _wav_bytes_to_pcm(wav_bytes: bytes) -> tuple[bytes, int, int]:
    with wave.open(io.BytesIO(wav_bytes), 'rb') as r:
        sample_rate = r.getframerate()
        sample_width = r.getsampwidth()
        pcm = r.readframes(r.getnframes())
    return pcm, sample_rate, sample_width


def main() -> None:
    audio_bytes = AUDIO_PATH.read_bytes()
    with wave.open(str(AUDIO_PATH), 'rb') as src_wav:
        asr_in_rate = src_wav.getframerate()
        asr_in_channels = src_wav.getnchannels()
        asr_in_width = src_wav.getsampwidth()
        asr_in_frames = src_wav.getnframes()

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
        # 1) ASR latency
        t0 = time.perf_counter()
        asr_resp = client.post(
            ASR_URL,
            content=audio_bytes,
            headers={'Content-Type': 'application/octet-stream'},
        )
        asr_resp.raise_for_status()
        asr_ms = (time.perf_counter() - t0) * 1000
        asr_json = asr_resp.json()
        transcript = asr_json.get('text', '')

        result['asr_ms'] = round(asr_ms, 1)
        result['asr_rtf'] = round((asr_ms / 1000) / asr_in_duration_s, 3) if asr_in_duration_s > 0 else None
        result['transcript_preview'] = transcript[:220]

        # 2) Brain latency (stream)
        payload = {
            'query': transcript,
            'session_id': 'latency-check',
            'conversation_history': [],
        }

        brain_text_parts = []
        brain_first_chunk_ms = None
        brain_start = time.perf_counter()
        brain_timing_from_service = {}

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

                if data.get('text'):
                    brain_text_parts.append(data['text'])

                if data.get('timing'):
                    brain_timing_from_service.update(data['timing'])

                if data.get('is_final'):
                    break

        brain_total_ms = (time.perf_counter() - brain_start) * 1000
        brain_text = ''.join(brain_text_parts)

        result['brain_first_chunk_ms'] = round(brain_first_chunk_ms or 0, 1)
        result['brain_total_ms'] = round(brain_total_ms, 1)
        result['brain_timing'] = brain_timing_from_service
        result['brain_text_preview'] = brain_text[:260]
        result['brain_text_chars'] = len(brain_text)

        # 3) TTS latency + final audio output
        tts_start = time.perf_counter()
        tts_ttfb_ms = None
        pcm_chunks = []

        tts_resp = client.post(TTS_URL, json={'text': brain_text})
        tts_resp.raise_for_status()
        if tts_ttfb_ms is None:
            tts_ttfb_ms = (time.perf_counter() - tts_start) * 1000
        wav_payload = tts_resp.content

        pcm_raw, sample_rate, sample_width = _wav_bytes_to_pcm(wav_payload)
        if sample_width != 2:
            raise RuntimeError(f'Unexpected TTS WAV sample width={sample_width}, expected 2 bytes (int16)')

        if FAKE_STREAMING_AUDIO:
            chunk_size = max(1, FAKE_STREAM_CHUNK_BYTES)
            for i in range(0, len(pcm_raw), chunk_size):
                chunk = pcm_raw[i:i + chunk_size]
                if chunk:
                    pcm_chunks.append(chunk)
            result['tts_mode'] = 'fake_stream_from_wav'
            result['tts_fake_stream_chunk_bytes'] = chunk_size
            result['tts_fake_stream_chunks'] = len(pcm_chunks)
            result['tts_fake_stream_first_chunk_ms'] = round(tts_ttfb_ms or 0, 1)
        else:
            pcm_chunks.append(pcm_raw)
            result['tts_mode'] = 'single_chunk_from_wav'

        tts_total_ms = (time.perf_counter() - tts_start) * 1000
        pcm = b''.join(pcm_chunks)
        if not pcm:
            raise RuntimeError('TTS returned empty audio stream')

        OUT_PCM.write_bytes(pcm)

        with wave.open(str(OUT_WAV), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm)

        duration_s = len(pcm) / 2 / sample_rate if pcm else 0
        result['tts_ttfb_ms'] = round(tts_ttfb_ms or 0, 1)
        result['tts_total_ms'] = round(tts_total_ms, 1)
        result['audio_duration_s'] = round(duration_s, 2)
        result['audio_bytes'] = len(pcm)
        result['tts_output_sample_rate'] = sample_rate
        result['tts_output_sample_width'] = sample_width
        result['tts_text_chars'] = len(brain_text)
        result['tts_rtf'] = round((tts_total_ms / 1000) / duration_s, 3) if duration_s > 0 else None

        # 4) Overall pipeline (service-by-service, serialized)
        result['pipeline_total_ms'] = round(asr_ms + brain_total_ms + tts_total_ms, 1)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
