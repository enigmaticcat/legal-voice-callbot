import json
import time
import wave
from pathlib import Path

import httpx

AUDIO_PATH = Path('/Users/nguyenthithutam/Desktop/Callbot/wav_16k/eval_1/thucuc_s1_003.wav')
OUT_PCM = Path('/Users/nguyenthithutam/Desktop/Callbot/latency_final_audio.pcm')
OUT_WAV = Path('/Users/nguyenthithutam/Desktop/Callbot/latency_final_audio.wav')
SAMPLE_RATE = 24000

ASR_URL = 'http://localhost:50051/transcribe'
BRAIN_URL = 'http://localhost:50052/think/stream'
TTS_URL = 'http://localhost:50053/speak/stream'


def main() -> None:
    audio_bytes = AUDIO_PATH.read_bytes()

    result = {
        'input_audio': str(AUDIO_PATH),
        'output_audio': str(OUT_WAV),
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

        # 3) TTS latency + final audio output
        tts_start = time.perf_counter()
        tts_ttfb_ms = None
        pcm_chunks = []

        with client.stream('POST', TTS_URL, json={'text': brain_text}) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes():
                if not chunk:
                    continue
                if tts_ttfb_ms is None:
                    tts_ttfb_ms = (time.perf_counter() - tts_start) * 1000
                pcm_chunks.append(chunk)

        tts_total_ms = (time.perf_counter() - tts_start) * 1000
        pcm = b''.join(pcm_chunks)
        OUT_PCM.write_bytes(pcm)

        with wave.open(str(OUT_WAV), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)

        duration_s = len(pcm) / 2 / SAMPLE_RATE if pcm else 0
        result['tts_ttfb_ms'] = round(tts_ttfb_ms or 0, 1)
        result['tts_total_ms'] = round(tts_total_ms, 1)
        result['audio_duration_s'] = round(duration_s, 2)
        result['audio_bytes'] = len(pcm)

        # 4) Overall pipeline (service-by-service, serialized)
        result['pipeline_total_ms'] = round(asr_ms + brain_total_ms + tts_total_ms, 1)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
