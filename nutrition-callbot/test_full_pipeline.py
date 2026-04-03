"""
Test Full Pipeline: ASR → Brain → TTS
Chạy trực tiếp (không cần gRPC server), import handler thẳng.

Modes:
  --mode full        : WAV file → ASR → Brain → TTS → output.wav
  --mode brain-tts   : text input → Brain → TTS → output.wav
  --mode brain-only  : text input → Brain → in ra màn hình (không TTS)

Ví dụ:
  python test_full_pipeline.py --mode brain-tts --text "Tôi nên ăn gì để giảm cân?"
  python test_full_pipeline.py --mode full --audio input.wav
  python test_full_pipeline.py --mode brain-only --text "Protein trong thịt gà là bao nhiêu?"
"""
import argparse
import asyncio
import os
import sys
import time
import wave
import struct

# ─── Paths ───────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tts"))   # vieneu, core.synthesizer
sys.path.insert(0, os.path.join(ROOT, "brain")) # core.llm, core.rag ...
sys.path.insert(0, os.path.join(ROOT, "asr"))   # core.transcriber


# ─── Timing helper ───────────────────────────────────────
class Timer:
    def __init__(self, name):
        self.name = name
        self.start = None
        self.ttfb = None   # time to first byte/token
        self.end = None

    def tick(self):
        self.start = time.time()

    def first(self):
        if self.ttfb is None:
            self.ttfb = time.time() - self.start

    def tock(self):
        self.end = time.time()

    @property
    def total(self):
        return (self.end or time.time()) - (self.start or 0)

    def report(self):
        ttfb_str = f"TTFB={self.ttfb*1000:.0f}ms  " if self.ttfb else ""
        print(f"  [{self.name}] {ttfb_str}Total={self.total*1000:.0f}ms")


# ─── PCM → WAV ───────────────────────────────────────────
def save_wav(pcm_bytes: bytes, path: str, sample_rate: int = 24000):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    duration = len(pcm_bytes) / 2 / sample_rate
    print(f"  [TTS] Saved → {path}  ({duration:.2f}s audio)")


# ─── ASR ─────────────────────────────────────────────────
def run_asr(audio_path: str) -> tuple[str, float]:
    from core.transcriber import Transcriber

    t = Timer("ASR")
    t.tick()

    with wave.open(audio_path, "rb") as wf:
        audio_pcm = wf.readframes(wf.getnframes())

    transcriber = Transcriber()
    stream = transcriber.create_stream()
    text = transcriber.accept_wave(stream, audio_pcm)
    t.tock()

    if not text:
        text = "Tôi nên ăn gì để giảm cân?"
        print(f"  [ASR] (No result, dùng fallback text)")
    else:
        print(f"  [ASR] → \"{text}\"")

    t.report()
    return text, t.total


# ─── Brain ───────────────────────────────────────────────
async def run_brain(text: str, brain_handler) -> tuple[str, float, float]:
    t = Timer("Brain")
    t.tick()

    full_text = ""
    async for chunk in brain_handler.think(query=text, session_id="pipeline-test"):
        if chunk.get("text"):
            t.first()
            full_text += chunk["text"]
            print(chunk["text"], end="", flush=True)

        if chunk.get("is_final") and chunk.get("timing"):
            timing = chunk["timing"]
            if "first_chunk_total_ms" in timing:
                print(f"\n  [Brain] TTFT={timing['first_chunk_total_ms']:.0f}ms  "
                      f"RAG={timing.get('rag_ms', 0):.0f}ms  "
                      f"LLM={timing.get('llm_ttft_ms', 0):.0f}ms")

    t.tock()
    t.report()
    return full_text, t.ttfb or 0, t.total


# ─── Brain → TTS streaming (overlap) ────────────────────
async def run_brain_tts(text: str, brain_handler, tts_handler) -> tuple[float, float, float, float]:
    """
    Brain chunks → TTS song song.
    Khi Brain emit chunk đầu tiên → TTS bắt đầu synthesize ngay.
    """
    t_brain = Timer("Brain")
    t_tts = Timer("TTS")
    e2e_start = time.time()

    all_pcm = bytearray()
    brain_text = ""

    t_brain.tick()
    print("  AI: ", end="", flush=True)

    async def brain_gen():
        nonlocal brain_text
        async for chunk in brain_handler.think(query=text, session_id="pipeline-test"):
            if chunk.get("text"):
                t_brain.first()
                brain_text += chunk["text"]
                print(chunk["text"], end="", flush=True)
                yield chunk["text"]
            if chunk.get("is_final") and chunk.get("timing"):
                timing = chunk["timing"]
                if "first_chunk_total_ms" in timing:
                    print(f"\n  [Brain] TTFT={timing['first_chunk_total_ms']:.0f}ms  "
                          f"RAG={timing.get('rag_ms', 0):.0f}ms")
        t_brain.tock()

    # TTS nhận từng chunk text từ Brain, synthesize song song
    t_tts.tick()
    buffer = ""
    MIN_CHUNK = 40  # ký tự tối thiểu để gửi TTS

    async for brain_chunk in brain_gen():
        buffer += brain_chunk
        # Gửi TTS khi đủ dài hoặc có dấu câu
        if len(buffer) >= MIN_CHUNK or any(p in buffer for p in [".", "!", "?", "\n"]):
            async for pcm_frame in tts_handler.speak(buffer):
                t_tts.first()  # TTFB của TTS
                all_pcm.extend(pcm_frame)
            buffer = ""

    # Flush phần còn lại
    if buffer:
        async for pcm_frame in tts_handler.speak(buffer):
            t_tts.first()
            all_pcm.extend(pcm_frame)

    t_tts.tock()

    e2e = time.time() - e2e_start
    return bytes(all_pcm), t_brain.ttfb or 0, t_tts.ttfb or 0, e2e


# ─── Init Brain handler ──────────────────────────────────
def init_brain():
    from brain.grpc_handler import BrainServiceHandler
    from brain.core.llm import LLMClient
    from brain.core.rag import RAGPipeline
    from brain.config import config as brain_config

    print("  [Brain] Đang khởi tạo LLM + RAG...")
    llm = LLMClient(
        api_key=brain_config.gemini_api_key,
        model=brain_config.gemini_model,
    )
    rag = RAGPipeline(
        qdrant_url=brain_config.qdrant_url,
        qdrant_api_key=brain_config.qdrant_api_key,
        qdrant_path=brain_config.qdrant_path,
        collection=brain_config.qdrant_collection,
    )
    return BrainServiceHandler(llm=llm, rag=rag)


# ─── Init TTS handler ────────────────────────────────────
def init_tts():
    from core.synthesizer import Synthesizer
    from grpc_handler import TTSServiceHandler

    print("  [TTS] Đang load VieNeu model (lần đầu sẽ download)...")
    synth = Synthesizer()
    synth.load_model()
    return TTSServiceHandler(synthesizer=synth)


# ─── Main ────────────────────────────────────────────────
async def main(args):
    print("\n" + "="*60)
    print(f"  MODE: {args.mode.upper()}")
    print("="*60)

    timings = {}

    # ── Mode: brain-only ─────────────────────────────────
    if args.mode == "brain-only":
        text = args.text
        print(f"\n[Input] {text}\n")
        brain = init_brain()
        print("\n[Brain Output]")
        full, ttft, total = await run_brain(text, brain)
        timings["brain_ttft_ms"] = ttft * 1000
        timings["brain_total_ms"] = total * 1000

    # ── Mode: brain-tts ──────────────────────────────────
    elif args.mode == "brain-tts":
        text = args.text
        print(f"\n[Input] {text}\n")
        brain = init_brain()
        tts = init_tts()

        print("\n[Brain → TTS]\n")
        pcm, brain_ttft, tts_ttfb, e2e = await run_brain_tts(text, brain, tts)
        save_wav(pcm, args.output)

        timings["brain_ttft_ms"] = brain_ttft * 1000
        timings["tts_ttfb_ms"] = tts_ttfb * 1000
        timings["e2e_ms"] = e2e * 1000

    # ── Mode: full (ASR → Brain → TTS) ───────────────────
    elif args.mode == "full":
        if not args.audio or not os.path.exists(args.audio):
            print(f"  [!] --audio file không tồn tại: {args.audio}")
            return

        print("\n[1/3] ASR")
        text, asr_time = run_asr(args.audio)
        timings["asr_ms"] = asr_time * 1000

        print(f"\n[2/3 + 3/3] Brain → TTS\n")
        brain = init_brain()
        tts = init_tts()
        pcm, brain_ttft, tts_ttfb, e2e = await run_brain_tts(text, brain, tts)
        save_wav(pcm, args.output)

        timings["brain_ttft_ms"] = brain_ttft * 1000
        timings["tts_ttfb_ms"] = tts_ttfb * 1000
        timings["e2e_ms"] = e2e * 1000

    # ── Report ───────────────────────────────────────────
    print("\n" + "="*60)
    print("  TIMING REPORT")
    print("="*60)
    if "asr_ms" in timings:
        print(f"  ASR transcribe     : {timings['asr_ms']:.0f}ms")
    if "brain_ttft_ms" in timings:
        print(f"  Brain TTFT         : {timings['brain_ttft_ms']:.0f}ms")
    if "tts_ttfb_ms" in timings:
        print(f"  TTS TTFB           : {timings['tts_ttfb_ms']:.0f}ms")
    if "e2e_ms" in timings:
        print(f"  ─────────────────────────────")
        status = "✅ OK" if timings["e2e_ms"] < 1500 else ("⚠️  Chấp nhận" if timings["e2e_ms"] < 3000 else "❌ Chậm")
        print(f"  E2E (đến audio đầu): {timings['e2e_ms']:.0f}ms  {status}")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["full", "brain-tts", "brain-only"],
        default="brain-tts",
        help="full=ASR+Brain+TTS | brain-tts=text+Brain+TTS | brain-only=text+Brain"
    )
    parser.add_argument("--text", default="Tôi nên ăn gì để giảm cân?", help="Input text (brain-tts, brain-only)")
    parser.add_argument("--audio", default="", help="Path to WAV file (full mode)")
    parser.add_argument("--output", default="output.wav", help="Output WAV file (full, brain-tts)")
    args = parser.parse_args()

    asyncio.run(main(args))
