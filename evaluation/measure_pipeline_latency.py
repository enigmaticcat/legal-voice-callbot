"""
Pipeline Latency Measurement
Đo thời gian từ khi gửi query vào LLM đến khi ra chunk TTS đầu tiên.

Metrics:
  llm_ttft_ms  : start → câu đầu tiên từ LLM
  tts_ttfc_ms  : câu đầu tiên → audio chunk đầu tiên từ TTS
  e2e_ttfa_ms  : tổng hai cái trên (thời gian người dùng thực sự chờ)
  total_ms     : toàn bộ pipeline chạy xong

LLM providers:
  --llm gemini      → Gemini API (cần GEMINI_API_KEY)
  --llm openai      → OpenAI-compatible self-hosted (vLLM, Ollama, LM Studio)
                      cần --llm-url và --llm-model

Nguồn câu hỏi:
  --qa thucuc       → đọc thucuc_qa.jsonl (255 câu)
  --qa default      → 5 câu mặc định
  --query "..."     → 1 câu cụ thể

Ví dụ:
  # Gemini, 5 câu mặc định, 3 lần mỗi câu
  python measure_pipeline_latency.py --llm gemini

  # Gemini, toàn bộ thucuc_qa.jsonl, 1 lần mỗi câu
  python measure_pipeline_latency.py --llm gemini --qa thucuc --n 1

  # vLLM self-hosted, 50 câu đầu từ thucuc_qa, lưu kết quả
  python measure_pipeline_latency.py --llm openai \\
      --llm-url http://localhost:8000/v1 \\
      --llm-model Qwen/Qwen2.5-1.5B-Instruct \\
      --qa thucuc --limit 50 --out latency_results.jsonl

  # Ollama local
  python measure_pipeline_latency.py --llm openai \\
      --llm-url http://localhost:11434/v1 \\
      --llm-model qwen2.5:1.5b-instruct \\
      --qa thucuc --n 1
"""

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "nutrition-callbot" / ".env")

GEMINI_MODEL        = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
BACKBONE_REPO       = "pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf"
CODEC_REPO          = "neuphonic/distill-neucodec"
SENTENCE_DELIMITERS = ".!?;:"
CLAUSE_DELIMITERS   = ","
QA_PATH             = Path(__file__).parent / "thucuc_qa.jsonl"

SYS_PROMPT = (
    "Bạn là bác sĩ dinh dưỡng. Trả lời ngắn gọn, tự nhiên bằng tiếng Việt. "
    "KHÔNG dùng Markdown, emoji, hay gạch đầu dòng. "
    "Bắt đầu bằng 'Chào bạn,'. Trả lời 2-3 câu."
)

DEFAULT_QUERIES = [
    "Thưa bác sĩ, tôi muốn hỏi có nên uống Omega 3-6-9 mỗi ngày hay không?",
    "Bác sĩ ơi, trẻ 6 tháng tuổi bắt đầu ăn dặm thì nên ăn gì?",
    "Người bị tiểu đường type 2 có ăn được trái cây ngọt không?",
    "Tôi bị thiếu sắt, nên ăn gì để bổ sung sắt hiệu quả nhất?",
    "Phụ nữ mang thai cần bổ sung vitamin gì trong 3 tháng đầu?",
]

WARMUP_QUERIES = [
    "Ăn rau xanh có tốt không?",
    "Uống nước đủ 2 lít mỗi ngày có lợi gì?",
    "Canxi có trong thực phẩm nào?",
    "Trẻ em nên ăn mấy bữa một ngày?",
    "Vitamin D lấy từ đâu?",
    "Ăn nhiều đường có hại không?",
    "Protein quan trọng như thế nào?",
    "Chất xơ có tác dụng gì?",
    "Người cao tuổi nên ăn gì?",
    "Omega-3 có trong thực phẩm nào?",
    "Sắt quan trọng như thế nào với cơ thể?",
    "Ăn sáng có quan trọng không?",
    "Kẽm có tác dụng gì?",
    "Cholesterol xấu là gì?",
    "Người gầy cần bổ sung gì?",
]


# ── Sentence splitter (dùng chung cho mọi LLM provider) ──────────────────────

def _split_into_sentences(token_stream):
    """
    Nhận iterator yield từng token string,
    yield từng câu hoàn chỉnh khi gặp dấu câu.
    """
    buffer = ""
    for token in token_stream:
        if not token:
            continue
        buffer += token
        last_idx = -1
        for delim in SENTENCE_DELIMITERS + CLAUSE_DELIMITERS:
            idx = buffer.rfind(delim)
            if idx > last_idx:
                last_idx = idx
        if last_idx >= 0:
            sentence = buffer[: last_idx + 1].strip()
            buffer = buffer[last_idx + 1:]
            if sentence:
                yield sentence
    if buffer.strip():
        yield buffer.strip()


# ── LLM providers ─────────────────────────────────────────────────────────────

def llm_gemini(query: str, api_key: str):
    """Gemini API streaming → yield từng câu."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=SYS_PROMPT,
            temperature=0.7,
        ),
    )

    def tokens():
        for chunk in response:
            if chunk.text:
                yield chunk.text

    yield from _split_into_sentences(tokens())


def llm_openai_compatible(query: str, base_url: str, model: str, api_key: str = "dummy"):
    """
    OpenAI-compatible streaming → yield từng câu.
    Dùng cho vLLM, Ollama (/v1), LM Studio, v.v.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user",   "content": query},
        ],
        stream=True,
        temperature=0.7,
    )

    def tokens():
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    yield from _split_into_sentences(tokens())


# ── Measurement core ──────────────────────────────────────────────────────────

def measure_one(query: str, tts, voice, llm_fn) -> dict:
    """
    Đo latency cho 1 query.

    Architecture (đúng theo llm_tts_demo.py trong VieNeu-TTS):
      Thread A: gọi LLM, push câu → text_queue
      Main:     lấy câu từ queue, gọi tts.infer_stream() ngay
    """
    text_queue: queue.Queue = queue.Queue()
    result = {}
    t_start = time.perf_counter()

    def llm_worker():
        try:
            first = True
            for sentence in llm_fn(query):
                if first:
                    result["t_llm_first"] = time.perf_counter()
                    first = False
                text_queue.put(sentence)
        except Exception as e:
            result["llm_error"] = str(e)
        finally:
            text_queue.put(None)

    llm_thread = threading.Thread(target=llm_worker, daemon=True)
    llm_thread.start()

    all_audio = []
    sentence_idx = 0
    tts_chunk_times = []

    while True:
        sentence = text_queue.get()
        if sentence is None:
            break

        sentence_idx += 1
        t_tts_s = time.perf_counter()

        for audio_chunk in tts.infer_stream(text=sentence, voice=voice, max_chars=256):
            if "t_tts_first" not in result:
                result["t_tts_first"] = time.perf_counter()
            all_audio.append(audio_chunk)

        tts_chunk_times.append(round((time.perf_counter() - t_tts_s) * 1000, 1))

    llm_thread.join()
    result["t_end"] = time.perf_counter()

    t_llm_first = result.get("t_llm_first")
    t_tts_first = result.get("t_tts_first")

    llm_ttft_ms = round((t_llm_first - t_start) * 1000, 1) if t_llm_first else None
    tts_ttfc_ms = round((t_tts_first - t_llm_first) * 1000, 1) if (t_tts_first and t_llm_first) else None
    e2e_ttfa_ms = round((t_tts_first - t_start) * 1000, 1) if t_tts_first else None
    total_ms    = round((result["t_end"] - t_start) * 1000, 1)

    total_samples = sum(len(a) for a in all_audio)
    audio_dur_s   = total_samples / 24_000

    return {
        "query"             : query,
        "llm_ttft_ms"       : llm_ttft_ms,
        "tts_ttfc_ms"       : tts_ttfc_ms,
        "e2e_ttfa_ms"       : e2e_ttfa_ms,
        "total_ms"          : total_ms,
        "audio_duration_s"  : round(audio_dur_s, 2),
        "rtf"               : round(total_ms / 1000 / audio_dur_s, 3) if audio_dur_s > 0 else None,
        "sentences"         : sentence_idx,
        "tts_chunk_times_ms": tts_chunk_times,
        "error"             : result.get("llm_error"),
    }


# ── Warmup ────────────────────────────────────────────────────────────────────

def run_warmup(tts, voice, llm_fn, n: int = 15):
    """
    Chạy n câu dummy để làm nóng:
      - LLM: khởi tạo kết nối HTTP, load KV-cache
      - TTS: load GGUF weights vào RAM/VRAM, warm up codec
    Kết quả warmup bị bỏ qua, không tính vào metrics.
    """
    warmup_qs = (WARMUP_QUERIES * ((n // len(WARMUP_QUERIES)) + 1))[:n]
    print(f"Warmup: {n} câu dummy (kết quả bị bỏ qua)...")

    for i, q in enumerate(warmup_qs, 1):
        try:
            # Drain LLM + TTS, không lưu gì
            text_queue: queue.Queue = queue.Queue()

            def _llm():
                try:
                    for s in llm_fn(q):
                        text_queue.put(s)
                finally:
                    text_queue.put(None)

            t = threading.Thread(target=_llm, daemon=True)
            t.start()
            while True:
                s = text_queue.get()
                if s is None:
                    break
                for _ in tts.infer_stream(text=s, voice=voice, max_chars=256):
                    pass
            t.join()
            print(f"  warmup {i}/{n} OK", end="\r")
        except Exception as e:
            print(f"  warmup {i}/{n} ERR: {e}", end="\r")

    print(f"  Warmup xong ({n} câu).            ")


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_benchmark(queries: list, tts, voice, llm_fn, n_runs: int, out_path: str = None):
    results = []
    total = len(queries) * n_runs
    out_f = open(out_path, "w", encoding="utf-8") if out_path else None

    print(f"\nBắt đầu đo {total} lần ({len(queries)} query × {n_runs} run)...")
    if out_path:
        print(f"Ghi kết quả real-time → {out_path}\n")

    header = f"{'#':>5}  {'LLM TTFT':>10}  {'TTS TTFC':>10}  {'E2E TTFA':>10}  {'Total':>9}  Query"
    print(header)
    print("-" * 85)

    run_num = 0
    for query in queries:
        for _ in range(n_runs):
            run_num += 1
            m = measure_one(query, tts, voice, llm_fn)
            results.append(m)

            status = "ERR" if m["error"] else ""
            print(
                f"{run_num:>5}  "
                f"{str(m['llm_ttft_ms'])+'ms':>10}  "
                f"{str(m['tts_ttfc_ms'])+'ms':>10}  "
                f"{str(m['e2e_ttfa_ms'])+'ms':>10}  "
                f"{str(m['total_ms'])+'ms':>9}  "
                f"{status} {query[:55]}"
            )

            if out_f:
                out_f.write(json.dumps(m, ensure_ascii=False) + "\n")
                out_f.flush()

    if out_f:
        out_f.close()

    return results


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: list):
    ok = [r for r in results if not r.get("error")]
    print(f"\n{'='*70}")
    print(f"LATENCY SUMMARY  ({len(ok)}/{len(results)} thành công)")
    print(f"{'='*70}")

    def pct(vals, label):
        a = np.array([v for v in vals if v is not None])
        if len(a) == 0:
            return
        print(
            f"  {label:<15s}  "
            f"p50={np.percentile(a,50):.0f}  "
            f"p90={np.percentile(a,90):.0f}  "
            f"p95={np.percentile(a,95):.0f}  "
            f"p99={np.percentile(a,99):.0f}  "
            f"mean={a.mean():.0f}  "
            f"min={a.min():.0f}  max={a.max():.0f}  (ms)"
        )

    pct([r["llm_ttft_ms"] for r in ok], "LLM TTFT")
    pct([r["tts_ttfc_ms"] for r in ok], "TTS TTFC")
    pct([r["e2e_ttfa_ms"] for r in ok], "E2E TTFA")
    pct([r["total_ms"]    for r in ok], "Total")

    rtfs = [r["rtf"] for r in ok if r["rtf"]]
    if rtfs:
        print(f"\n  RTF mean={np.mean(rtfs):.3f}  (< 1.0 = faster than real-time)")
    print(f"{'='*70}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Đo pipeline latency: LLM TTFT / TTS TTFC / E2E TTFA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # LLM
    parser.add_argument("--llm",       default="gemini",
                        choices=["gemini", "openai"],
                        help="LLM provider (default: gemini)")
    parser.add_argument("--llm-url",   default="http://localhost:8000/v1",
                        help="Base URL cho OpenAI-compatible server")
    parser.add_argument("--llm-model", default=None,
                        help="Tên model (bắt buộc với --llm openai)")
    parser.add_argument("--llm-key",   default=None,
                        help="API key (default: GEMINI_API_KEY từ env)")

    # Câu hỏi
    parser.add_argument("--qa",        default="default",
                        choices=["default", "thucuc"],
                        help="Nguồn câu hỏi: default (5 câu) hoặc thucuc (255 câu)")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Giới hạn số câu từ thucuc_qa (VD: --limit 50)")
    parser.add_argument("--query",     default=None,
                        help="1 câu hỏi cụ thể (bỏ qua --qa)")
    parser.add_argument("--n",         type=int, default=1,
                        help="Số lần lặp mỗi câu (default: 1)")

    # TTS device
    parser.add_argument("--device",    default=None,
                        help="TTS device: cpu / gpu (auto-detect nếu bỏ)")

    # Warmup
    parser.add_argument("--warmup",    type=int, default=15,
                        help="Số câu dummy warmup trước khi đo (default: 15, 0 để tắt)")

    # Output
    parser.add_argument("--out",       default=None,
                        help="Lưu kết quả JSONL ra file (ghi real-time)")

    args = parser.parse_args()

    # ── Chọn LLM function ────────────────────────────────────────────────────
    if args.llm == "gemini":
        api_key = args.llm_key or os.environ.get("GEMINI_API_KEY") or input("GEMINI_API_KEY: ").strip()
        llm_fn = lambda q: llm_gemini(q, api_key)
        provider_label = f"Gemini/{GEMINI_MODEL}"

    else:  # openai-compatible
        if not args.llm_model:
            parser.error("--llm openai yêu cầu --llm-model <tên model>")
        api_key = args.llm_key or os.environ.get("OPENAI_API_KEY", "dummy")
        llm_fn = lambda q: llm_openai_compatible(q, args.llm_url, args.llm_model, api_key)
        provider_label = f"OpenAI-compat/{args.llm_model} @ {args.llm_url}"

    print(f"LLM: {provider_label}")

    # ── Chọn danh sách câu hỏi ───────────────────────────────────────────────
    if args.query:
        queries = [args.query]
    elif args.qa == "thucuc":
        if not QA_PATH.exists():
            raise FileNotFoundError(f"Không tìm thấy {QA_PATH}")
        with open(QA_PATH, encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]
        if args.limit:
            records = records[: args.limit]
        queries = [r["question"] for r in records]
        print(f"Nguồn: thucuc_qa.jsonl → {len(queries)} câu")
    else:
        queries = DEFAULT_QUERIES
        print(f"Nguồn: default ({len(queries)} câu)")

    # ── TTS device ───────────────────────────────────────────────────────────
    if args.device:
        backbone_device = args.device
        codec_device    = "cuda" if args.device == "gpu" else "cpu"
    else:
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        backbone_device = "gpu" if has_cuda else "cpu"
        codec_device    = "cuda" if has_cuda else "cpu"

    print(f"TTS device: backbone={backbone_device}, codec={codec_device}")
    print("Đang load VieNeu-TTS (lần đầu có thể mất 1-2 phút)...")

    from vieneu import Vieneu
    tts   = Vieneu(
        backbone_repo=BACKBONE_REPO,
        backbone_device=backbone_device,
        codec_repo=CODEC_REPO,
        codec_device=codec_device,
    )
    voice = tts.get_preset_voice()
    print("TTS ready.")

    # ── Warmup ───────────────────────────────────────────────────────────────
    if args.warmup > 0:
        run_warmup(tts, voice, llm_fn, n=args.warmup)

    # ── Chạy đo ──────────────────────────────────────────────────────────────
    try:
        results = run_benchmark(queries, tts, voice, llm_fn, n_runs=args.n, out_path=args.out)
        print_summary(results)
    finally:
        tts.close()


if __name__ == "__main__":
    main()
