"""
Part 2: ASR → LLM Pipeline Evaluation với RAGAS
=================================================
Đánh giá toàn bộ pipeline: audio WAV → Sherpa-Onnx ASR → RAG + Gemini → câu trả lời.

Mục đích chính:
  1. ASR quality  — WER/CER giữa transcript và câu hỏi gốc
  2. LLM quality  — RAGAS scores khi LLM nhận text đã qua ASR
  3. Degradation  — so sánh RAGAS pipeline vs LLM-only (cần chạy eval_llm_ragas.py trước)
  4. Latency      — asr_ms + rag_ms + llm_ttft_ms = e2e từ audio → câu trả lời

Input:
  wav/eval_{N}/        — audio gốc (Sherpa-Onnx tự resample về 16kHz)
  eval_split_{N}.jsonl — câu hỏi gốc + reference answer

Cách dùng:
  # Eval nhanh, 10 câu đầu
  python eval_pipeline_ragas.py --splits 1 --limit 10

  # Chỉ đo ASR quality (WER), không chạy LLM
  python eval_pipeline_ragas.py --splits 1 --asr-only

  # Đầy đủ, so sánh với LLM-only baseline
  python eval_pipeline_ragas.py --splits 1 2 3 4 5 \\
      --compare results/llm_eval_results.jsonl

  # Resume
  python eval_pipeline_ragas.py --splits 1 --out results/pipeline_eval.jsonl --resume
"""

import argparse
import asyncio
import json
import os
import sys
import struct
import time
import wave
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
BRAIN_DIR = REPO_ROOT / "nutrition-callbot" / "brain"
ASR_DIR = REPO_ROOT / "nutrition-callbot" / "asr"
CALLBOT_DIR = REPO_ROOT / "nutrition-callbot"
EVAL_DIR = Path(__file__).resolve().parent
WAV_DIR = REPO_ROOT / "wav"

sys.path.insert(0, str(BRAIN_DIR))
sys.path.insert(0, str(ASR_DIR))
sys.path.insert(0, str(CALLBOT_DIR))

from dotenv import load_dotenv
load_dotenv(CALLBOT_DIR / ".env")

from eval_utils import (
    load_eval_split, load_results, wer, cer,
    run_ragas, print_timing_table, timing_stats,
)


# ── ASR ───────────────────────────────────────────────────────────────────────

def load_asr_transcriber():
    """Load Sherpa-Onnx Transcriber. Returns None if unavailable."""
    try:
        from core.transcriber import Transcriber
        return Transcriber()
    except ImportError:
        print("WARNING: sherpa_onnx not installed. Cannot run ASR.")
        print("  pip install sherpa-onnx")
        return None
    except Exception as e:
        print(f"WARNING: Failed to load Sherpa-Onnx: {e}")
        return None


def read_wav_pcm(wav_path: Path) -> tuple[bytes, int]:
    """Read WAV file, return (pcm_bytes, sample_rate)."""
    with wave.open(str(wav_path), "rb") as w:
        n_frames = w.getnframes()
        n_ch = w.getnchannels()
        sr = w.getframerate()
        raw = w.readframes(n_frames)

    if n_ch == 1:
        return raw, sr

    # Stereo → mono (take left channel)
    total = n_frames * n_ch
    samples = struct.unpack(f"<{total}h", raw)
    mono = samples[::n_ch]
    return struct.pack(f"<{len(mono)}h", *mono), sr


def transcribe_wav(transcriber, wav_path: Path) -> tuple[str, float]:
    """
    Transcribe a WAV file using Sherpa-Onnx (batch mode).
    Calls input_finished() to flush the online transducer properly.
    Returns (transcript, asr_ms).
    """
    import numpy as np
    pcm, sr = read_wav_pcm(wav_path)

    t0 = time.time()
    stream = transcriber.create_stream()

    # Feed all audio
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    stream.accept_waveform(sr, samples)

    # Signal end-of-utterance so online transducer processes remaining frames
    stream.input_finished()

    while transcriber.recognizer.is_ready(stream):
        transcriber.recognizer.decode_stream(stream)

    text = transcriber.recognizer.get_result(stream).strip()
    asr_ms = (time.time() - t0) * 1000

    return text, asr_ms


# ── Brain pipeline ────────────────────────────────────────────────────────────

async def init_brain():
    from config import config as brain_config
    from core.llm import LLMClient
    from core.rag import RAGPipeline

    if not brain_config.gemini_api_key:
        print("ERROR: GEMINI_API_KEY not set. Check nutrition-callbot/.env")
        sys.exit(1)

    llm = LLMClient(api_key=brain_config.gemini_api_key, model=brain_config.gemini_model)

    qdrant_kwargs = {}
    if brain_config.qdrant_path:
        qdrant_kwargs["qdrant_path"] = brain_config.qdrant_path
    elif brain_config.qdrant_url:
        qdrant_kwargs["qdrant_url"] = brain_config.qdrant_url
        qdrant_kwargs["qdrant_api_key"] = brain_config.qdrant_api_key
    else:
        print("ERROR: QDRANT_URL or QDRANT_PATH not set in .env")
        sys.exit(1)

    rag = RAGPipeline(collection=brain_config.qdrant_collection, **qdrant_kwargs)
    return llm, rag, brain_config


async def run_llm(llm, rag, asr_transcript: str) -> tuple[str, list, float, float]:
    """Run RAG + LLM on ASR transcript. Returns (answer, contexts, rag_ms, llm_ttft_ms)."""
    from core.prompt import build_prompt, NUTRITION_SYSTEM_PROMPT

    t0 = time.time()
    docs = await rag.search(asr_transcript)
    rag_ms = (time.time() - t0) * 1000

    contexts = [d["content"] for d in docs]
    context_str = "\n\n".join(
        f"[{d.get('source', '').upper()} — {d.get('title', '')}]\n{d.get('content', '')}"
        for d in docs
    )

    prompt = build_prompt(query=asr_transcript, nutrition_context=context_str)
    t_llm = time.time()
    llm_ttft_ms = None
    parts = []

    async for chunk in llm.generate_stream(
        prompt=prompt,
        system_instruction=NUTRITION_SYSTEM_PROMPT,
    ):
        if llm_ttft_ms is None:
            llm_ttft_ms = (time.time() - t_llm) * 1000
        parts.append(chunk.get("text", ""))

    return "".join(parts), contexts, rag_ms, llm_ttft_ms or 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="ASR→LLM pipeline evaluation with RAGAS")
    parser.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="results/pipeline_eval_results.jsonl")
    parser.add_argument("--asr-only", action="store_true",
                        help="Only run ASR + compute WER, skip LLM + RAGAS")
    parser.add_argument("--no-ragas", action="store_true",
                        help="Skip RAGAS scoring")
    parser.add_argument("--compare", default=None, metavar="FILE",
                        help="Path to llm_eval_results.jsonl for degradation comparison")
    parser.add_argument("--resume", action="store_true",
                        help="Skip IDs already in output file")
    args = parser.parse_args()

    out_path = EVAL_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if args.resume:
        done_ids = {r["id"] for r in load_results(out_path)}
        print(f"Resume: {len(done_ids)} samples already done.")

    # Load ASR
    print("Loading ASR (Sherpa-Onnx)...")
    transcriber = load_asr_transcriber()
    if transcriber is None:
        print("ERROR: Cannot run pipeline eval without ASR.")
        sys.exit(1)

    # Load Brain (unless --asr-only)
    llm, rag, brain_config = None, None, None
    if not args.asr_only:
        llm, rag, brain_config = await init_brain()

    all_results = list(load_results(out_path)) if args.resume else []

    for split_n in args.splits:
        jsonl_path = EVAL_DIR / f"eval_split_{split_n}.jsonl"
        wav_dir = WAV_DIR / f"eval_{split_n}"

        if not jsonl_path.exists():
            print(f"SKIP: {jsonl_path} not found")
            continue
        if not wav_dir.exists():
            print(f"SKIP: {wav_dir} not found")
            continue

        samples = load_eval_split(jsonl_path)
        if args.limit:
            samples = samples[: args.limit]

        pending = [s for s in samples if s["id"] not in done_ids]
        print(f"\n=== Split {split_n}: {len(pending)}/{len(samples)} samples to run ===")

        for i, sample in enumerate(pending):
            sid = sample["id"]
            wav_path = wav_dir / f"{sid}.wav"

            if not wav_path.exists():
                print(f"  [{i+1}/{len(pending)}] {sid} — WAV not found, skip")
                continue

            print(f"  [{i+1}/{len(pending)}] {sid} ...", end="", flush=True)

            try:
                t_total = time.time()

                # ASR
                asr_transcript, asr_ms = transcribe_wav(transcriber, wav_path)

                # WER/CER vs reference question
                ref_question = sample["question"]
                word_err = wer(ref_question, asr_transcript)
                char_err = cer(ref_question, asr_transcript)

                result = {
                    "id": sid,
                    "split": sample.get("split"),
                    "source": sample.get("source"),
                    "question": ref_question,        # original reference question
                    "asr_transcript": asr_transcript,  # what ASR produced
                    "wer": round(word_err, 4),
                    "cer": round(char_err, 4),
                    "reference": sample["answer"],
                    "timing": {"asr_ms": round(asr_ms, 1)},
                }

                if not args.asr_only:
                    answer, contexts, rag_ms, llm_ttft_ms = await run_llm(llm, rag, asr_transcript)
                    total_ms = (time.time() - t_total) * 1000

                    result["answer"] = answer
                    result["contexts"] = contexts
                    result["timing"].update({
                        "rag_ms": round(rag_ms, 1),
                        "llm_ttft_ms": round(llm_ttft_ms, 1),
                        "total_ms": round(total_ms, 1),
                    })

                    print(
                        f" wer={word_err:.2f}  asr={asr_ms:.0f}ms"
                        f"  rag={rag_ms:.0f}ms  llm={llm_ttft_ms:.0f}ms"
                    )
                else:
                    print(f" wer={word_err:.2f}  cer={char_err:.2f}  asr={asr_ms:.0f}ms")

                all_results.append(result)
                done_ids.add(sid)

                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f" ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"Saved {len(all_results)} results → {out_path}")

    if not all_results:
        return

    # ── ASR quality summary ───────────────────────────────────────────────────
    wer_vals = [r["wer"] for r in all_results if "wer" in r]
    cer_vals = [r["cer"] for r in all_results if "cer" in r]
    if wer_vals:
        ws = timing_stats(wer_vals)
        cs = timing_stats(cer_vals)
        print(f"\n=== ASR Quality (n={len(wer_vals)}) ===")
        print(f"  WER  avg={ws['avg']:.3f}  p50={ws['p50']:.3f}  p95={ws['p95']:.3f}")
        print(f"  CER  avg={cs['avg']:.3f}  p50={cs['p50']:.3f}  p95={cs['p95']:.3f}")

    # ── Latency summary ───────────────────────────────────────────────────────
    if not args.asr_only:
        print("\n=== Latency Summary ===")
        print_timing_table(all_results, extra_keys=["asr_ms"])

    # ── RAGAS ─────────────────────────────────────────────────────────────────
    llm_results = all_results  # filter out asr-only samples
    if args.no_ragas or args.asr_only:
        print("\nSkipping RAGAS scoring.")
    else:
        llm_results = [r for r in all_results if r.get("answer") and r.get("contexts")]
        if not llm_results:
            print("\nNo LLM results to evaluate with RAGAS.")
        else:
            print(f"\n=== Running RAGAS Evaluation on {len(llm_results)} samples ===")
            try:
                # Evaluate using original question (not ASR transcript) so scores are
                # comparable with eval_llm_ragas.py results.
                summary = run_ragas(llm_results, brain_config.gemini_api_key, question_key="question")

                print("\n=== RAGAS Scores (pipeline — LLM sees ASR text) ===")
                for metric, score in summary.items():
                    bar = "█" * int(score * 20)
                    print(f"  {metric:<25} {score:.4f}  {bar}")

                # ── Degradation comparison ─────────────────────────────────
                if args.compare:
                    compare_path = EVAL_DIR / args.compare
                    llm_only = load_results(compare_path)
                    if llm_only:
                        llm_only_ragas = {
                            r["id"]: r.get("ragas", {})
                            for r in llm_only
                            if r.get("ragas")
                        }
                        if llm_only_ragas:
                            print("\n=== Degradation: LLM-only vs Pipeline ===")
                            print(f"  {'Metric':<25} {'LLM-only':>10} {'Pipeline':>10} {'Delta':>10}")
                            print("  " + "-" * 55)
                            for metric in summary:
                                llm_scores = [
                                    llm_only_ragas[rid][metric]
                                    for rid in llm_only_ragas
                                    if metric in llm_only_ragas[rid]
                                    and llm_only_ragas[rid][metric] is not None
                                ]
                                if llm_scores:
                                    llm_avg = sum(llm_scores) / len(llm_scores)
                                    delta = summary[metric] - llm_avg
                                    sign = "+" if delta >= 0 else ""
                                    print(
                                        f"  {metric:<25} {llm_avg:>10.4f} {summary[metric]:>10.4f} "
                                        f"{sign}{delta:>9.4f}"
                                    )

                # Save summary
                summary_path = out_path.parent / "pipeline_eval_ragas_summary.json"
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "splits": args.splits,
                            "n_samples": len(llm_results),
                            "asr_wer_avg": round(sum(wer_vals) / len(wer_vals), 4) if wer_vals else None,
                            "asr_cer_avg": round(sum(cer_vals) / len(cer_vals), 4) if cer_vals else None,
                            "ragas": summary,
                        },
                        f, ensure_ascii=False, indent=2,
                    )
                print(f"\nRAGAS summary → {summary_path}")

                # Update output with per-sample RAGAS scores
                with open(out_path, "w", encoding="utf-8") as f:
                    for r in all_results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

            except ImportError as e:
                print(f"\nRAGAS not installed: {e}")
                print("Install:  pip install ragas>=0.2.0 langchain-google-genai langchain")


if __name__ == "__main__":
    asyncio.run(main())
