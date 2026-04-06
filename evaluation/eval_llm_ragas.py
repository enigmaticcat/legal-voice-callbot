"""
Part 1: LLM + RAG Evaluation với RAGAS
=======================================
Đánh giá chất lượng Brain service (RAG + Gemini) trực tiếp từ câu hỏi text.

Metrics:
  answer_relevancy    — câu trả lời có liên quan câu hỏi không?
  faithfulness        — câu trả lời có trung thực với context không?
  context_precision   — context được retrieve có chính xác không?
  context_recall      — context có đủ thông tin để trả lời không?

Latency:
  rag_ms              — thời gian retrieve context từ Qdrant
  llm_ttft_ms         — time to first token từ Gemini
  total_ms            — tổng thời gian từ câu hỏi đến câu trả lời hoàn chỉnh

Cách dùng:
  # Eval nhanh với 10 câu đầu của split 1
  python eval_llm_ragas.py --splits 1 --limit 10

  # Eval toàn bộ, bỏ qua RAGAS scoring (chỉ collect outputs)
  python eval_llm_ragas.py --splits 1 2 3 4 5 --no-ragas

  # Eval đầy đủ, lưu kết quả
  python eval_llm_ragas.py --splits 1 2 3 4 5 --out results/llm_eval.jsonl

  # Resume: bỏ qua ID đã có trong file output
  python eval_llm_ragas.py --splits 1 --out results/llm_eval.jsonl --resume
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
BRAIN_DIR = REPO_ROOT / "nutrition-callbot" / "brain"
CALLBOT_DIR = REPO_ROOT / "nutrition-callbot"
EVAL_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BRAIN_DIR))
sys.path.insert(0, str(CALLBOT_DIR))

from dotenv import load_dotenv
load_dotenv(CALLBOT_DIR / ".env")

from eval_utils import (
    load_eval_split, load_results, run_ragas, print_timing_table, timing_stats
)


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


async def run_one(llm, rag, sample: dict) -> dict:
    """Run one question through RAG + LLM. Returns result dict."""
    from core.prompt import build_prompt, NUTRITION_SYSTEM_PROMPT

    question = sample["question"]
    t_start = time.time()

    # RAG search
    t0 = time.time()
    docs = await rag.search(question)
    rag_ms = (time.time() - t0) * 1000

    contexts = [d["content"] for d in docs]
    context_str = "\n\n".join(
        f"[{d.get('source', '').upper()} — {d.get('title', '')}]\n{d.get('content', '')}"
        for d in docs
    )

    # LLM generation
    prompt = build_prompt(query=question, nutrition_context=context_str)
    t_llm = time.time()
    llm_ttft_ms = None
    answer_parts = []

    async for chunk in llm.generate_stream(
        prompt=prompt,
        system_instruction=NUTRITION_SYSTEM_PROMPT,
    ):
        if llm_ttft_ms is None:
            llm_ttft_ms = (time.time() - t_llm) * 1000
        answer_parts.append(chunk.get("text", ""))

    answer = "".join(answer_parts)
    total_ms = (time.time() - t_start) * 1000

    return {
        "id": sample["id"],
        "split": sample.get("split"),
        "source": sample.get("source"),
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "reference": sample["answer"],
        "timing": {
            "rag_ms": round(rag_ms, 1),
            "llm_ttft_ms": round(llm_ttft_ms or 0, 1),
            "total_ms": round(total_ms, 1),
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="LLM+RAG evaluation with RAGAS")
    parser.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        metavar="N", help="Which eval splits to run (default: all 5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per split (for quick testing)")
    parser.add_argument("--out", default="results/llm_eval_results.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--no-ragas", action="store_true",
                        help="Skip RAGAS scoring (only collect LLM outputs + latency)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip IDs already in output file")
    args = parser.parse_args()

    out_path = EVAL_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-done IDs if resuming
    done_ids = set()
    if args.resume:
        done_ids = {r["id"] for r in load_results(out_path)}
        print(f"Resume: {len(done_ids)} samples already done, skipping.")

    llm, rag, brain_config = await init_brain()

    all_results = list(load_results(out_path)) if args.resume else []

    for split_n in args.splits:
        jsonl_path = EVAL_DIR / f"eval_split_{split_n}.jsonl"
        if not jsonl_path.exists():
            print(f"SKIP: {jsonl_path} not found")
            continue

        samples = load_eval_split(jsonl_path)
        if args.limit:
            samples = samples[: args.limit]

        pending = [s for s in samples if s["id"] not in done_ids]
        print(f"\n=== Split {split_n}: {len(pending)}/{len(samples)} samples to run ===")

        for i, sample in enumerate(pending):
            print(f"  [{i+1}/{len(pending)}] {sample['id']} ...", end="", flush=True)
            try:
                result = await run_one(llm, rag, sample)
                all_results.append(result)
                done_ids.add(result["id"])

                t = result["timing"]
                print(f" rag={t['rag_ms']:.0f}ms  llm_ttft={t['llm_ttft_ms']:.0f}ms  total={t['total_ms']:.0f}ms")

                # Save incrementally so we can resume
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f" ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"Saved {len(all_results)} results → {out_path}")

    if not all_results:
        return

    # ── Latency summary ───────────────────────────────────────────────────────
    print("\n=== Latency Summary ===")
    print_timing_table(all_results)

    # ── RAGAS evaluation ──────────────────────────────────────────────────────
    if args.no_ragas:
        print("\nSkipping RAGAS (--no-ragas). Re-run without flag to get quality scores.")
        return

    print("\n=== Running RAGAS Evaluation (this may take a few minutes) ===")
    try:
        summary = run_ragas(all_results, brain_config.gemini_api_key, question_key="question")

        print("\n=== RAGAS Scores ===")
        for metric, score in summary.items():
            bar = "█" * int(score * 20)
            print(f"  {metric:<25} {score:.4f}  {bar}")

        # Save summary
        summary_path = out_path.parent / "llm_eval_ragas_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {"splits": args.splits, "n_samples": len(all_results), "ragas": summary},
                f, ensure_ascii=False, indent=2,
            )
        print(f"\nRAGAS summary → {summary_path}")

        # Update results file with per-sample RAGAS scores
        with open(out_path, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    except ImportError as e:
        print(f"\nRAGAS not installed: {e}")
        print("Install with:  pip install ragas>=0.2.0 langchain-google-genai langchain")


if __name__ == "__main__":
    asyncio.run(main())
