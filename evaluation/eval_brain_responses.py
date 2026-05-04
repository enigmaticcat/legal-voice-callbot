"""
Đánh giá câu trả lời của Brain service trên tập synthetic_qa.jsonl.

Usage:
  # Chạy thử 5 mẫu (nhanh, để kiểm tra trước)
  python eval_brain_responses.py --sample 5

  # Chạy toàn bộ 360 mẫu, concurrency 4
  python eval_brain_responses.py --concurrency 4

  # Warm up hệ thống trước (10 câu dummy)
  python eval_brain_responses.py --warmup-only

  # Full run với warmup
  python eval_brain_responses.py --warmup --concurrency 4

  # Chỉ định brain URL khác (ví dụ khi dùng port forwarding Lightning AI)
  python eval_brain_responses.py --brain-url http://localhost:50052 --sample 10
"""

import argparse
import asyncio
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("eval")

SYNTHETIC_QA_PATH = Path(__file__).parent / "synthetic_qa.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"

WARMUP_QUESTIONS = [
    "Ăn nhiều rau xanh có lợi gì cho sức khỏe?",
    "Trẻ em cần bổ sung canxi bao nhiêu mỗi ngày?",
    "Người tiểu đường nên ăn gì và kiêng gì?",
    "Omega-3 có trong thực phẩm nào?",
    "Cách bổ sung sắt cho người thiếu máu?",
    "Vitamin D có tác dụng gì với xương?",
    "Người cao tuổi cần chú ý gì trong chế độ ăn?",
    "Ăn sáng có quan trọng không?",
    "Protein thực vật khác protein động vật như thế nào?",
    "Uống bao nhiêu nước mỗi ngày là đủ?",
]


def load_samples(path: Path, n: int | None = None) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if n is not None:
        samples = samples[:n]
    return samples


def count_words(text: str) -> int:
    return len(text.split())


async def call_brain(
    client: httpx.AsyncClient,
    brain_url: str,
    question: str,
    session_id: str,
) -> dict:
    try:
        t0 = time.time()
        resp = await client.post(
            f"{brain_url}/think",
            json={"query": question, "session_id": session_id},
            timeout=120.0,
        )
        wall_ms = (time.time() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
        return {
            "success": True,
            "response": data.get("text", ""),
            "timing": data.get("timing", {}),
            "wall_ms": round(wall_ms, 1),
        }
    except Exception as e:
        return {
            "success": False,
            "response": "",
            "error": str(e),
            "timing": {},
            "wall_ms": 0,
        }


async def warmup(brain_url: str, concurrency: int = 3):
    logger.info("=== Warming up với %d câu dummy ===", len(WARMUP_QUESTIONS))
    sem = asyncio.Semaphore(concurrency)

    async def _one(q: str, i: int):
        async with sem:
            async with httpx.AsyncClient() as client:
                result = await call_brain(client, brain_url, q, f"warmup-{i}")
                status = "OK" if result["success"] else "ERR"
                timing = result.get("timing", {})
                logger.info(
                    "  [warmup %2d] %s | total=%.0fms | wall=%.0fms | %s",
                    i + 1,
                    status,
                    timing.get("total_ms", 0),
                    result["wall_ms"],
                    q[:40],
                )

    await asyncio.gather(*[_one(q, i) for i, q in enumerate(WARMUP_QUESTIONS)])
    logger.info("=== Warmup hoàn tất ===\n")


async def evaluate_all(
    samples: list[dict],
    brain_url: str,
    concurrency: int,
) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    results = [None] * len(samples)

    async def _one(idx: int, sample: dict):
        async with sem:
            async with httpx.AsyncClient() as client:
                result = await call_brain(
                    client,
                    brain_url,
                    sample["question"],
                    f"eval-{sample.get('id', idx)}",
                )
            timing = result.get("timing", {})
            log_parts = [
                f"[{idx+1:3d}/{len(samples)}]",
                "OK" if result["success"] else "ERR",
                f"total={timing.get('total_ms', 0):.0f}ms",
                f"rag={timing.get('rag_ms', 0):.0f}ms",
                f"ttft={timing.get('llm_ttft_ms', 0):.0f}ms",
                f"words={count_words(result['response'])}",
            ]
            logger.info("  " + " | ".join(log_parts))

            results[idx] = {
                "id": sample.get("id", f"sample_{idx}"),
                "source": sample.get("source", ""),
                "question": sample["question"],
                "reference_answer": sample.get("reference_answer", ""),
                "response": result["response"],
                "success": result["success"],
                "error": result.get("error"),
                "timing": timing,
                "wall_ms": result["wall_ms"],
                "response_word_count": count_words(result["response"]),
                "reference_word_count": count_words(sample.get("reference_answer", "")),
            }

    await asyncio.gather(*[_one(i, s) for i, s in enumerate(samples)])
    return results


def save_results(results: list[dict], out_dir: Path, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"brain_responses_{tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Kết quả đã lưu: %s", out_path)
    return out_path


def compute_metrics(results: list[dict]) -> dict:
    success = [r for r in results if r["success"]]
    errors = [r for r in results if not r["success"]]

    def _stat(values: list[float]) -> dict:
        if not values:
            return {}
        values = sorted(values)
        n = len(values)
        return {
            "mean": round(sum(values) / n, 1),
            "median": round(values[n // 2], 1),
            "p90": round(values[int(n * 0.9)], 1),
            "p95": round(values[int(n * 0.95)], 1),
            "min": round(values[0], 1),
            "max": round(values[-1], 1),
        }

    total_ms_list = [r["timing"].get("total_ms", r["wall_ms"]) for r in success]
    # first_chunk_total_ms = expand + rag + llm_ttft, đây là phần brain đóng góp vào TTFB thực tế
    first_chunk_list = [r["timing"]["first_chunk_total_ms"] for r in success if r["timing"].get("first_chunk_total_ms")]
    rag_ms_list = [r["timing"].get("rag_ms", 0) for r in success]
    ttft_ms_list = [r["timing"].get("llm_ttft_ms", 0) for r in success if r["timing"].get("llm_ttft_ms", 0) > 0]
    expand_ms_list = [r["timing"].get("expand_ms", 0) for r in success]
    word_count_list = [r["response_word_count"] for r in success]

    by_source: dict[str, list] = {}
    for r in success:
        src = r.get("source", "unknown")
        by_source.setdefault(src, []).append(r["timing"].get("first_chunk_total_ms") or r["timing"].get("total_ms", r["wall_ms"]))

    return {
        "total_samples": len(results),
        "success_count": len(success),
        "error_count": len(errors),
        "success_rate_pct": round(100 * len(success) / len(results), 1) if results else 0,
        "latency_first_chunk_ms": _stat(first_chunk_list),   # ← đây là metric TTFB-relevant
        "latency_total_ms": _stat(total_ms_list),
        "latency_rag_ms": _stat(rag_ms_list),
        "latency_ttft_ms": _stat(ttft_ms_list),
        "latency_expand_ms": _stat(expand_ms_list),
        "response_word_count": _stat(word_count_list),
        "errors": [{"id": r["id"], "error": r.get("error")} for r in errors],
        "by_source": {src: _stat(vals) for src, vals in by_source.items()},
    }


def save_metrics(metrics: dict, out_dir: Path, tag: str) -> Path:
    out_path = out_dir / f"metrics_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("Metrics đã lưu: %s", out_path)
    return out_path


def print_summary(metrics: dict):
    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ BRAIN SERVICE")
    print("=" * 60)
    print(f"  Tổng mẫu    : {metrics['total_samples']}")
    print(f"  Thành công  : {metrics['success_count']} ({metrics['success_rate_pct']}%)")
    print(f"  Lỗi         : {metrics['error_count']}")

    def _row(label, stat):
        if not stat:
            return
        print(f"  {label:<20}: mean={stat['mean']}ms  p90={stat['p90']}ms  max={stat['max']}ms")

    print("\n  --- Latency (TTFB-relevant: brain contribution) ---")
    _row("Brain first chunk", metrics.get("latency_first_chunk_ms", {}))
    print("    (= expand + RAG + LLM TTFT — cộng thêm ASR ~100-200ms + TTS ~200-300ms = TTFB)")
    print("\n  --- Latency chi tiết ---")
    _row("RAG search", metrics["latency_rag_ms"])
    _row("LLM TTFT", metrics["latency_ttft_ms"])
    _row("Query expand", metrics["latency_expand_ms"])
    print("\n  --- Latency toàn bộ (full generation) ---")
    _row("Total brain", metrics["latency_total_ms"])

    wc = metrics["response_word_count"]
    if wc:
        print(f"\n  --- Độ dài câu trả lời ---")
        print(f"  {'Word count':<20}: mean={wc['mean']}  p90={wc['p90']}  max={wc['max']}")

    if metrics.get("by_source"):
        print("\n  --- Latency theo nguồn ---")
        for src, stat in metrics["by_source"].items():
            if stat:
                print(f"  {src:<20}: mean={stat['mean']}ms  p90={stat['p90']}ms")

    if metrics["errors"]:
        print(f"\n  --- Lỗi ({len(metrics['errors'])} mẫu) ---")
        for e in metrics["errors"][:5]:
            print(f"  [{e['id']}] {e['error']}")
        if len(metrics["errors"]) > 5:
            print(f"  ... và {len(metrics['errors']) - 5} lỗi khác")
    print("=" * 60 + "\n")


def generate_charts(results: list[dict], metrics: dict, out_dir: Path, tag: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        logger.warning(
            "Khong ve duoc do thi (matplotlib/numpy conflict). "
            "Fix: pip install -U matplotlib  hoac  pip install 'numpy<2'"
        )
        return

    success = [r for r in results if r["success"]]
    if not success:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Brain Service Evaluation — {tag}\n({len(success)}/{len(results)} samples)", fontsize=13)

    first_chunk_ms = [r["timing"]["first_chunk_total_ms"] for r in success if r["timing"].get("first_chunk_total_ms")]
    total_ms = [r["timing"].get("total_ms", r["wall_ms"]) for r in success]
    rag_ms = [r["timing"].get("rag_ms", 0) for r in success]
    ttft_ms = [r["timing"].get("llm_ttft_ms", 0) for r in success if r["timing"].get("llm_ttft_ms", 0) > 0]
    word_counts = [r["response_word_count"] for r in success]

    def _hist(ax, data, title, xlabel, color="steelblue", bins=30):
        if not data:
            ax.set_title(title + " (no data)")
            return
        ax.hist(data, bins=min(bins, len(data)), color=color, edgecolor="white", linewidth=0.5)
        ax.axvline(np.mean(data), color="red", linestyle="--", linewidth=1.5, label=f"mean={np.mean(data):.0f}")
        ax.axvline(np.percentile(data, 90), color="orange", linestyle="--", linewidth=1.2, label=f"p90={np.percentile(data, 90):.0f}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    # [0,0] Brain first chunk = TTFB contribution (key UX metric)
    _hist(axes[0, 0], first_chunk_ms, "Brain First Chunk (TTFB contribution)", "ms", "darkorange")
    _hist(axes[0, 1], rag_ms, "RAG Search Latency", "ms", "teal")
    _hist(axes[0, 2], ttft_ms, "LLM Time-to-First-Token", "ms", "mediumorchid")
    _hist(axes[1, 0], word_counts, "Response Word Count", "words", "coral")

    # CDF of first_chunk_ms (most meaningful for UX)
    ax_cdf = axes[1, 1]
    cdf_data = first_chunk_ms if first_chunk_ms else total_ms
    cdf_label = "First Chunk" if first_chunk_ms else "Total"
    sorted_ms = sorted(cdf_data)
    cdf = np.arange(1, len(sorted_ms) + 1) / len(sorted_ms)
    ax_cdf.plot(sorted_ms, cdf, color="darkorange", linewidth=2)
    for p in [50, 90, 95]:
        pv = np.percentile(sorted_ms, p)
        ax_cdf.axvline(pv, linestyle="--", linewidth=1, alpha=0.7, label=f"p{p}={pv:.0f}ms")
    ax_cdf.set_title(f"CDF — Brain {cdf_label} Latency")
    ax_cdf.set_xlabel("ms")
    ax_cdf.set_ylabel("Cumulative fraction")
    ax_cdf.legend(fontsize=8)
    ax_cdf.grid(True, alpha=0.3)

    # Latency by source (box plot) — dùng first_chunk_ms nếu có
    ax_src = axes[1, 2]
    by_source: dict[str, list] = {}
    for r in success:
        src = r.get("source", "unknown")
        v = r["timing"].get("first_chunk_total_ms") or r["timing"].get("total_ms", r["wall_ms"])
        by_source.setdefault(src, []).append(v)
    if by_source:
        labels = list(by_source.keys())
        data_by_src = [by_source[k] for k in labels]
        ax_src.boxplot(data_by_src, labels=labels)
        ax_src.set_title("Total Latency by Source")
        ax_src.set_ylabel("ms")
        ax_src.tick_params(axis="x", rotation=15)
    else:
        ax_src.set_title("No source data")

    plt.tight_layout()
    chart_path = out_dir / f"charts_{tag}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Đồ thị đã lưu: %s", chart_path)


async def main():
    parser = argparse.ArgumentParser(description="Evaluate Brain service on synthetic_qa.jsonl")
    parser.add_argument("--brain-url", default="http://localhost:50052", help="Brain HTTP base URL")
    parser.add_argument("--sample", type=int, default=None, help="Số mẫu để chạy (mặc định: toàn bộ)")
    parser.add_argument("--concurrency", type=int, default=2, help="Số request song song (mặc định: 2)")
    parser.add_argument("--warmup", action="store_true", help="Warm up trước khi eval")
    parser.add_argument("--warmup-only", action="store_true", help="Chỉ warm up, không eval")
    parser.add_argument("--input", default=str(SYNTHETIC_QA_PATH), help="Đường dẫn file jsonl input")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR), help="Thư mục lưu kết quả")
    args = parser.parse_args()

    brain_url = args.brain_url.rstrip("/")
    out_dir = Path(args.out_dir)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.sample:
        tag += f"_n{args.sample}"

    # Kiểm tra brain health
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{brain_url}/health", timeout=10)
            resp.raise_for_status()
        logger.info("Brain health OK: %s", brain_url)
    except Exception as e:
        logger.error("Không kết nối được Brain tại %s: %s", brain_url, e)
        logger.error("Hãy đảm bảo brain đang chạy (./scripts/start_local_4_services.sh hoặc docker compose up)")
        return

    if args.warmup or args.warmup_only:
        await warmup(brain_url, concurrency=min(3, args.concurrency))

    if args.warmup_only:
        return

    samples = load_samples(Path(args.input), args.sample)
    logger.info("Đánh giá %d mẫu | concurrency=%d | brain=%s", len(samples), args.concurrency, brain_url)

    t_start = time.time()
    results = await evaluate_all(samples, brain_url, concurrency=args.concurrency)
    elapsed = time.time() - t_start

    logger.info("Hoàn tất trong %.1fs (%.2f câu/giây)", elapsed, len(results) / elapsed)

    results_path = save_results(results, out_dir, tag)
    metrics = compute_metrics(results)
    save_metrics(metrics, out_dir, tag)
    print_summary(metrics)
    generate_charts(results, metrics, out_dir, tag)

    print(f"File kết quả : {results_path}")
    print(f"File metrics : {out_dir / f'metrics_{tag}.json'}")
    print(f"Đồ thị       : {out_dir / f'charts_{tag}.png'}")


if __name__ == "__main__":
    asyncio.run(main())
