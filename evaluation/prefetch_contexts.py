"""
prefetch_contexts.py
====================
Retrieve RAG contexts cho tất cả câu hỏi trong eval_split_{N}.jsonl
và lưu vào eval_with_contexts.jsonl — chạy một lần, dùng lại mãi.

Output mỗi dòng:
  { id, split, source, question, reference, contexts: [...] }

Dùng file này để:
  - Chạy RAGAS mà không cần kết nối Qdrant lại
  - Generate LLM answers offline từ contexts đã có sẵn

Chạy:
  python evaluation/prefetch_contexts.py
  python evaluation/prefetch_contexts.py --splits 1 2 --top-k 5
  python evaluation/prefetch_contexts.py --resume   # tiếp tục nếu bị gián đoạn
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent
BRAIN_DIR   = REPO_ROOT / "nutrition-callbot" / "brain"
CALLBOT_DIR = REPO_ROOT / "nutrition-callbot"
EVAL_DIR    = Path(__file__).resolve().parent

sys.path.insert(0, str(BRAIN_DIR))
sys.path.insert(0, str(CALLBOT_DIR))

from dotenv import load_dotenv
load_dotenv(CALLBOT_DIR / ".env")

from eval_utils import load_eval_split, load_results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default="results/eval_with_contexts.jsonl")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_path = EVAL_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load IDs đã xong
    done_ids = set()
    if args.resume:
        done_ids = {r["id"] for r in load_results(out_path)}
        print(f"Resume: {len(done_ids)} câu đã có context, bỏ qua.")

    # Khởi tạo RAGPipeline
    from config import config as brain_config
    from core.rag import RAGPipeline

    qdrant_kwargs = {}
    if brain_config.qdrant_path:
        qdrant_kwargs["qdrant_path"] = brain_config.qdrant_path
    elif brain_config.qdrant_url:
        qdrant_kwargs["qdrant_url"]     = brain_config.qdrant_url
        qdrant_kwargs["qdrant_api_key"] = brain_config.qdrant_api_key
    else:
        print("ERROR: QDRANT_URL hoặc QDRANT_PATH chưa set trong .env")
        sys.exit(1)

    print("Đang load RAGPipeline (embedding model ~1 phút lần đầu)...")
    rag = RAGPipeline(collection=brain_config.qdrant_collection, **qdrant_kwargs)
    print("RAGPipeline sẵn sàng.\n")

    total_written = 0

    with open(out_path, "a", encoding="utf-8") as fout:
        for split_n in args.splits:
            jsonl_path = EVAL_DIR / f"eval_split_{split_n}.jsonl"
            if not jsonl_path.exists():
                print(f"SKIP: {jsonl_path.name} không tồn tại")
                continue

            samples = load_eval_split(jsonl_path)
            pending = [s for s in samples if s["id"] not in done_ids]
            print(f"=== Split {split_n}: {len(pending)}/{len(samples)} câu cần fetch ===")

            for i, sample in enumerate(pending):
                print(f"  [{i+1}/{len(pending)}] {sample['id']} ...", end="", flush=True)
                try:
                    docs = await rag.search(sample["question"], top_k=args.top_k)
                    contexts = [d["content"] for d in docs]

                    record = {
                        "id"       : sample["id"],
                        "split"    : sample.get("split"),
                        "source"   : sample.get("source"),
                        "question" : sample["question"],
                        "reference": sample["answer"],
                        "contexts" : contexts,
                    }

                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    done_ids.add(sample["id"])
                    total_written += 1
                    print(f" {len(contexts)} contexts")

                except Exception as e:
                    print(f" ERROR: {e}")

    print(f"\nDone. Đã lưu {total_written} câu mới → {out_path}")
    print(f"Tổng trong file: {len(done_ids)} câu")


if __name__ == "__main__":
    asyncio.run(main())
