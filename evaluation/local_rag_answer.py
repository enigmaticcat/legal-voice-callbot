# -*- coding: utf-8 -*-
"""
local_rag_answer.py
===================
BM25 retrieval trên corpus_final.jsonl → trả lời câu hỏi eval bằng Claude API.

Chạy:
  python evaluation/local_rag_answer.py --n 20 --out evaluation/results/local_rag_sample.jsonl
  python evaluation/local_rag_answer.py --ids thucuc_s1_008 thucuc_s5_002
  python evaluation/local_rag_answer.py --all
"""

import argparse, json, random, sys, time
from pathlib import Path
from rank_bm25 import BM25Okapi

REPO_ROOT   = Path(__file__).resolve().parent.parent
CORPUS_FILE = REPO_ROOT / "data_final" / "corpus_final.jsonl"
EVAL_FILES  = [REPO_ROOT / "evaluation" / f"eval_split_{i}.jsonl" for i in range(1, 6)]
OUT_DEFAULT = REPO_ROOT / "evaluation" / "results" / "local_rag_sample.jsonl"

TOP_K = 5


# ── tokenizer tiếng Việt đơn giản ──────────────────────────────────────────
def tokenize(text: str):
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


# ── load corpus ─────────────────────────────────────────────────────────────
def load_corpus(path: Path):
    print("Đang load corpus...", flush=True)
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                chunks.append({
                    "chunk_id": row.get("chunk_id", ""),
                    "url":      row.get("url", ""),
                    "source":   row.get("source", ""),
                    "text":     row.get("text", ""),
                })
    print(f"Loaded {len(chunks):,} chunks.", flush=True)
    return chunks


# ── build BM25 index ────────────────────────────────────────────────────────
def build_index(chunks):
    print("Đang build BM25 index...", flush=True)
    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    print("Index sẵn sàng.", flush=True)
    return bm25


# ── retrieve top-k ──────────────────────────────────────────────────────────
def retrieve(bm25, chunks, query: str, top_k: int = TOP_K):
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [
        {**chunks[i], "bm25_score": round(float(scores[i]), 4)}
        for i in top_idx
    ]


# ── call Claude API ─────────────────────────────────────────────────────────
def call_claude(question: str, contexts: list[str]) -> str:
    try:
        import anthropic
    except ImportError:
        return "[ERROR: anthropic package chưa cài — pip install anthropic]"

    client = anthropic.Anthropic()
    ctx_text = "\n\n---\n\n".join(
        f"[Nguồn {i+1}]\n{c}" for i, c in enumerate(contexts)
    )
    system = (
        "Bạn là chuyên gia dinh dưỡng lâm sàng. "
        "Trả lời câu hỏi DỰA TRÊN các đoạn văn được cung cấp. "
        "Nếu thông tin không đủ, nói rõ. "
        "Trả lời ngắn gọn, súc tích, đúng trọng tâm bằng tiếng Việt."
    )
    prompt = f"Các đoạn văn tham khảo:\n{ctx_text}\n\nCâu hỏi: {question}\n\nTrả lời:"

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# ── load eval samples ────────────────────────────────────────────────────────
def load_eval(ids_filter=None):
    samples = []
    for path in EVAL_FILES:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if ids_filter is None or row["id"] in ids_filter:
                        samples.append(row)
    return samples


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=None,
                        help="Lấy ngẫu nhiên N câu từ eval")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="Chỉ định ID cụ thể")
    parser.add_argument("--all", action="store_true",
                        help="Chạy toàn bộ 604 câu")
    parser.add_argument("--out", default=str(OUT_DEFAULT),
                        help="File output JSONL")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--resume", action="store_true",
                        help="Bỏ qua ID đã có trong --out")
    args = parser.parse_args()

    # Load
    chunks = load_corpus(CORPUS_FILE)
    bm25   = build_index(chunks)

    # Eval samples
    if args.ids:
        samples = load_eval(set(args.ids))
    elif args.all:
        samples = load_eval()
    else:
        samples = load_eval()
        n = args.n or 20
        random.seed(42)
        samples = random.sample(samples, min(n, len(samples)))

    # Resume
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["id"])
        print(f"Resume: {len(done_ids)} câu đã xong.")

    pending = [s for s in samples if s["id"] not in done_ids]
    print(f"\nSẽ xử lý {len(pending)} câu hỏi (top_k={args.top_k})\n")

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, sample in enumerate(pending):
            print(f"[{i+1}/{len(pending)}] {sample['id']} ...", end="", flush=True)

            docs = retrieve(bm25, chunks, sample["question"], top_k=args.top_k)
            contexts = [d["text"] for d in docs]

            answer = call_claude(sample["question"], contexts)

            record = {
                "id":        sample["id"],
                "question":  sample["question"],
                "reference": sample["answer"],
                "contexts":  contexts,
                "bm25_scores": [d["bm25_score"] for d in docs],
                "context_urls": [d["url"] for d in docs],
                "generated_answer": answer,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            print(f" OK | scores={[d['bm25_score'] for d in docs]}")
            time.sleep(0.3)   # tránh rate limit

    print(f"\nDone → {out_path}")


if __name__ == "__main__":
    main()
