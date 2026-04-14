# ==============================================================
# CELL 1: Cài dependencies
# ==============================================================
# !pip install -q qdrant-client sentence-transformers


# ==============================================================
# CELL 2: CONFIG — sửa các đường dẫn và key trước khi chạy
# ==============================================================

QDRANT_URL     = "https://YOUR-CLUSTER-URL.aws.cloud.qdrant.io"
QDRANT_API_KEY = "YOUR-API-KEY"
COLLECTION     = "nutrition_articles"

INPUT_FILES = [
    "/content/drive/MyDrive/Nutrition-Callbot/eval_split_1.jsonl",
    "/content/drive/MyDrive/Nutrition-Callbot/eval_split_2.jsonl",
    "/content/drive/MyDrive/Nutrition-Callbot/eval_split_3.jsonl",
    "/content/drive/MyDrive/Nutrition-Callbot/eval_split_4.jsonl",
    "/content/drive/MyDrive/Nutrition-Callbot/eval_split_5.jsonl",
]

OUTPUT_FILE      = "/content/drive/MyDrive/Nutrition-Callbot/eval_with_contexts.jsonl"
SCORE_STATS_FILE = "/content/drive/MyDrive/Nutrition-Callbot/context_score_stats.jsonl"

TOP_K             = 5     # số contexts cuối cùng truyền vào LLM
FETCH_K           = 20    # lấy rộng từ Qdrant trước khi rerank (nên = TOP_K * 4)
RERANK_MODEL      = "BAAI/bge-reranker-v2-m3"
RERANK_THRESHOLD  = 0.5   # loại chunk có rerank score < threshold; luôn giữ ít nhất 1 chunk
RESUME            = True


# ==============================================================
# CELL 3: Load models và kết nối Qdrant
# ==============================================================
import json
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

QUERY_PREFIX = "Instruct: Tìm thông tin dinh dưỡng liên quan\nQuery: "
EMBED_MODEL  = "intfloat/multilingual-e5-large-instruct"

print("Kết nối Qdrant...")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
print(f"  Collection '{COLLECTION}': {qdrant.get_collection(COLLECTION).points_count:,} vectors")

print(f"\nLoad embedding model: {EMBED_MODEL}...")
embedder = SentenceTransformer(EMBED_MODEL)
print("  Embedding model sẵn sàng.")

print(f"\nLoad reranker: {RERANK_MODEL}...")
reranker = CrossEncoder(RERANK_MODEL)
print("  Reranker sẵn sàng.\n")


# ==============================================================
# CELL 4: Prefetch + Rerank contexts
# ==============================================================

# Load IDs đã xong (resume)
done_ids = set()
if RESUME and Path(OUTPUT_FILE).exists():
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done_ids.add(json.loads(line)["id"])
    print(f"Resume: {len(done_ids)} câu đã có context, bỏ qua.\n")

total_written = 0

with open(OUTPUT_FILE, "a", encoding="utf-8") as fout, \
     open(SCORE_STATS_FILE, "a", encoding="utf-8") as fout_stats:

    for input_path in INPUT_FILES:
        if not Path(input_path).exists():
            print(f"SKIP: {input_path} không tồn tại")
            continue

        samples = []
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        pending = [s for s in samples if s["id"] not in done_ids]
        print(f"=== {Path(input_path).name}: {len(pending)}/{len(samples)} câu ===")

        for i, sample in enumerate(pending):
            print(f"  [{i+1}/{len(pending)}] {sample['id']} ...", end="", flush=True)
            try:
                # Bước 1: embed query
                q_vec = embedder.encode(
                    [QUERY_PREFIX + sample["question"]],
                    normalize_embeddings=True,
                )[0].tolist()

                # Bước 2: lấy FETCH_K candidates từ Qdrant
                results = qdrant.query_points(
                    collection_name=COLLECTION,
                    query=q_vec,
                    limit=FETCH_K,
                    with_payload=True,
                )
                candidates = results.points
                embed_scores = [round(p.score, 4) for p in candidates]

                # Bước 3: rerank bằng cross-encoder
                texts = [p.payload.get("text", "") for p in candidates]
                pairs = [[sample["question"], t] for t in texts]
                rerank_scores = reranker.predict(pairs, batch_size=32).tolist()

                # Bước 4: sort theo rerank score, lấy top TOP_K
                ranked = sorted(
                    zip(candidates, texts, embed_scores, rerank_scores),
                    key=lambda x: -x[3],
                )[:TOP_K]

                # Áp threshold: luôn giữ chunk tốt nhất, bỏ các chunk dưới ngưỡng
                ranked_filtered = ranked[:1] + [
                    r for r in ranked[1:] if r[3] >= RERANK_THRESHOLD
                ]

                contexts        = [r[1] for r in ranked_filtered]
                final_embed_sc  = [r[2] for r in ranked_filtered]
                final_rerank_sc = [round(r[3], 4) for r in ranked_filtered]
                final_chunk_ids = [r[0].payload.get("chunk_id", str(r[0].id)) for r in ranked_filtered]

                record = {
                    "id"       : sample["id"],
                    "split"    : sample.get("split"),
                    "source"   : sample.get("source"),
                    "question" : sample["question"],
                    "reference": sample["answer"],
                    "contexts" : contexts,
                }

                score_record = {
                    "id"              : sample["id"],
                    "split"           : sample.get("split"),
                    "source"          : sample.get("source"),
                    "question"        : sample["question"],
                    "embed_scores"    : final_embed_sc,
                    "rerank_scores"   : final_rerank_sc,
                    "rerank_score_max": max(final_rerank_sc),
                    "rerank_score_min": min(final_rerank_sc),
                    "rerank_score_avg": round(sum(final_rerank_sc) / len(final_rerank_sc), 4),
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                fout_stats.write(json.dumps(score_record, ensure_ascii=False) + "\n")
                fout_stats.flush()
                done_ids.add(sample["id"])
                total_written += 1
                print(f" OK | rerank: {final_rerank_sc} | chunks: {final_chunk_ids}")

            except Exception as e:
                print(f" ERROR: {e}")

print(f"\nDone. Lưu {total_written} câu mới → {OUTPUT_FILE}")
print(f"Tổng trong file: {len(done_ids)} câu")
