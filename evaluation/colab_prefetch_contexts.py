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

# Đường dẫn các file eval trên Google Drive (có thể thêm/bớt)
INPUT_FILES = [
    "/content/drive/MyDrive/Nutrition-Callbot/eval_split_1.jsonl",
    "/content/drive/MyDrive/Nutrition-Callbot/eval_split_2.jsonl",
    "/content/drive/MyDrive/Nutrition-Callbot/eval_split_3.jsonl",
    "/content/drive/MyDrive/Nutrition-Callbot/eval_split_4.jsonl",
    "/content/drive/MyDrive/Nutrition-Callbot/eval_split_5.jsonl",
]

OUTPUT_FILE      = "/content/drive/MyDrive/Nutrition-Callbot/eval_with_contexts.jsonl"
SCORE_STATS_FILE = "/content/drive/MyDrive/Nutrition-Callbot/context_score_stats.jsonl"

TOP_K      = 5      # số contexts mỗi câu hỏi
RESUME     = True   # True = bỏ qua ID đã có trong OUTPUT_FILE


# ==============================================================
# CELL 3: Prefetch contexts
# ==============================================================
import json
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QUERY_PREFIX = "Instruct: Tìm thông tin dinh dưỡng liên quan\nQuery: "
MODEL_NAME   = "intfloat/multilingual-e5-large-instruct"

# Kết nối Qdrant
print("Kết nối Qdrant...")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
print(f"  Collection '{COLLECTION}': {qdrant.get_collection(COLLECTION).points_count:,} vectors")

# Load embedding model
print(f"\nLoad model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
print("  Sẵn sàng.")

# Load IDs đã xong (resume)
done_ids = set()
if RESUME and Path(OUTPUT_FILE).exists():
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done_ids.add(json.loads(line)["id"])
    print(f"\nResume: {len(done_ids)} câu đã có context, bỏ qua.")

# Prefetch
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
        print(f"\n=== {Path(input_path).name}: {len(pending)}/{len(samples)} câu ===")

        for i, sample in enumerate(pending):
            print(f"  [{i+1}/{len(pending)}] {sample['id']} ...", end="", flush=True)
            try:
                # Embed query
                q_vec = model.encode(
                    [QUERY_PREFIX + sample["question"]],
                    normalize_embeddings=True,
                )[0].tolist()

                # Search Qdrant
                results = qdrant.query_points(
                    collection_name=COLLECTION,
                    query=q_vec,
                    limit=TOP_K,
                    with_payload=True,
                )
                contexts = [p.payload.get("text", "") for p in results.points]
                scores   = [round(p.score, 4) for p in results.points]

                record = {
                    "id"       : sample["id"],
                    "split"    : sample.get("split"),
                    "source"   : sample.get("source"),
                    "question" : sample["question"],
                    "reference": sample["answer"],
                    "contexts" : contexts,
                }

                score_record = {
                    "id"        : sample["id"],
                    "split"     : sample.get("split"),
                    "source"    : sample.get("source"),
                    "question"  : sample["question"],
                    "scores"    : scores,
                    "score_max" : max(scores),
                    "score_min" : min(scores),
                    "score_avg" : round(sum(scores) / len(scores), 4),
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                fout_stats.write(json.dumps(score_record, ensure_ascii=False) + "\n")
                fout_stats.flush()
                done_ids.add(sample["id"])
                total_written += 1
                print(f" OK | scores: {scores}")

            except Exception as e:
                print(f" ERROR: {e}")

print(f"\nDone. Lưu {total_written} câu mới → {OUTPUT_FILE}")
print(f"Tổng trong file: {len(done_ids)} câu")
