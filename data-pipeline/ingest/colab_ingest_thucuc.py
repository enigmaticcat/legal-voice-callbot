"""
=================================================================
Nutrition CallBot — Ingest thucuc chunks vào Qdrant (UPSERT)
=================================================================
Script này UPSERT thucuc_chunks.jsonl vào collection đã có sẵn —
KHÔNG xóa dữ liệu cũ (vinmec, skds, vdd).

Trước khi chạy:
  - Upload thucuc_chunks.jsonl lên Google Drive
  - Điền QDRANT_URL, QDRANT_API_KEY ở Cell 3
=================================================================
"""

# ==============================================================
# CELL 1: Cài đặt dependencies
# ==============================================================
# !pip install -q sentence-transformers qdrant-client


# ==============================================================
# CELL 2: Mount Google Drive
# ==============================================================
from google.colab import drive
drive.mount('/content/drive')


# ==============================================================
# CELL 3: CONFIG
# ==============================================================

QDRANT_URL     = "https://YOUR-CLUSTER-URL.aws.cloud.qdrant.io"
QDRANT_API_KEY = "YOUR-API-KEY"
COLLECTION     = "nutrition_articles"   # collection đã tồn tại

DATA_PATH = "/content/drive/MyDrive/Nutrition-Callbot/thucuc_chunks.jsonl"

MODEL_NAME       = "intfloat/multilingual-e5-base"  # phải khớp với model đã dùng lúc ingest ban đầu
EMBED_BATCH_SIZE = 64


# ==============================================================
# CELL 4: Load chunks
# ==============================================================
import json

chunks = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))

print(f"Loaded {len(chunks):,} thucuc chunks")


# ==============================================================
# CELL 5: Load model
# ==============================================================
from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = SentenceTransformer(MODEL_NAME, device=device)
print(f"Model: {MODEL_NAME} | dim={model.get_sentence_embedding_dimension()}")


# ==============================================================
# CELL 6: Kết nối Qdrant + kiểm tra collection tồn tại
# ==============================================================
from qdrant_client import QdrantClient
from qdrant_client.http import models
import hashlib, uuid, time

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

info = qdrant.get_collection(COLLECTION)
print(f"Collection '{COLLECTION}' hiện có: {info.points_count:,} vectors")
print("Sẽ UPSERT thucuc chunks vào collection này (không xóa dữ liệu cũ)")


# ==============================================================
# CELL 7: Embed + Upsert
# ==============================================================
def make_uuid(chunk_id: str) -> str:
    h = hashlib.md5(chunk_id.encode()).hexdigest()
    return str(uuid.UUID(h))


def embed_and_upsert(chunks_list):
    total = len(chunks_list)
    uploaded = 0
    t_start = time.time()

    for batch_start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks_list[batch_start : batch_start + EMBED_BATCH_SIZE]
        texts = [f"passage: {c['embed_text']}" for c in batch]
        vecs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        points = [
            models.PointStruct(
                id=make_uuid(c["chunk_id"]),
                vector=vecs[i].tolist(),
                payload={
                    "chunk_id"   : c["chunk_id"],
                    "doc_id"     : c["doc_id"],
                    "source"     : c["source"],
                    "url"        : c["url"],
                    "title"      : c["title"],
                    "category"   : c["category"],
                    "chunk_index": c["chunk_index"],
                    "text"       : c["text"],
                },
            )
            for i, c in enumerate(batch)
        ]

        qdrant.upsert(collection_name=COLLECTION, points=points)
        uploaded += len(batch)

        if uploaded % (EMBED_BATCH_SIZE * 10) == 0 or uploaded == total:
            elapsed = time.time() - t_start
            speed   = uploaded / elapsed
            eta     = (total - uploaded) / speed if speed > 0 else 0
            print(f"  {uploaded:,}/{total:,} ({uploaded/total*100:.1f}%) | {speed:.0f} c/s | ETA {eta/60:.1f} min")

    print(f"\nHoàn tất! {uploaded:,} vectors upserted trong {(time.time()-t_start)/60:.1f} phút")


print("=" * 60)
print(f"BẮT ĐẦU UPSERT {len(chunks):,} thucuc chunks")
print("=" * 60)
embed_and_upsert(chunks)


# ==============================================================
# CELL 8: Xác nhận
# ==============================================================
info = qdrant.get_collection(COLLECTION)
print(f"\nCollection '{COLLECTION}' sau upsert: {info.points_count:,} vectors")

# Test search
def search(query: str, top_k: int = 3, source_filter: str = None):
    q_vec = model.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()
    filter_ = None
    if source_filter:
        filter_ = models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=source_filter))]
        )
    results = qdrant.query_points(collection_name=COLLECTION, query=q_vec, limit=top_k,
                                  query_filter=filter_, with_payload=True)
    return results.points

print("\n--- Test search trong thucuc ---")
hits = search("Trẻ sơ sinh cần bổ sung vitamin D bao nhiêu?", source_filter="benhvienthucuc")
for i, p in enumerate(hits):
    print(f"  {i+1}. [{p.score:.4f}] {p.payload['title'][:60]}")
    print(f"       {p.payload['text'][:100]}...")
