"""
=================================================================
Nutrition CallBot — Colab Embedding + Qdrant Ingest Pipeline
=================================================================
Script này chạy trên Google Colab (GPU T4) để:
1. Mount Google Drive → đọc nutrition_chunks.jsonl
2. Embed từng chunk bằng intfloat/multilingual-e5-base
3. Upload vectors lên Qdrant Cloud

HƯỚNG DẪN SỬ DỤNG:
─────────────────
Bước 1: Tạo Qdrant Cloud cluster miễn phí
  1. Vào https://cloud.qdrant.io → Sign up (free tier: 1GB, ~200K vectors)
  2. "Create Cluster" → Free tier → region Singapore/Tokyo
  3. Lấy URL + API Key từ cluster dashboard

Bước 2: Upload file lên Google Drive
  - Tạo thư mục "Nutrition-Callbot" trên Drive
  - Upload file nutrition_chunks.jsonl vào đó

Bước 3: Mở script này trên Colab
  - Runtime → Change runtime type → GPU (T4)
  - Chạy từng cell theo thứ tự

Bước 4: Điền QDRANT_URL, QDRANT_API_KEY ở Cell 3
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
# CELL 3: CONFIG — Sửa các giá trị này trước khi chạy
# ==============================================================

QDRANT_URL     = "https://YOUR-CLUSTER-URL.aws.cloud.qdrant.io"  # ← URL cluster
QDRANT_API_KEY = "YOUR-API-KEY"                                   # ← API key
COLLECTION     = "nutrition_articles"                             # tên collection

# Đường dẫn file trên Google Drive
DATA_PATH = "/content/drive/MyDrive/Nutrition-Callbot/nutrition_chunks.jsonl"

# Embedding
MODEL_NAME       = "intfloat/multilingual-e5-base"  # 768 dims, tốt cho tiếng Việt
EMBED_BATCH_SIZE = 64    # giảm xuống 32 nếu hết VRAM

# Giới hạn để test nhanh (None = toàn bộ ~92K chunks)
MAX_CHUNKS = None  # đặt = 200 để test trước khi chạy full


# ==============================================================
# CELL 4: Load chunks từ Google Drive
# ==============================================================
import json
import time

print(f"Loading {DATA_PATH}...")

chunks = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))

total = len(chunks)
if MAX_CHUNKS:
    chunks = chunks[:MAX_CHUNKS]
    print(f"Loaded {total:,} chunks → giới hạn test {len(chunks):,} chunks")
else:
    print(f"Loaded {total:,} chunks")

# Stats nhanh
sources = {}
for c in chunks:
    src = c["source"]
    sources[src] = sources.get(src, 0) + 1
print(f"\nPhân bổ:")
for src, cnt in sorted(sources.items()):
    print(f"  {src}: {cnt:,}")


# ==============================================================
# CELL 5: Load embedding model
# ==============================================================
from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print(f"\nDownloading model {MODEL_NAME}... (lần đầu ~2 phút)")
model = SentenceTransformer(MODEL_NAME, device=device)
print(f"Model loaded! Embedding dim: {model.get_sentence_embedding_dimension()}")

# Test nhanh
test_emb = model.encode(["passage: Dinh dưỡng là gì?"])
print(f"Test embedding shape: {test_emb.shape}")


# ==============================================================
# CELL 6: Kết nối Qdrant Cloud + tạo collection
# ==============================================================
from qdrant_client import QdrantClient
from qdrant_client.http import models
import hashlib, uuid

print(f"Kết nối Qdrant: {QDRANT_URL[:45]}...")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

# Kiểm tra kết nối
info = qdrant.get_collections()
print(f"Kết nối OK! Collections hiện có: {[c.name for c in info.collections]}")

DIM = model.get_sentence_embedding_dimension()  # 768

# Tạo mới collection (xóa nếu đã tồn tại)
if qdrant.collection_exists(COLLECTION):
    print(f"Xóa collection cũ: {COLLECTION}")
    qdrant.delete_collection(COLLECTION)

qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
)

# Index payload fields để filter nhanh
for field in ["source", "category", "doc_id"]:
    qdrant.create_payload_index(COLLECTION, field, field_schema="keyword")

print(f"Tạo collection '{COLLECTION}' (dim={DIM})")


# ==============================================================
# CELL 7: Embed + Upload (CELL CHÍNH — ~15-25 phút cho 92K chunks)
# ==============================================================
def make_uuid(chunk_id: str) -> str:
    """UUID deterministic từ chunk_id string."""
    h = hashlib.md5(chunk_id.encode()).hexdigest()
    return str(uuid.UUID(h))


def embed_and_upload(chunks_list, collection_name, batch_size=EMBED_BATCH_SIZE):
    total = len(chunks_list)
    uploaded = 0
    t_start = time.time()

    for batch_start in range(0, total, batch_size):
        batch = chunks_list[batch_start : batch_start + batch_size]

        # Dùng embed_text (= "title\ncontent") với prefix "passage:" theo chuẩn E5
        texts = [f"passage: {c['embed_text']}" for c in batch]

        # Embed batch
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        # Build Qdrant points
        points = []
        for i, chunk in enumerate(batch):
            points.append(
                models.PointStruct(
                    id=make_uuid(chunk["chunk_id"]),
                    vector=vecs[i].tolist(),
                    payload={
                        "chunk_id"   : chunk["chunk_id"],
                        "doc_id"     : chunk["doc_id"],
                        "source"     : chunk["source"],
                        "url"        : chunk["url"],
                        "title"      : chunk["title"],
                        "category"   : chunk["category"],
                        "chunk_index": chunk["chunk_index"],
                        "text"       : chunk["text"],       # text gốc để trả lời
                    },
                )
            )

        # Upload batch
        qdrant.upsert(collection_name=collection_name, points=points)
        uploaded += len(batch)

        # Progress
        elapsed = time.time() - t_start
        speed = uploaded / elapsed
        eta   = (total - uploaded) / speed if speed > 0 else 0
        if uploaded % (batch_size * 20) == 0 or uploaded == total:
            print(
                f"  {uploaded:,}/{total:,} ({uploaded/total*100:.1f}%) | "
                f"{speed:.0f} chunks/s | ETA: {eta/60:.1f} phút"
            )

    print(f"\nHoàn tất! {uploaded:,} vectors đã upload trong {(time.time()-t_start)/60:.1f} phút")


print("=" * 60)
print(f"BẮT ĐẦU EMBEDDING + UPLOAD ({len(chunks):,} chunks)")
print("=" * 60)

embed_and_upload(chunks, COLLECTION)

# Xác nhận số điểm trong Qdrant
info = qdrant.get_collection(COLLECTION)
print(f"\nQdrant collection '{COLLECTION}': {info.points_count:,} vectors")


# ==============================================================
# CELL 8: Verify — Test search thử
# ==============================================================
print("\n" + "="*60)
print("TEST SEARCH")
print("="*60)

def search(query: str, top_k: int = 3):
    q_vec = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
    )[0].tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=q_vec,
        limit=top_k,
        with_payload=True,
    )
    return results.points

# Test với 2 câu hỏi
test_queries = [
    "Trẻ em mấy tháng tuổi thì bắt đầu ăn dặm được?",
    "Người tiểu đường nên ăn gì?",
]

for q in test_queries:
    print(f"\nQuery: '{q}'")
    hits = search(q)
    for i, p in enumerate(hits):
        print(f"  {i+1}. [{p.score:.4f}] [{p.payload['source']}] {p.payload['title']}")
        print(f"       {p.payload['text'][:120]}...")

print("\nPipeline hoàn tất! Qdrant sẵn sàng cho Nutrition CallBot.")
