"""
=================================================================
Legal CallBot — Colab Embedding Pipeline
=================================================================
Notebook này chạy trên Google Colab (GPU runtime) để:
1. Mount Google Drive → đọc law_data.json
2. Chunk dữ liệu bằng LegalChunker
3. Embed bằng BAAI/bge-m3
4. Upload vectors lên Qdrant Cloud

HƯỚNG DẪN SỬ DỤNG:
─────────────────
Bước 1: Tạo Qdrant Cloud cluster miễn phí
  1. Vào https://cloud.qdrant.io → Sign up (free)
  2. Nhấn "Create Cluster" → chọn Free tier (1GB, đủ cho ~170K vectors)
  3. Đặt tên cluster, chọn region gần nhất (Singapore/Tokyo)
  4. Sau khi tạo xong, nhấn vào cluster → lấy:
     - URL:     https://xxx-xxx.aws.cloud.qdrant.io
     - API Key: nhấn "API Keys" → "Create" → copy key

Bước 2: Upload law_data.json lên Google Drive
  - Vào drive.google.com
  - Tạo thư mục "Legal-Callbot"
  - Upload file law_data.json (180MB) vào đó

Bước 3: Mở notebook này trên Colab
  - Runtime → Change runtime type → GPU (T4)
  - Chạy từng cell theo thứ tự

Bước 4: Điền QDRANT_URL và QDRANT_API_KEY ở Cell Config bên dưới
=================================================================
"""

# ==============================================================
# CELL 1: Cài đặt dependencies
# ==============================================================
# !pip install -q FlagEmbedding qdrant-client

# ==============================================================
# CELL 2: Mount Google Drive
# ==============================================================
from google.colab import drive
drive.mount('/content/drive')

# ==============================================================
# CELL 3: CONFIG — Điền thông tin Qdrant Cloud ở đây
# ==============================================================

# THAY ĐỔI 2 DÒNG NÀY:
QDRANT_URL = "https://YOUR-CLUSTER-URL.aws.cloud.qdrant.io"  # ← Paste URL cluster
QDRANT_API_KEY = "YOUR-API-KEY"                               # ← Paste API key

# Đường dẫn file trên Google Drive
DATA_PATH = "/content/drive/MyDrive/Legal-Callbot/law_data.json"

# Chunking parameters
MIN_WORDS = 80
MAX_WORDS = 500
OVERLAP_WORDS = 30

# Embedding batch size (giảm nếu hết VRAM)
EMBED_BATCH_SIZE = 32

# Giới hạn số entries (None = toàn bộ, đặt số nhỏ để test trước)
MAX_ENTRIES = None  # Đặt = 100 để test nhanh, None để chạy toàn bộ


# ==============================================================
# CELL 4: Legal Chunker (paste từ legal_chunker.py)
# ==============================================================
import re
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger("legal_chunker")

SPLIT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'\n\n(?=\d+\.\s)'),           "Khoản"),
    (re.compile(r'\n\n(?=[a-zđ]\)\s)'),          "Điểm"),
    (re.compile(r'\n\n(?=\d+\.\d+\.\s)'),       "Mục"),
    (re.compile(r'\n\n(?=\d+\.\d+\.\d+\.\s)'),  "Tiểu_mục"),
    (re.compile(r'\n\n(?=[a-zđ]\.\s)'),          "Điểm_chấm"),
    (re.compile(r'\n\n(?=-\s)'),                 "Gạch"),
    (re.compile(r'\n\n(?=\+\s)'),                "Cộng"),
    (re.compile(r'\n\n'),                        "Đoạn"),
]

LABEL_DETECTORS = [
    (re.compile(r'^(\d+)\.\s'),              lambda m: f"Khoản {m.group(1)}"),
    (re.compile(r'^([a-zđ])\)\s'),            lambda m: f"Điểm {m.group(1)}"),
    (re.compile(r'^(\d+\.\d+)\.\s'),         lambda m: f"Mục {m.group(1)}"),
    (re.compile(r'^(\d+\.\d+\.\d+)\.\s'),    lambda m: f"Tiểu mục {m.group(1)}"),
    (re.compile(r'^([a-zđ])\.\s'),            lambda m: f"Điểm {m.group(1)}"),
]


def _word_count(text: str) -> int:
    return len(text.split())


def _detect_chunk_label(text: str) -> str:
    stripped = text.strip()
    for pattern, label_fn in LABEL_DETECTORS:
        m = pattern.match(stripped)
        if m:
            return label_fn(m)
    return ""


class LegalChunker:
    def __init__(self, min_words=80, max_words=500, overlap_words=30, use_contextual_enrichment=True):
        self.min_words = min_words
        self.max_words = max_words
        self.overlap_words = overlap_words
        self.use_contextual_enrichment = use_contextual_enrichment

    def extract_chunks(self, mapc, ten_dieu, noidung, metadata):
        noidung = (noidung or "").strip()
        parent_chunk = {
            "id": f"{mapc}_parent", "type": "parent", "mapc": mapc,
            "text": noidung, "ten_dieu": ten_dieu, "chunk_label": "", "metadata": metadata,
        }
        if _word_count(noidung) <= self.max_words:
            child = self._make_child(mapc, ten_dieu, noidung, metadata, 0, "")
            return [parent_chunk, child]

        raw = self._recursive_split(noidung, 0)
        merged = self._merge_small(raw)
        sized = self._hard_split(merged)
        final = self._add_overlap(sized)

        children = []
        for idx, text in enumerate(final):
            text = text.strip()
            if len(text) < 10:
                continue
            children.append(self._make_child(mapc, ten_dieu, text, metadata, idx, _detect_chunk_label(text)))

        if not children:
            children = [self._make_child(mapc, ten_dieu, noidung, metadata, 0, "")]
        return [parent_chunk] + children

    def _recursive_split(self, text, idx):
        if _word_count(text) <= self.max_words or idx >= len(SPLIT_PATTERNS):
            return [text]
        regex, _ = SPLIT_PATTERNS[idx]
        parts = [p for p in regex.split(text) if p.strip()]
        if len(parts) <= 1:
            return self._recursive_split(text, idx + 1)
        result = []
        for p in parts:
            if _word_count(p) > self.max_words:
                result.extend(self._recursive_split(p, idx + 1))
            else:
                result.append(p)
        return result

    def _merge_small(self, chunks):
        if not chunks:
            return chunks
        merged, buf = [], chunks[0]
        for i in range(1, len(chunks)):
            c = chunks[i]
            combined = buf + "\n\n" + c
            if _word_count(buf) < self.min_words and _word_count(combined) <= self.max_words:
                buf = combined
            elif _word_count(c) < self.min_words and _word_count(combined) <= self.max_words:
                buf = combined
            else:
                merged.append(buf)
                buf = c
        merged.append(buf)
        if len(merged) > 1 and _word_count(merged[-1]) < self.min_words:
            combined = merged[-2] + "\n\n" + merged[-1]
            if _word_count(combined) <= self.max_words * 1.2:
                merged[-2] = combined
                merged.pop()
        return merged

    def _hard_split(self, chunks):
        result = []
        for c in chunks:
            if _word_count(c) <= self.max_words:
                result.append(c)
            else:
                sents = re.split(r'(?<=[.!?;])\s+', c)
                buf = ""
                for s in sents:
                    cand = (buf + " " + s).strip() if buf else s
                    if _word_count(cand) <= self.max_words:
                        buf = cand
                    else:
                        if buf: result.append(buf)
                        if _word_count(s) > self.max_words:
                            words = s.split()
                            for i in range(0, len(words), self.max_words):
                                result.append(" ".join(words[i:i+self.max_words]))
                            buf = ""
                        else:
                            buf = s
                if buf: result.append(buf)
        return result

    def _add_overlap(self, chunks):
        if self.overlap_words <= 0 or len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_words = chunks[i-1].split()
            ov = min(self.overlap_words, len(prev_words))
            result.append(f"...{' '.join(prev_words[-ov:])}\n\n{chunks[i]}")
        return result

    def _make_child(self, mapc, ten_dieu, text, metadata, idx, label):
        enriched = self._enrich(ten_dieu, text.strip())
        return {
            "id": f"{mapc}_child_{idx}", "type": "child",
            "parent_id": f"{mapc}_parent", "mapc": mapc,
            "text": enriched, "ten_dieu": ten_dieu,
            "chunk_label": label, "metadata": metadata,
        }

    def _enrich(self, ten_dieu, text):
        if not self.use_contextual_enrichment:
            return text
        parts = ten_dieu.rsplit(".", 1)
        if len(parts) == 2 and parts[1].strip():
            return f"[{parts[1].strip()}] {text}"
        return f"[{ten_dieu}] {text}"


# ==============================================================
# CELL 5: Load data + chạy chunking
# ==============================================================
import json
import time

print("Loading law_data.json từ Google Drive...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

total = len(raw_data)
target = min(MAX_ENTRIES, total) if MAX_ENTRIES else total
print(f"Loaded {total} entries. Sẽ xử lý {target} entries.\n")

chunker = LegalChunker(min_words=MIN_WORDS, max_words=MAX_WORDS, overlap_words=OVERLAP_WORDS)

parents = []
children = []

t0 = time.time()
for i, record in enumerate(raw_data[:target]):
    nd = record.get("noidung", "")
    if not nd.strip():
        continue

    mapc = record.get("mapc", "")
    ten = record.get("ten", "")
    meta = {
        "chude": record.get("chude", ""),
        "demuc": record.get("demuc", ""),
        "chuong": record.get("chuong", ""),
        "vbqppl": record.get("vbqppl", ""),
        "stt": record.get("stt", 0),
    }

    chunks = chunker.extract_chunks(mapc, ten, nd, meta)
    for c in chunks:
        if c["type"] == "parent":
            parents.append(c)
        else:
            children.append(c)

    if (i + 1) % 10000 == 0:
        print(f"  Chunked {i+1}/{target}...")

chunk_time = time.time() - t0
print(f"\nChunking xong trong {chunk_time:.1f}s")
print(f"   Parents:  {len(parents):,}")
print(f"   Children: {len(children):,}")
print(f"   Tổng texts cần embed: {len(parents) + len(children):,}")


# ==============================================================
# CELL 6: Load BGE-M3 model
# ==============================================================
from FlagEmbedding import BGEM3FlagModel

print("Downloading & loading BAAI/bge-m3... (lần đầu ~3 phút)")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
print("Model loaded!")

# Quick test
test_emb = model.encode(["Xin chào"], return_dense=True, return_sparse=True, return_colbert_vecs=False)
print(f"   Dense dim: {len(test_emb['dense_vecs'][0])}")
print(f"   Sparse keys: {len(test_emb['lexical_weights'][0])}")


# ==============================================================
# CELL 7: Kết nối Qdrant Cloud + tạo collections
# ==============================================================
from qdrant_client import QdrantClient
from qdrant_client.http import models
import hashlib
import uuid

print(f"Kết nối Qdrant Cloud: {QDRANT_URL[:40]}...")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

# Test connection
collections = qdrant.get_collections()
print(f"Kết nối thành công! Collections hiện có: {[c.name for c in collections.collections]}")

# Tạo 2 collections
COLL_PARENT = "phap_dien_dieu"
COLL_CHILD = "phap_dien_khoan"

for coll in [COLL_PARENT, COLL_CHILD]:
    if qdrant.collection_exists(coll):
        qdrant.delete_collection(coll)
        print(f"  Đã xóa collection cũ: {coll}")

    qdrant.create_collection(
        collection_name=coll,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    # Tạo index cho filter search
    qdrant.create_payload_index(coll, "chude", field_schema="keyword")
    qdrant.create_payload_index(coll, "chuong", field_schema="keyword")
    qdrant.create_payload_index(coll, "ten_dieu", field_schema="keyword")
    print(f"  Tạo collection: {coll}")


# ==============================================================
# CELL 8: Embedding + Upload (ĐÂY LÀ CELL CHÍNH — CHẠY ~20 PHÚT)
# ==============================================================
def make_uuid(chunk_id: str) -> str:
    """Tạo UUID deterministic từ chunk ID string."""
    h = hashlib.md5(chunk_id.encode()).hexdigest()
    return str(uuid.UUID(h))


def embed_and_upload(chunks_list, collection_name, batch_size=EMBED_BATCH_SIZE):
    """Embed danh sách chunks và upload lên Qdrant."""
    total = len(chunks_list)
    uploaded = 0
    t_start = time.time()

    for batch_start in range(0, total, batch_size):
        batch = chunks_list[batch_start:batch_start + batch_size]
        texts = [c["text"] for c in batch]

        # Embed
        encoded = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dense_vecs = encoded["dense_vecs"]
        sparse_weights = encoded["lexical_weights"]

        # Build Qdrant points
        points = []
        for idx, chunk in enumerate(batch):
            # Sparse vector
            sparse_indices = []
            sparse_values = []
            for token_id, weight in sparse_weights[idx].items():
                try:
                    sparse_indices.append(int(token_id))
                    sparse_values.append(float(weight))
                except:
                    pass

            point = models.PointStruct(
                id=make_uuid(chunk["id"]),
                vector={
                    "": dense_vecs[idx].tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                },
                payload={
                    "type": chunk["type"],
                    "mapc": chunk["mapc"],
                    "ten_dieu": chunk["ten_dieu"],
                    "text": chunk["text"],
                    "chunk_label": chunk.get("chunk_label", ""),
                    "parent_id": chunk.get("parent_id", ""),
                    **chunk["metadata"],
                },
            )
            points.append(point)

        # Upload batch
        qdrant.upsert(collection_name=collection_name, points=points)
        uploaded += len(batch)

        # Progress
        elapsed = time.time() - t_start
        speed = uploaded / elapsed if elapsed > 0 else 0
        eta = (total - uploaded) / speed if speed > 0 else 0
        if uploaded % (batch_size * 10) == 0 or uploaded == total:
            print(f"  [{collection_name}] {uploaded:,}/{total:,} "
                  f"({uploaded/total*100:.1f}%) | "
                  f"{speed:.0f} texts/s | "
                  f"ETA: {eta/60:.1f} phút")

    total_time = time.time() - t_start
    print(f"  {collection_name}: {uploaded:,} points uploaded in {total_time/60:.1f} phút\n")


# ─── Chạy embedding + upload ─────────────────────────────────
print("=" * 60)
print("BẮT ĐẦU EMBEDDING + UPLOAD")
print("=" * 60)

print(f"\nCollection 1: {COLL_CHILD} ({len(children):,} child chunks)")
embed_and_upload(children, COLL_CHILD)

print(f"Collection 2: {COLL_PARENT} ({len(parents):,} parent chunks)")
embed_and_upload(parents, COLL_PARENT)

print("=" * 60)
print("HOÀN THÀNH! Tất cả vectors đã được upload lên Qdrant Cloud.")
print("=" * 60)


# ==============================================================
# CELL 9: Verify — Test search thử
# ==============================================================
print("\nTest search thử...")

# Embed câu query test
test_query = "Quyền hạn của bảo vệ dân phố là gì?"
q_emb = model.encode([test_query], return_dense=True, return_sparse=True, return_colbert_vecs=False)

results = qdrant.query_points(
    collection_name=COLL_CHILD,
    query=q_emb["dense_vecs"][0].tolist(),
    limit=3,
    with_payload=True,
)

print(f"\nQuery: '{test_query}'")
print(f"Top 3 kết quả từ {COLL_CHILD}:\n")
for i, point in enumerate(results.points):
    print(f"  {i+1}. [{point.score:.4f}] {point.payload['ten_dieu']}")
    print(f"     {point.payload['text'][:150]}...")
    print()

# Truy ngược parent
if results.points:
    parent_id = results.points[0].payload.get("parent_id", "")
    if parent_id:
        parent_uuid = make_uuid(parent_id)
        parent_points = qdrant.retrieve(
            collection_name=COLL_PARENT,
            ids=[parent_uuid],
            with_payload=True,
        )
        if parent_points:
            p = parent_points[0]
            print(f"Parent (Điều gốc): {p.payload['ten_dieu']}")
            print(f"   Toàn văn ({len(p.payload['text'].split())} từ):")
            print(f"   {p.payload['text'][:300]}...")

print("\nPipeline hoàn tất! Qdrant Cloud đã sẵn sàng cho Legal CallBot.")
