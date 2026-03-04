import json
import logging
import time
import os
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.http import models

# Import models dynamically to support isolated setup
import importlib.util

# Adjust logic folder resolution
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.legal_chunker import LegalChunker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ingest")

def has_flags_embedded():
    spec = importlib.util.find_spec('FlagEmbedding')
    return spec is not None
        
def get_dense_embeddings(text: str) -> List[float]:
    """Temporary fallback dummy embedding generator or true embedding retrieval depending on module condition."""
    return [0.0] * 1024

class QdrantIngester:
    def __init__(self, data_path: str, reset: bool = False, use_model: bool = False):
        self.data_path = data_path
        self.qdrant = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=os.getenv("QDRANT_PORT", "6333"))
        
        # BGE-M3 (supports Hybrid searching) - Dense (1024), Sparse Lexical
        self.use_model = use_model
        if self.use_model:
            from FlagEmbedding import BGEM3FlagModel
            logger.info("Loading BGE-M3 model... This might take a while.")
            # Use 'BAAI/bge-m3' - Supports English and Vietnamese extremely well
            # Will be initialized below inside ingest_all() for lazy loading.
            self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        else:
            self.model = None

        self.collection_parent = "phap_dien_dieu"
        self.collection_child = "phap_dien_khoan"
        
        if reset:
            self._create_collections()
            
    def _create_collections(self):
        """Create 2 collections. Parent holds full context text. Child holds exact semantic structure matching Points and Clauses"""
        for coll in [self.collection_parent, self.collection_child]:
             if self.qdrant.collection_exists(coll):
                 self.qdrant.delete_collection(coll)
                 logger.info(f"Deleted old collection {coll}")
                 
             self.qdrant.create_collection(
                 collection_name=coll,
                 # Vector configuration follows BGE-M3 specifications
                 vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
                 # Optional setup sparse support could be added directly via sparse_vectors config.
                 sparse_vectors_config={"sparse": models.SparseVectorParams()}
             )
             logger.info(f"Created collection {coll}")

             # Create metadata index for faster filter search
             self.qdrant.create_payload_index(coll, "chude", field_schema="keyword")
             self.qdrant.create_payload_index(coll, "chuong", field_schema="keyword")

    def _encode_batch(self, texts: List[str]):
        """Encode batch using BGE-M3 producing both dense and sparse representations"""
        if not self.use_model:
             # Create dummy records when script acts as dry-run. Dense 1024D Sparse
             return {
                 "dense_vecs": [[0.01] * 1024 for _ in texts],
                 "lexical_weights": [{"dummy": 0.1} for _ in texts]
             }
        
        # Note: BGEM3 returns a dictionary with 'dense_vecs', 'lexical_weights', and 'colbert_vecs'
        embeddings = self.model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        return embeddings

    def ingest_all(self, batch_size: int = 200, max_items: int = None):
         logger.info(f"Loading data from {self.data_path}")
         with open(self.data_path, "r", encoding="utf-8") as f:
             raw_data = json.load(f)
             
         chunker = LegalChunker(use_contextual_enrichment=True)
         
         total_records = len(raw_data)
         target_items = min(max_items, total_records) if max_items else total_records
         logger.info(f"Starting ingestion process. Total raw distinct articles processing: {target_items}")

         parent_batch = []
         child_batch = []

         processed_source = 0
         
         for record in raw_data:
             if processed_source >= target_items:
                 break

             noidung = record.get("noidung", "")
             # Empty text rules
             if not noidung or len(noidung.strip()) == 0:
                 continue
                 
             mapc = record.get("mapc", "")
             ten_dieu = record.get("ten", "")
             
             # Group metadata explicitly
             metadata = {
                 "chude": record.get("chude", "Unknown"),
                 "demuc": record.get("demuc", "Unknown"),
                 "chuong": record.get("chuong", "Unknown"),
                 "vbqppl": record.get("vbqppl", "Unknown"),
                 "stt": record.get("stt", 0)
             }
             
             chunks = chunker.extract_chunks(mapc, ten_dieu, noidung, metadata)
             
             for chunk in chunks:
                   if chunk["type"] == "parent":
                       parent_batch.append(chunk)
                   else:
                       child_batch.append(chunk)
             
             processed_source += 1

             # Trigger batch upsert logic when hitting boundaries
             if len(child_batch) >= batch_size:
                  self._process_and_upload(child_batch, self.collection_child)
                  child_batch.clear()

             if len(parent_batch) >= batch_size:
                  self._process_and_upload(parent_batch, self.collection_parent)
                  parent_batch.clear()
                  logger.info(f"Progress: Processed {processed_source}/{target_items} items")

         # Process remaining buffered items
         if child_batch:
              self._process_and_upload(child_batch, self.collection_child)
         if parent_batch:
              self._process_and_upload(parent_batch, self.collection_parent)

         logger.info("Ingestion completed successfully.")
         
    def _process_and_upload(self, chunks: List[Dict], collection_name: str):
         """Helper processing embeddings on specific payload groups and uploading points to Qdrant."""
         texts = [c["text"] for c in chunks]
         encoded = self._encode_batch(texts)
         
         dense_vecs = encoded["dense_vecs"]
         sparse_vecs = encoded["lexical_weights"]
         
         points = []
         for idx, chunk in enumerate(chunks):
              
              # Map dictionary `{word: weight}` returned by BGEM3 model to Qdrant SparseVector representation
              sparse_dict_raw = sparse_vecs[idx]
              
              sparse_indices = []
              sparse_values = []
              
              if self.use_model:
                  # When using true BGEM3 model, lexical weights are given through vocab representation
                  # Note: we need integer representation for indices (depends on transformer tokenizer)
                  # Assuming BGE-M3 tokenizer mappings string back to ID:
                  for token_id_str, weight in sparse_dict_raw.items():
                       try:
                           sparse_indices.append(int(token_id_str))
                           sparse_values.append(weight)
                       except:
                           pass
              else:
                  sparse_indices = [0]
                  sparse_values = [0.1]
              
              point = models.PointStruct(
                   # Use generated predictable chunk id, converting to UUID format using internal helper or directly passing string Id when supported.
                   # Qdrant supports UUID or integer ID.
                   id=chunk["id"], # <-- NOTE: Ensure this matches Qdrant id constraints (UUID). Let's hash it instead.
                   vector={
                       # By default, anonymous vector (usually "") handles dense points. Named vector handles sparse.
                       "": dense_vecs[idx],
                       "sparse": models.SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                       )
                   },
                   payload={
                        "type": chunk["type"],
                        "mapc": chunk["mapc"],
                        "ten_dieu": chunk["ten_dieu"],
                        "text": chunk["text"],
                        "parent_id": chunk.get("parent_id", ""), # Missing explicitly for type=parent 
                        **chunk["metadata"] 
                   }
              )
              
              # Hash ID appropriately to fit Qdrant UUID
              import hashlib
              import uuid
              hash_object = hashlib.md5(chunk["id"].encode())
              UUID_generated = str(uuid.UUID(hash_object.hexdigest()))
              point.id = UUID_generated
              
              points.append(point)
              
         self.qdrant.upsert(
              collection_name=collection_name, 
              points=points
         )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest law data JSON into Qdrant using BGE-M3 vector bindings")
    parser.add_argument("--data", type=str, default="/Users/nguyenthithutam/Desktop/Callbot/legal-callbot/brain/data/law_data.json", help="Path to json export")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate collections")
    parser.add_argument("--use-model", action="store_true", help="Download and initialize BGEM3 - otherwise runs as dry run")
    parser.add_argument("--max-items", type=int, default=None, help="Limit numbers parsed (for testing)")
    
    args = parser.parse_args()
    
    ingester = QdrantIngester(
        data_path=args.data,
        reset=args.reset,
        use_model=args.use_model
    )
    
    ingester.ingest_all(batch_size=100, max_items=args.max_items)
