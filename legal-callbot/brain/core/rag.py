"""
RAG Pipeline — Hybrid Search + Re-ranking
Tìm Điều luật liên quan bằng vector search + lexical search.
Sử dụng BGE-M3 và Qdrant Cloud.
"""
import logging
import asyncio
from typing import List, Dict
from qdrant_client import QdrantClient, models

# Patch FlagEmbedding import error
import transformers.utils.import_utils
if not hasattr(transformers.utils.import_utils, 'is_torch_fx_available'):
    transformers.utils.import_utils.is_torch_fx_available = lambda: False
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger("brain.core.rag")


class RAGPipeline:
    """
    Advanced RAG cho pháp luật.

    Pipeline:
      1. Khởi tạo Qdrant Client (Cloud) và BGE-M3 Model
      2. Hybrid Search (Dense + Sparse vectors) query expansion
    """

    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection: str):
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.collection = collection
        
        logger.info(f"Initializing QdrantClient: {qdrant_url[:30]}...")
        if qdrant_api_key:
            self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            self.qdrant = QdrantClient(url=qdrant_url)
        
        logger.info("Initializing BGE-M3 Model for queries...")
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)

    async def search(self, expanded_query: str, top_k: int = 10) -> List[Dict]:
        """
        Tìm top_k Điều luật liên quan nhất thông qua Qdrant Hybrid Search (RRF).
        
        Returns:
            List of {"dieu": str, "content": str, "score": float}
        """
        logger.debug(f"RAG search for: {expanded_query[:50]}...")
        
        # 1. Nhúng câu hỏi (tạo Sparse và Dense vectors)
        q_emb = self.model.encode([expanded_query], return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dense_query = q_emb["dense_vecs"][0].tolist()
        
        # Chuẩn bị SparseVector format
        si, sv = [], []
        for tid, w in q_emb["lexical_weights"][0].items():
            try:
                si.append(int(tid))
                sv.append(float(w))
            except: pass
        sparse_query = models.SparseVector(indices=si, values=sv)

        # 2. Hybrid Search (RRF)
        # Vì Qdrant Client `query_points` là đồng bộ (trừ khi dùng AsyncQdrantClient), 
        # nên chạy qua asyncio.to_thread để không chặn event loop.
        results = await asyncio.to_thread(
            self.qdrant.query_points,
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(query=dense_query, using="", limit=20),
                models.Prefetch(query=sparse_query, using="sparse", limit=20),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True
        )

        formatted_results = []
        for p in results.points:
            formatted_results.append({
                "dieu": p.payload.get("ten_dieu", "Không rõ Điều luật"),
                "content": p.payload.get("text", ""),
                "score": p.score
            })
            
        return formatted_results

