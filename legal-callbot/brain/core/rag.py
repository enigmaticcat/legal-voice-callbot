import logging
import asyncio
from typing import List, Dict
from qdrant_client import QdrantClient, models

import transformers.utils.import_utils
if not hasattr(transformers.utils.import_utils, 'is_torch_fx_available'):
    transformers.utils.import_utils.is_torch_fx_available = lambda: False
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger("brain.core.rag")


class RAGPipeline:

    def __init__(self, qdrant_url: str, collection: str, qdrant_api_key: str = None, qdrant_path: str = None):
        self.collection = collection
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.qdrant_path = qdrant_path
        
        if qdrant_path:
            logger.info(f"Initializing Local QdrantClient at: {qdrant_path}")
            self.qdrant = QdrantClient(path=qdrant_path)
        else:
            logger.info(f"Initializing Cloud QdrantClient: {qdrant_url[:30]}...")
            if qdrant_api_key:
                self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                self.qdrant = QdrantClient(url=qdrant_url)
        
        logger.info("Initializing BGE-M3 Model for queries...")
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)

    async def search(self, expanded_query: str, filters: dict = None, top_k: int = 10) -> List[Dict]:
        logger.debug(f"RAG search for: {expanded_query[:50]}...")
        
        q_emb = self.model.encode([expanded_query], return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dense_query = q_emb["dense_vecs"][0].tolist()
        
        si, sv = [], []
        for tid, w in q_emb["lexical_weights"][0].items():
            try:
                si.append(int(tid))
                sv.append(float(w))
            except: pass
        sparse_query = models.SparseVector(indices=si, values=sv)

        # Xây dựng Pre-Filter Object chặn Vector Search
        qfilter = None
        if filters:
            must_conds = []
            for k, v in filters.items():
                must_conds.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
            qfilter = models.Filter(must=must_conds)

        results = await asyncio.to_thread(
            self.qdrant.query_points,
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(query=dense_query, using="dense", limit=20, filter=qfilter),
                models.Prefetch(query=sparse_query, using="sparse", limit=20, filter=qfilter),
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

