import logging
import asyncio
from typing import List, Dict
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

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
        self.dense_vector_name = "dense"
        self.sparse_vector_name = "sparse"
        
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
        self._detect_vector_names()

    def _detect_vector_names(self):
        """Tự động phát hiện tên vector của collection (dense/sparse)."""
        try:
            info = self.qdrant.get_collection(self.collection)
            params = info.config.params

            vectors_cfg = getattr(params, "vectors", None)
            if isinstance(vectors_cfg, dict):
                if "dense" in vectors_cfg:
                    self.dense_vector_name = "dense"
                elif len(vectors_cfg) > 0:
                    self.dense_vector_name = next(iter(vectors_cfg.keys()))
            else:
                # Single unnamed dense vector
                self.dense_vector_name = ""

            sparse_cfg = getattr(params, "sparse_vectors", None)
            if isinstance(sparse_cfg, dict):
                if "sparse" in sparse_cfg:
                    self.sparse_vector_name = "sparse"
                elif len(sparse_cfg) > 0:
                    self.sparse_vector_name = next(iter(sparse_cfg.keys()))

            logger.info(
                f"Detected vector names - dense: '{self.dense_vector_name}', sparse: '{self.sparse_vector_name}'"
            )
        except Exception as e:
            logger.warning(
                f"Could not auto-detect vector names for collection '{self.collection}': {e}. "
                "Fallback to dense/sparse defaults."
            )

    def _build_prefetch(self, dense_query, sparse_query, qfilter):
        dense_prefetch = models.Prefetch(query=dense_query, limit=20, filter=qfilter)
        sparse_prefetch = models.Prefetch(query=sparse_query, limit=20, filter=qfilter)

        if self.dense_vector_name is not None:
            dense_prefetch.using = self.dense_vector_name
        if self.sparse_vector_name:
            sparse_prefetch.using = self.sparse_vector_name

        return [dense_prefetch, sparse_prefetch]

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

        try:
            results = await asyncio.to_thread(
                self.qdrant.query_points,
                collection_name=self.collection,
                prefetch=self._build_prefetch(dense_query, sparse_query, qfilter),
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True
            )
        except UnexpectedResponse as e:
            # Fallback cho các snapshot cũ dùng unnamed dense vector (using="")
            err_msg = str(e)
            if "Not existing vector name" in err_msg and "dense" in err_msg:
                logger.warning("Dense vector name mismatch. Retrying with unnamed dense vector ('').")
                self.dense_vector_name = ""
                results = await asyncio.to_thread(
                    self.qdrant.query_points,
                    collection_name=self.collection,
                    prefetch=self._build_prefetch(dense_query, sparse_query, qfilter),
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                    with_payload=True
                )
            else:
                raise

        formatted_results = []
        for p in results.points:
            formatted_results.append({
                "dieu": p.payload.get("ten_dieu", "Không rõ Điều luật"),
                "content": p.payload.get("text", ""),
                "score": p.score
            })
            
        return formatted_results

