import logging
import asyncio
from typing import List, Dict
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("brain.core.rag")

QUERY_PREFIX = "Instruct: Tìm thông tin dinh dưỡng liên quan\nQuery: "
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"


class RAGPipeline:

    def __init__(self, qdrant_url: str, collection: str, qdrant_api_key: str = None, qdrant_path: str = None):
        self.collection = collection

        if qdrant_path:
            logger.info(f"Initializing Local QdrantClient at: {qdrant_path}")
            self.qdrant = QdrantClient(path=qdrant_path)
        else:
            logger.info(f"Initializing Cloud QdrantClient: {qdrant_url[:30]}...")
            if qdrant_api_key:
                self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                self.qdrant = QdrantClient(url=qdrant_url)

        logger.info(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        logger.info("RAGPipeline ready.")

    async def search(self, query: str, filters: dict = None, top_k: int = 5) -> List[Dict]:
        logger.debug(f"RAG search: {query[:80]}...")

        q_vec = await asyncio.to_thread(
            self.model.encode,
            [QUERY_PREFIX + query],
            normalize_embeddings=True,
        )
        q_vec = q_vec[0].tolist()

        qfilter = None
        if filters:
            from qdrant_client import models
            must_conds = [
                models.FieldCondition(key=k, match=models.MatchValue(value=v))
                for k, v in filters.items()
            ]
            qfilter = models.Filter(must=must_conds)

        results = await asyncio.to_thread(
            self.qdrant.query_points,
            collection_name=self.collection,
            query=q_vec,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
        )

        return [
            {
                "title": p.payload.get("title", ""),
                "source": p.payload.get("source", ""),
                "url": p.payload.get("url", ""),
                "content": p.payload.get("text", ""),
                "score": p.score,
            }
            for p in results.points
        ]
