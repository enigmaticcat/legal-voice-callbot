from __future__ import annotations

import logging
import asyncio
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import httpx

logger = logging.getLogger("brain.core.rag")

# Giới hạn concurrent GPU inference (embed + rerank).
# Mỗi request encode 1 query + rerank ~15 pairs — trên A100 có thể tăng lên 16-20.
_RAG_MAX_CONCURRENT = int(os.getenv("RAG_MAX_CONCURRENT", "10"))
_rag_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _rag_semaphore
    if _rag_semaphore is None:
        _rag_semaphore = asyncio.Semaphore(_RAG_MAX_CONCURRENT)
    return _rag_semaphore

try:
    from brain.config import config as _cfg
except ImportError:
    from config import config as _cfg
MODEL_NAME = _cfg.embedding_model
RERANKER_MODEL = _cfg.reranker_model


_HYDE_PROMPT = (
    "Bạn là chuyên gia dinh dưỡng. Viết một đoạn văn ngắn (~80 từ) "
    "trả lời câu hỏi sau đây theo phong cách tài liệu y tế, "
    "chỉ dùng thông tin dinh dưỡng phổ biến, không bịa đặt:\n\n{query}\n\nĐoạn văn:"
)


class RAGPipeline:

    def __init__(
        self,
        qdrant_url: str,
        collection: str,
        qdrant_api_key: str = None,
        qdrant_path: str = None,
        qdrant_snapshot_path: str = None,
        qdrant_snapshot_force_restore: bool = False,
        qdrant_snapshot_timeout_s: int = 600,
        qdrant_snapshot_priority: str = "snapshot",
        llm_client=None,
        retrieval_cache=None,
    ):
        self.collection = collection
        self.qdrant_path = qdrant_path
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.llm_client = llm_client
        self.retrieval_cache = retrieval_cache

        if qdrant_path:
            logger.info(f"Initializing Local QdrantClient at: {qdrant_path}")
            self.qdrant = QdrantClient(path=qdrant_path)
        else:
            logger.info(f"Initializing Cloud QdrantClient: {qdrant_url[:30]}...")
            if qdrant_api_key:
                self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                self.qdrant = QdrantClient(url=qdrant_url)

        if qdrant_snapshot_path:
            self._restore_snapshot(
                snapshot_path=qdrant_snapshot_path,
                force_restore=qdrant_snapshot_force_restore,
                timeout_s=qdrant_snapshot_timeout_s,
                priority=qdrant_snapshot_priority,
            )

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        logger.info(f"Loading embedding model: {MODEL_NAME} (device={device})")
        self.model = SentenceTransformer(
            MODEL_NAME, device=device,
            model_kwargs=model_kwargs,
        )
        logger.info(f"Loading reranker model: {RERANKER_MODEL} (device={device})")
        self.reranker = CrossEncoder(RERANKER_MODEL, device=device,
                                     model_kwargs=model_kwargs)
        logger.info("RAGPipeline ready.")

    def _collection_has_data(self) -> bool:
        try:
            info = self.qdrant.get_collection(self.collection)
            points_count = getattr(info, "points_count", None)
            return bool(points_count and points_count > 0)
        except Exception:
            return False

    def _restore_snapshot(
        self,
        snapshot_path: str,
        force_restore: bool,
        timeout_s: int,
        priority: str,
    ):
        if not snapshot_path or not os.path.exists(snapshot_path):
            logger.warning("Qdrant snapshot path not found: %s", snapshot_path)
            return

        if not force_restore and self._collection_has_data():
            logger.info(
                "Qdrant collection '%s' already has data. Skip snapshot restore.",
                self.collection,
            )
            return

        if self.qdrant_path:
            self._restore_snapshot_local(snapshot_path)
        else:
            self._restore_snapshot_http(snapshot_path, timeout_s, priority)

    def _restore_snapshot_local(self, snapshot_path: str):
        """Restore snapshot into local embedded Qdrant via recover_snapshot()."""
        snapshot_uri = Path(snapshot_path).resolve().as_uri()
        logger.info("Restoring snapshot to local Qdrant: %s", snapshot_uri)
        try:
            self.qdrant.recover_snapshot(
                collection_name=self.collection,
                location=snapshot_uri,
            )
            logger.info("Local snapshot restore complete: collection=%s", self.collection)
        except Exception as e:
            raise RuntimeError(f"Local snapshot restore failed: {e}") from e

    def _restore_snapshot_http(self, snapshot_path: str, timeout_s: int, priority: str):
        """Upload snapshot to a running Qdrant HTTP server."""
        if not self.qdrant_url:
            raise RuntimeError("QDRANT_URL must be set for HTTP snapshot restore")

        base_url = self.qdrant_url.rstrip("/")
        headers = {"api-key": self.qdrant_api_key} if self.qdrant_api_key else {}

        logger.info("Uploading snapshot to Qdrant HTTP server: %s", base_url)
        with httpx.Client(timeout=timeout_s) as client:
            with open(snapshot_path, "rb") as f:
                resp = client.post(
                    f"{base_url}/collections/{self.collection}/snapshots/upload",
                    params={"priority": priority},
                    files={"snapshot": f},
                    headers=headers,
                )
                resp.raise_for_status()

            deadline = time.time() + timeout_s
            while time.time() < deadline:
                r = client.get(f"{base_url}/collections/{self.collection}", headers=headers)
                if r.status_code == 200:
                    data = r.json().get("result", {})
                    if data.get("status") == "green":
                        logger.info(
                            "HTTP snapshot restored. collection=%s points=%s",
                            self.collection,
                            data.get("points_count", 0),
                        )
                        return
                time.sleep(2)

        raise RuntimeError(
            f"Qdrant HTTP snapshot restore timed out after {timeout_s}s"
        )

    async def _generate_hyde_doc(self, query: str) -> str:
        """Sinh đoạn văn giả định (HyDE) từ câu hỏi để cải thiện embedding search."""
        if self.llm_client is None:
            return query
        try:
            prompt = _HYDE_PROMPT.format(query=query)
            hyde_doc = await asyncio.wait_for(
                self.llm_client.generate(prompt, temperature=0.3),
                timeout=8.0,
            )
            hyde_doc = hyde_doc.strip()
            if not hyde_doc:
                return query
            logger.debug("HyDE doc: %s", hyde_doc[:100])
            return hyde_doc
        except Exception:
            logger.warning("HyDE generation failed, falling back to original query", exc_info=True)
            return query

    async def search(self, query: str, filters: dict = None,
                     top_k: int = 5, fetch_k: int = 8,
                     use_hyde: bool = False) -> List[Dict]:
        documents, _ = await self.search_with_meta(
            query=query,
            filters=filters,
            top_k=top_k,
            fetch_k=fetch_k,
            use_hyde=use_hyde,
        )
        return documents

    async def search_with_meta(
        self,
        query: str,
        filters: dict = None,
        top_k: int = 5,
        fetch_k: int = 8,
        use_hyde: bool = False,
    ) -> tuple[List[Dict], dict]:
        cache_key = None
        if self.retrieval_cache is not None and not filters:
            cache_key = self.retrieval_cache.build_key(
                query=query,
                embedding_model=MODEL_NAME,
                reranker_model=RERANKER_MODEL,
                fetch_k=fetch_k,
                top_k=top_k,
                use_hyde=use_hyde,
            )
            cached_documents, cache_meta = await self.retrieval_cache.get(cache_key)
            if cached_documents is not None:
                return cached_documents, cache_meta
        else:
            cache_meta = {"status": "bypass", "reason": "filters" if filters else "not_configured"}

        started = time.perf_counter()
        documents = await self._search_uncached(
            query=query,
            filters=filters,
            top_k=top_k,
            fetch_k=fetch_k,
            use_hyde=use_hyde,
        )
        compute_ms = (time.perf_counter() - started) * 1000

        if cache_key is not None:
            await self.retrieval_cache.set(cache_key, documents, compute_ms)
            cache_meta["compute_ms"] = round(compute_ms, 1)

        return documents, cache_meta

    async def _search_uncached(self, query: str, filters: dict = None,
                               top_k: int = 5, fetch_k: int = 8,
                               use_hyde: bool = False) -> List[Dict]:
        logger.debug(f"RAG search: {query[:80]}...")

        embed_text = query
        if use_hyde and self.llm_client is not None:
            t_hyde = time.time()
            embed_text = await self._generate_hyde_doc(query)
            logger.info("HyDE: %.0fms | doc[:80]=%s", (time.time() - t_hyde) * 1000, embed_text[:80])

        try:
            async with asyncio.timeout(30.0):
                async with _get_semaphore():
                    q_vec = await asyncio.to_thread(
                        self.model.encode,
                        [embed_text],
                        normalize_embeddings=True,
                    )
        except asyncio.TimeoutError:
            logger.error("RAG embedding timeout after 30s")
            return []
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
            limit=fetch_k,
            query_filter=qfilter,
            with_payload=True,
        )

        docs = [
            {
                "title": p.payload.get("title", ""),
                "source": p.payload.get("source", ""),
                "url": p.payload.get("url", ""),
                "content": p.payload.get("text", ""),
                "score": p.score,
            }
            for p in results.points
        ]

        if not docs:
            return docs

        # Rerank with cross-encoder
        pairs = [[query, d["content"]] for d in docs]
        try:
            async with asyncio.timeout(30.0):
                async with _get_semaphore():
                    rerank_scores = await asyncio.to_thread(self.reranker.predict, pairs)
        except asyncio.TimeoutError:
            logger.error("RAG reranking timeout after 30s, returning top-%d by embedding score", top_k)
            return docs[:top_k]
        rerank_scores = [float(s) if not (s != s) else -999.0 for s in rerank_scores]  # handle NaN
        docs_scored = sorted(zip(rerank_scores, docs), key=lambda x: -x[0])
        reranked = [d for _, d in docs_scored[:top_k]]
        logger.debug(f"Reranked {fetch_k} → {top_k} docs")
        return reranked
