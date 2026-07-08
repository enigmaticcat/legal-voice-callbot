from __future__ import annotations

import logging
import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import httpx

from .semantic_signature import semantic_signature, signature_key

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
        semantic_cache_enabled: bool = True,
        semantic_cache_collection: str = "nutrition_retrieval_cache",
        semantic_cache_threshold: float = 0.90,
        semantic_cache_ttl_seconds: int = 604800,
    ):
        self.collection = collection
        self.qdrant_path = qdrant_path
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.llm_client = llm_client
        self.retrieval_cache = retrieval_cache
        self.semantic_cache_enabled = semantic_cache_enabled
        self.semantic_cache_collection = semantic_cache_collection
        self.semantic_cache_threshold = semantic_cache_threshold
        self.semantic_cache_ttl_seconds = semantic_cache_ttl_seconds
        self.semantic_cache_ready = False

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
        if self.semantic_cache_enabled:
            self._ensure_semantic_cache_collection()
        logger.info("RAGPipeline ready.")

    def _ensure_semantic_cache_collection(self) -> None:
        from qdrant_client import models

        try:
            self.qdrant.get_collection(self.semantic_cache_collection)
            self.semantic_cache_ready = True
            return
        except Exception:
            pass

        vector_size = self.model.get_sentence_embedding_dimension()
        try:
            self.qdrant.create_collection(
                collection_name=self.semantic_cache_collection,
                vectors_config=models.VectorParams(
                    size=int(vector_size),
                    distance=models.Distance.COSINE,
                ),
            )
            self.semantic_cache_ready = True
            logger.info(
                "Created semantic cache collection: %s",
                self.semantic_cache_collection,
            )
        except Exception:
            logger.warning(
                "Semantic cache collection unavailable; semantic cache disabled",
                exc_info=True,
            )
            self.semantic_cache_ready = False

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

        semantic_vector = None
        semantic_meta = {"status": "bypass"}
        semantic_signature_value = semantic_signature(query)
        if (
            self.semantic_cache_enabled
            and self.semantic_cache_ready
            and not filters
            and not use_hyde
        ):
            semantic_vector = await self._encode_query(query)
            if semantic_vector is None:
                return [], {
                    **cache_meta,
                    "semantic_status": "embedding_timeout",
                }
            semantic_documents, semantic_meta = await self._semantic_lookup(
                vector=semantic_vector,
                signature=semantic_signature_value,
                fetch_k=fetch_k,
                top_k=top_k,
            )
            if semantic_documents is not None:
                saved_ms = float(semantic_meta.get("estimated_saved_ms", 0))
                if self.retrieval_cache is not None:
                    await self.retrieval_cache.record_semantic("hits", saved_ms)
                    if cache_key is not None:
                        await self.retrieval_cache.set(
                            cache_key,
                            semantic_documents,
                            saved_ms,
                        )
                return semantic_documents, {
                    "status": "semantic_hit",
                    **semantic_meta,
                }
            if self.retrieval_cache is not None:
                await self.retrieval_cache.record_semantic("misses")

        started = time.perf_counter()
        documents = await self._search_uncached(
            query=query,
            filters=filters,
            top_k=top_k,
            fetch_k=fetch_k,
            use_hyde=use_hyde,
            query_vector=semantic_vector,
        )
        compute_ms = (time.perf_counter() - started) * 1000

        if cache_key is not None and documents:
            await self.retrieval_cache.set(cache_key, documents, compute_ms)
            cache_meta["compute_ms"] = round(compute_ms, 1)

        if semantic_vector is not None and documents:
            await self._semantic_store(
                vector=semantic_vector,
                query=query,
                signature=semantic_signature_value,
                documents=documents,
                fetch_k=fetch_k,
                top_k=top_k,
                compute_ms=compute_ms,
            )
            if self.retrieval_cache is not None:
                await self.retrieval_cache.record_semantic("writes")

        cache_meta["semantic_status"] = semantic_meta.get("status", "bypass")
        cache_meta["semantic_signature"] = signature_key(semantic_signature_value)
        return documents, cache_meta

    async def _search_uncached(self, query: str, filters: dict = None,
                               top_k: int = 5, fetch_k: int = 8,
                               use_hyde: bool = False,
                               query_vector: list[float] | None = None) -> List[Dict]:
        logger.debug(f"RAG search: {query[:80]}...")

        embed_text = query
        if use_hyde and self.llm_client is not None:
            t_hyde = time.time()
            embed_text = await self._generate_hyde_doc(query)
            logger.info("HyDE: %.0fms | doc[:80]=%s", (time.time() - t_hyde) * 1000, embed_text[:80])

        q_vec = query_vector
        if q_vec is None:
            q_vec = await self._encode_query(embed_text)
            if q_vec is None:
                return []

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

    async def _encode_query(self, text: str) -> list[float] | None:
        try:
            async with asyncio.timeout(30.0):
                async with _get_semaphore():
                    vector = await asyncio.to_thread(
                        self.model.encode,
                        [text],
                        normalize_embeddings=True,
                    )
            return vector[0].tolist()
        except asyncio.TimeoutError:
            logger.error("RAG embedding timeout after 30s")
            return None

    async def _semantic_lookup(
        self,
        vector: list[float],
        signature: dict[str, str],
        fetch_k: int,
        top_k: int,
    ) -> tuple[list[dict] | None, dict]:
        from qdrant_client import models

        signature_value = signature_key(signature)
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="signature",
                    match=models.MatchValue(value=signature_value),
                ),
                models.FieldCondition(
                    key="corpus_version",
                    match=models.MatchValue(
                        value=getattr(self.retrieval_cache, "corpus_version", "v1")
                    ),
                ),
                models.FieldCondition(
                    key="embedding_model",
                    match=models.MatchValue(value=MODEL_NAME),
                ),
                models.FieldCondition(
                    key="reranker_model",
                    match=models.MatchValue(value=RERANKER_MODEL),
                ),
                models.FieldCondition(
                    key="fetch_k",
                    match=models.MatchValue(value=fetch_k),
                ),
                models.FieldCondition(
                    key="top_k",
                    match=models.MatchValue(value=top_k),
                ),
                models.FieldCondition(
                    key="expires_at",
                    range=models.Range(gte=int(time.time())),
                ),
            ]
        )
        try:
            result = await asyncio.to_thread(
                self.qdrant.query_points,
                collection_name=self.semantic_cache_collection,
                query=vector,
                limit=1,
                score_threshold=self.semantic_cache_threshold,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception:
            logger.warning("Semantic cache lookup failed", exc_info=True)
            return None, {"status": "error", "signature": signature_value}

        if not result.points:
            return None, {
                "status": "miss",
                "signature": signature_value,
                "threshold": self.semantic_cache_threshold,
            }

        point = result.points[0]
        payload = point.payload or {}
        documents = payload.get("documents") or []
        if not documents:
            return None, {
                "status": "miss",
                "signature": signature_value,
                "threshold": self.semantic_cache_threshold,
            }

        logger.info(
            "Semantic retrieval cache HIT score=%.4f signature=%s query=%s",
            point.score,
            signature_value,
            str(payload.get("query", ""))[:80],
        )
        return documents, {
            "semantic_status": "hit",
            "similarity": round(float(point.score), 4),
            "threshold": self.semantic_cache_threshold,
            "matched_query": payload.get("query", ""),
            "signature": signature_value,
            "estimated_saved_ms": float(payload.get("compute_ms", 0)),
        }

    async def _semantic_store(
        self,
        vector: list[float],
        query: str,
        signature: dict[str, str],
        documents: list[dict],
        fetch_k: int,
        top_k: int,
        compute_ms: float,
    ) -> None:
        from qdrant_client import models

        signature_value = signature_key(signature)
        corpus_version = getattr(self.retrieval_cache, "corpus_version", "v1")
        point_material = "|".join([
            query.lower().strip(),
            signature_value,
            corpus_version,
            MODEL_NAME,
            RERANKER_MODEL,
            str(fetch_k),
            str(top_k),
        ])
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, point_material))
        now = int(time.time())
        payload = {
            "query": query,
            "signature": signature_value,
            "audience": signature["audience"],
            "condition": signature["condition"],
            "intent": signature["intent"],
            "documents": documents,
            "compute_ms": round(compute_ms, 1),
            "corpus_version": corpus_version,
            "embedding_model": MODEL_NAME,
            "reranker_model": RERANKER_MODEL,
            "fetch_k": fetch_k,
            "top_k": top_k,
            "created_at": now,
            "expires_at": now + self.semantic_cache_ttl_seconds,
        }
        try:
            await asyncio.to_thread(
                self.qdrant.upsert,
                collection_name=self.semantic_cache_collection,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
                wait=True,
            )
            logger.info(
                "Semantic retrieval cache WRITE signature=%s ttl=%ds",
                signature_value,
                self.semantic_cache_ttl_seconds,
            )
        except Exception:
            logger.warning("Semantic cache write failed", exc_info=True)

    async def semantic_cache_stats(self) -> dict:
        if not self.semantic_cache_ready:
            return {"enabled": self.semantic_cache_enabled, "ready": False}
        try:
            info = await asyncio.to_thread(
                self.qdrant.get_collection,
                self.semantic_cache_collection,
            )
            return {
                "enabled": self.semantic_cache_enabled,
                "ready": True,
                "collection": self.semantic_cache_collection,
                "points_count": int(getattr(info, "points_count", 0) or 0),
                "threshold": self.semantic_cache_threshold,
                "ttl_seconds": self.semantic_cache_ttl_seconds,
            }
        except Exception:
            return {"enabled": self.semantic_cache_enabled, "ready": False}

    async def clear_semantic_cache(self) -> bool:
        if not self.semantic_cache_ready:
            return False
        try:
            await asyncio.to_thread(
                self.qdrant.delete_collection,
                self.semantic_cache_collection,
            )
            self.semantic_cache_ready = False
            self._ensure_semantic_cache_collection()
            return self.semantic_cache_ready
        except Exception:
            logger.warning("Semantic cache clear failed", exc_info=True)
            return False

    # ── User document upload ──────────────────────────────────────────

    USER_DOCS_COLLECTION = "user_documents"
    _user_docs_collection_exists: bool = False

    def _ensure_user_docs_collection(self) -> None:
        from qdrant_client import models as qm
        try:
            self.qdrant.get_collection(self.USER_DOCS_COLLECTION)
            self._user_docs_collection_exists = True
        except Exception:
            vector_size = self.model.get_sentence_embedding_dimension()
            self.qdrant.create_collection(
                collection_name=self.USER_DOCS_COLLECTION,
                vectors_config=qm.VectorParams(
                    size=int(vector_size),
                    distance=qm.Distance.COSINE,
                ),
            )
            self._user_docs_collection_exists = True
            logger.info("Created user_documents collection")

    async def upsert_user_docs(
        self,
        session_id: str,
        filename: str,
        chunks: list[str],
    ) -> int:
        from qdrant_client import models as qm

        await asyncio.to_thread(self._ensure_user_docs_collection)

        vectors = await asyncio.to_thread(
            self.model.encode,
            chunks,
            normalize_embeddings=True,
        )

        points = [
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i].tolist(),
                payload={
                    "content": chunk,
                    "session_id": session_id,
                    "filename": filename,
                    "source": "user_upload",
                    "title": filename,
                },
            )
            for i, chunk in enumerate(chunks)
        ]

        await asyncio.to_thread(
            self.qdrant.upsert,
            collection_name=self.USER_DOCS_COLLECTION,
            points=points,
            wait=True,
        )
        logger.info("Upserted %d chunks for session=%s file=%s", len(chunks), session_id, filename)
        return len(chunks)

    async def search_user_docs(
        self,
        session_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        from qdrant_client import models as qm

        if not self._user_docs_collection_exists:
            return []

        q_vec = await self._encode_query(query)
        if q_vec is None:
            return []

        qfilter = qm.Filter(
            must=[qm.FieldCondition(
                key="session_id",
                match=qm.MatchValue(value=session_id),
            )]
        )

        try:
            results = await asyncio.to_thread(
                self.qdrant.query_points,
                collection_name=self.USER_DOCS_COLLECTION,
                query=q_vec,
                limit=top_k * 2,
                query_filter=qfilter,
                with_payload=True,
            )
        except Exception:
            logger.warning("User docs search failed", exc_info=True)
            return []

        docs = [
            {
                "title": p.payload.get("filename", "Tài liệu của bạn"),
                "source": "user_upload",
                "content": p.payload.get("content", ""),
                "score": p.score,
            }
            for p in results.points
        ]

        if not docs:
            return []

        pairs = [[query, d["content"]] for d in docs]
        try:
            async with asyncio.timeout(15.0):
                rerank_scores = await asyncio.to_thread(self.reranker.predict, pairs)
            rerank_scores = [float(s) if s == s else -999.0 for s in rerank_scores]
            docs = [d for _, d in sorted(zip(rerank_scores, docs), key=lambda x: -x[0])[:top_k]]
        except Exception:
            docs = docs[:top_k]

        return docs
