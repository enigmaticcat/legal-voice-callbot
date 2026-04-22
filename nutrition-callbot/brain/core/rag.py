import logging
import asyncio
import os
import time
from typing import List, Dict
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import httpx

logger = logging.getLogger("brain.core.rag")

QUERY_PREFIX = "Instruct: Tìm thông tin dinh dưỡng liên quan\nQuery: "
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"


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
    ):
        self.collection = collection
        self.qdrant_path = qdrant_path
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key

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
            self._restore_snapshot_if_needed(
                snapshot_path=qdrant_snapshot_path,
                force_restore=qdrant_snapshot_force_restore,
                timeout_s=qdrant_snapshot_timeout_s,
                priority=qdrant_snapshot_priority,
            )

        logger.info(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        logger.info("RAGPipeline ready.")

    def _restore_snapshot_if_needed(
        self,
        snapshot_path: str,
        force_restore: bool,
        timeout_s: int,
        priority: str,
    ):
        if self.qdrant_path:
            logger.warning(
                "QDRANT_SNAPSHOT_PATH is set but QDRANT_PATH local-embedded mode is active. "
                "Snapshot auto-restore supports HTTP Qdrant server mode only; skipping restore."
            )
            return

        if not snapshot_path or not os.path.exists(snapshot_path):
            logger.warning("Qdrant snapshot path not found: %s", snapshot_path)
            return

        if not self.qdrant_url:
            logger.warning("Qdrant snapshot restore skipped: qdrant_url is empty")
            return

        base_url = self.qdrant_url.rstrip("/")
        headers = {}
        if self.qdrant_api_key:
            headers["api-key"] = self.qdrant_api_key

        try:
            if not force_restore:
                info = self.qdrant.get_collection(self.collection)
                points_count = getattr(info, "points_count", None)
                status = getattr(getattr(info, "status", None), "value", getattr(info, "status", None))
                if points_count and points_count > 0:
                    logger.info(
                        "Qdrant collection '%s' already has %s points (status=%s). Skip snapshot restore.",
                        self.collection,
                        points_count,
                        status,
                    )
                    return
        except Exception:
            # Collection missing or inaccessible; continue restore flow.
            pass

        logger.info("Restoring Qdrant snapshot from: %s", snapshot_path)
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
                    status = data.get("status")
                    points_count = data.get("points_count", 0)
                    if status == "green":
                        logger.info(
                            "Qdrant snapshot restored. collection=%s status=%s points=%s",
                            self.collection,
                            status,
                            points_count,
                        )
                        return
                time.sleep(2)

        raise RuntimeError(
            f"Qdrant snapshot restore timed out after {timeout_s}s for collection '{self.collection}'"
        )

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
