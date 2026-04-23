import logging
import asyncio
import os
import time
from pathlib import Path
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
            self._restore_snapshot(
                snapshot_path=qdrant_snapshot_path,
                force_restore=qdrant_snapshot_force_restore,
                timeout_s=qdrant_snapshot_timeout_s,
                priority=qdrant_snapshot_priority,
            )

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model: {MODEL_NAME} (device={device})")
        self.model = SentenceTransformer(MODEL_NAME, device=device)
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
