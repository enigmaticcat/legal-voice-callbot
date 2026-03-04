"""
gRPC Handler — Think() implementation
Hiện tại dùng HTTP server. Bước 2 sẽ chuyển sang gRPC thật.
Pipeline: Query Expansion → RAG → Build Prompt → LLM Streaming → Word-Safe Chunks
"""
import logging
import time
from typing import AsyncGenerator

from core.llm import LLMClient
from core.rag import RAGPipeline
from core.query_expander import expand_query
from core.prompt import build_prompt, LEGAL_SYSTEM_PROMPT
from core.chunker import chunk_llm_stream

logger = logging.getLogger("brain.grpc_handler")


class BrainServiceHandler:
    """
    Handler cho Brain gRPC Service.

    Think():
      1. Nhận query từ Gateway
      2. Expand query (từ điển pháp lý)
      3. RAG search → top 3 Điều luật (dummy ở Step 1)
      4. Build prompt (system + context + query)
      5. LLM generate streaming → word-safe chunks
    """

    def __init__(self, llm: LLMClient, rag: RAGPipeline):
        self.llm = llm
        self.rag = rag

    async def think(
        self,
        query: str,
        session_id: str,
        conversation_history: list = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Xử lý câu hỏi pháp luật — streaming response.

        Yields:
            {"text": str, "is_final": bool, "timing": dict}
        """
        start_time = time.time()
        logger.info(f"[{session_id}] Think: {query[:80]}...")

        # ── Step 1: Expand query ──────────────────────────────
        t0 = time.time()
        expanded = expand_query(query)
        expand_ms = (time.time() - t0) * 1000

        # ── Step 2: RAG search (dummy ở Step 1, real ở Step 3) ──
        t0 = time.time()
        legal_docs = await self.rag.search(expanded)
        rag_ms = (time.time() - t0) * 1000

        # ── Step 3: Build prompt ──────────────────────────────
        context = "\n".join([d.get("content", "") for d in legal_docs])
        prompt = build_prompt(
            query=expanded,
            legal_context=context,
            conversation_history=conversation_history,
        )

        # ── Step 4: LLM streaming with word-safe chunking ────
        t_llm = time.time()
        first_chunk = True
        ttfc_ms = 0  # Time to First LLM Chunk

        llm_stream = self.llm.generate_stream(
            prompt=prompt,
            system_instruction=LEGAL_SYSTEM_PROMPT,
        )

        # Word-safe chunking (Async)
        async for chunk_text in chunk_llm_stream(llm_stream):
            result = {"text": chunk_text, "is_final": False}

            if first_chunk:
                first_chunk_total_ms = (time.time() - start_time) * 1000
                llm_ms = (time.time() - t_llm) * 1000
                result["timing"] = {
                    "expand_ms": round(expand_ms, 1),
                    "rag_ms": round(rag_ms, 1),
                    "llm_full_ms": round(llm_ms, 1),
                    "first_chunk_total_ms": round(first_chunk_total_ms, 1),
                }
                first_chunk = False
            yield result

        # Final signal
        total_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[{session_id}] Done in {total_ms:.0f}ms "
            f"(expand={expand_ms:.0f}, rag={rag_ms:.0f})"
        )
        yield {
            "text": "",
            "is_final": True,
            "timing": {"total_ms": round(total_ms, 1)},
        }
