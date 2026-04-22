import logging
import time
from typing import AsyncGenerator

from .core.llm import LLMClient
from .core.rag import RAGPipeline
from .core.query_expander import expand_query
from .core.prompt import build_prompt, NUTRITION_SYSTEM_PROMPT
from .core.chunker import chunk_llm_stream

logger = logging.getLogger("brain.grpc_handler")


class BrainServiceHandler:

    def __init__(self, llm: LLMClient, rag: RAGPipeline):
        self.llm = llm
        self.rag = rag

    async def think(
        self,
        query: str,
        session_id: str,
        conversation_history: list = None,
    ) -> AsyncGenerator[dict, None]:
        start_time = time.time()
        logger.info(f"[{session_id}] Think: {query[:80]}...")
        contexts = []
        expand_ms = 0.0
        rag_ms = 0.0

        try:
            t0 = time.time()
            expanded = expand_query(query)
            expand_ms = (time.time() - t0) * 1000

            t0 = time.time()
            docs = await self.rag.search(expanded)
            rag_ms = (time.time() - t0) * 1000
            contexts = [d.get("content", "") for d in docs]
            context = "\n\n".join(
                f"[{d.get('source','').upper()} - {d.get('title','')}]\n{d.get('content','')}"
                for d in docs
            )
            prompt = build_prompt(
                query=expanded,
                nutrition_context=context,
                conversation_history=conversation_history,
            )

            t_llm = time.time()
            first_chunk = True
            llm_ttft_ms = 0

            llm_stream = self.llm.generate_stream(
                prompt=prompt,
                system_instruction=NUTRITION_SYSTEM_PROMPT,
            )

            async def _llm_with_ttft():
                nonlocal llm_ttft_ms
                async for chunk in llm_stream:
                    if llm_ttft_ms == 0 and chunk.get("ttfc_ms"):
                        llm_ttft_ms = chunk["ttfc_ms"]
                    yield chunk

            async for chunk_text in chunk_llm_stream(_llm_with_ttft()):
                result = {"text": chunk_text, "is_final": False}

                if first_chunk:
                    first_chunk_total_ms = (time.time() - start_time) * 1000
                    llm_full_ms = (time.time() - t_llm) * 1000
                    result["timing"] = {
                        "expand_ms": round(expand_ms, 1),
                        "rag_ms": round(rag_ms, 1),
                        "llm_ttft_ms": round(llm_ttft_ms, 1),
                        "llm_full_ms": round(llm_full_ms, 1),
                        "first_chunk_total_ms": round(first_chunk_total_ms, 1),
                    }
                    first_chunk = False
                yield result

        except Exception as e:
            total_ms = (time.time() - start_time) * 1000
            logger.exception("[%s] Brain think failed", session_id)
            yield {
                "text": "Xin lỗi, tôi đang gặp lỗi truy xuất dữ liệu. Vui lòng thử lại sau.",
                "is_final": True,
                "error": True,
                "error_type": type(e).__name__,
                "timing": {
                    "expand_ms": round(expand_ms, 1),
                    "rag_ms": round(rag_ms, 1),
                    "total_ms": round(total_ms, 1),
                },
            }
            return

        total_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[{session_id}] Done in {total_ms:.0f}ms "
            f"(expand={expand_ms:.0f}, rag={rag_ms:.0f})"
        )
        yield {
            "text": "",
            "is_final": True,
            "contexts": contexts,
            "timing": {"total_ms": round(total_ms, 1)},
        }
