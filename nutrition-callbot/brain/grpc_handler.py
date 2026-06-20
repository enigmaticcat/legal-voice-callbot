import logging
import re
import time
from typing import AsyncGenerator

_SOURCE_TAG = re.compile(r'\s*\([A-Z][A-Z0-9,\.\-]*\)')

from .core.llm import LLMClient
from .core.rag import RAGPipeline
from .core.query_expander import expand_query
from .core.prompt import build_prompt, NUTRITION_SYSTEM_PROMPT
from .core.chunker import chunk_llm_stream
from .core.safety import assess_safety
from .core.safe_rag import (
    assess_evidence,
    assess_output_safety,
    clean_retrieved_content,
    clean_voice_output,
    missing_disclaimer_suffix,
)

logger = logging.getLogger("brain.grpc_handler")


class BrainServiceHandler:

    def __init__(self, llm: LLMClient, rag: RAGPipeline,
                 rag_fetch_k: int = 15, rag_top_k: int = 5,
                 min_chunk_size: int = 40, use_hyde: bool = False):
        self.llm = llm
        self.rag = rag
        self.rag_fetch_k = rag_fetch_k
        self.rag_top_k = rag_top_k
        self.min_chunk_size = min_chunk_size
        self.use_hyde = use_hyde

    async def think(
        self,
        query: str,
        session_id: str,
        conversation_history: list = None,
        conversation_summary: str = "",
    ) -> AsyncGenerator[dict, None]:
        start_time = time.time()
        logger.info(f"[{session_id}] Think: {query[:80]}...")
        contexts = []
        retrieval_cache = {"status": "not_checked"}
        expand_ms = 0.0
        rag_ms = 0.0

        try:
            safety = assess_safety(query)
            if safety.triggered:
                total_ms = (time.time() - start_time) * 1000
                logger.info(
                    "[%s] Safety guardrail triggered: %s",
                    session_id,
                    safety.category,
                )
                yield {
                    "text": safety.message,
                    "is_final": False,
                    "safety": {
                        "triggered": True,
                        "category": safety.category,
                    },
                }
                yield {
                    "text": "",
                    "is_final": True,
                    "contexts": [],
                    "safety": {
                        "triggered": True,
                        "category": safety.category,
                    },
                    "timing": {"total_ms": round(total_ms, 1)},
                }
                return

            t0 = time.time()
            expanded = expand_query(query)
            expand_ms = (time.time() - t0) * 1000

            t0 = time.time()
            docs, retrieval_cache = await self.rag.search_with_meta(
                expanded,
                top_k=self.rag_top_k,
                fetch_k=self.rag_fetch_k,
                use_hyde=self.use_hyde,
            )
            rag_ms = (time.time() - t0) * 1000
            contexts = [d.get("content", "") for d in docs]

            evidence = assess_evidence(docs)
            if not evidence.sufficient:
                total_ms = (time.time() - start_time) * 1000
                message = (
                    "Chào bạn, tôi chưa có đủ tài liệu dinh dưỡng đáng tin cậy để trả lời chính xác câu hỏi này. "
                    "Bạn có thể hỏi lại cụ thể hơn về thực phẩm, khẩu phần, vi chất hoặc tình trạng dinh dưỡng. "
                    "Để được tư vấn chính xác, bạn nên gặp bác sĩ dinh dưỡng."
                )
                logger.info(
                    "[%s] Evidence insufficient: %s docs=%d chars=%d",
                    session_id,
                    evidence.reason,
                    evidence.doc_count,
                    evidence.total_chars,
                )
                yield {
                    "text": message,
                    "is_final": False,
                    "retrieval_cache": retrieval_cache,
                    "evidence": {
                        "sufficient": False,
                        "reason": evidence.reason,
                        "doc_count": evidence.doc_count,
                        "total_chars": evidence.total_chars,
                    },
                }
                yield {
                    "text": "",
                    "is_final": True,
                    "contexts": contexts,
                    "retrieval_cache": retrieval_cache,
                    "evidence": {
                        "sufficient": False,
                        "reason": evidence.reason,
                        "doc_count": evidence.doc_count,
                        "total_chars": evidence.total_chars,
                    },
                    "timing": {
                        "expand_ms": round(expand_ms, 1),
                        "rag_ms": round(rag_ms, 1),
                        "total_ms": round(total_ms, 1),
                    },
                }
                return

            context = "\n\n".join(
                f"[Tài liệu {i+1}: {d.get('title','')}]\n{clean_retrieved_content(_SOURCE_TAG.sub('', d.get('content','')))}"
                for i, d in enumerate(docs)
            )
            prompt = build_prompt(
                query=expanded,
                nutrition_context=context,
                conversation_history=conversation_history,
                conversation_summary=conversation_summary,
            )

            t_llm = time.time()
            first_chunk = True
            llm_ttft_ms = 0
            emitted_text = ""
            output_removed_url = False

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

            async for chunk_text in chunk_llm_stream(_llm_with_ttft(), min_size=self.min_chunk_size):
                chunk_text, removed_url = clean_voice_output(chunk_text)
                output_removed_url = output_removed_url or removed_url
                if not chunk_text.strip():
                    continue
                emitted_text += chunk_text
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
                    result["retrieval_cache"] = retrieval_cache
                    first_chunk = False
                yield result

            output_safety = assess_output_safety(emitted_text)
            if output_safety.needs_disclaimer:
                suffix = missing_disclaimer_suffix(emitted_text)
                emitted_text += suffix
                yield {
                    "text": suffix,
                    "is_final": False,
                    "output_safety": {
                        "needs_disclaimer": True,
                        "unsafe_claim_detected": output_safety.unsafe_claim_detected,
                    },
                }

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
            "retrieval_cache": retrieval_cache,
            "evidence": {
                "sufficient": True,
                "reason": "sufficient",
                "doc_count": len(contexts),
                "total_chars": sum(len(context or "") for context in contexts),
            },
            "output_safety": {
                "removed_url": output_removed_url,
                "unsafe_claim_detected": assess_output_safety(emitted_text).unsafe_claim_detected,
            },
            "timing": {"total_ms": round(total_ms, 1)},
        }
