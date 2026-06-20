"""
Test RAG Pipeline — Nutrition collection + local Qwen.

Chạy:
  python test_rag_pipeline.py
"""
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.llm import LLMClient
from core.prompt import NUTRITION_SYSTEM_PROMPT, build_prompt
from core.query_expander import expand_query
from core.rag import RAGPipeline


env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)


async def main():
    qdrant_url = os.getenv("QDRANT_URL") or (
        f"http://{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}"
    )
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
    collection = os.getenv("QDRANT_COLLECTION", "nutrition_articles")
    llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    llm_model = os.getenv("LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    llm_api_key = os.getenv("LLM_API_KEY", "local")

    print("1. Khởi tạo LLM + RAG...")
    llm = LLMClient(
        api_key=llm_api_key,
        model=llm_model,
        base_url=llm_base_url,
    )
    rag = RAGPipeline(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection=collection,
        llm_client=llm,
    )

    query = "Người bị tiểu đường nên ăn gì để kiểm soát đường huyết?"
    expanded_query = expand_query(query)
    if expanded_query != query:
        print(f"Query mở rộng: {expanded_query}")

    print(f"\n2. Tìm kiếm collection `{collection}`...")
    docs = await rag.search(expanded_query, top_k=5, fetch_k=15)
    if not docs:
        print("Không tìm thấy tài liệu phù hợp trong Qdrant.")
        return

    print(f"Tìm thấy {len(docs)} tài liệu:\n")
    for index, doc in enumerate(docs, start=1):
        print(f"  {index}. [{doc.get('score', 0):.4f}] {doc.get('title', '')}")
        print(f"     {doc.get('content', '')[:150]}...")

    nutrition_context = "\n\n".join(
        f"[Tài liệu {index}: {doc.get('title', '')}]\n{doc.get('content', '')}"
        for index, doc in enumerate(docs, start=1)
    )
    prompt = build_prompt(query=query, nutrition_context=nutrition_context)

    print("\n3. Gọi Qwen local suy luận...")
    print("-" * 50)
    async for chunk in llm.generate_stream(
        prompt=prompt,
        system_instruction=NUTRITION_SYSTEM_PROMPT,
    ):
        if chunk.get("text"):
            print(chunk["text"], end="", flush=True)

    print("\n" + "-" * 50)
    print("Hoàn tất.")


if __name__ == "__main__":
    asyncio.run(main())
