import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.rag import RAGPipeline


async def main():
    load_dotenv(ROOT.parent / ".env")

    qdrant_url = os.getenv("QDRANT_URL") or (
        f"http://{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}"
    )
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
    collection = os.getenv("QDRANT_COLLECTION", "nutrition_articles")

    rag = RAGPipeline(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection=collection,
    )

    queries = [
        "Người bị tiểu đường nên ăn gì?",
        "Bà bầu cần bổ sung canxi như thế nào?",
        "Omega 3 có tác dụng gì với sức khỏe?",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        docs = await rag.search(query, top_k=3, fetch_k=10)
        if not docs:
            print("No results")
            continue
        for index, doc in enumerate(docs, start=1):
            print(f"{index}. [{doc.get('score', 0):.4f}] {doc.get('title', '')}")
            print(f"   {doc.get('content', '')[:160]}...")


if __name__ == "__main__":
    asyncio.run(main())
