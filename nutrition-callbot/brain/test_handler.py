import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from core.llm import LLMClient
from core.rag import RAGPipeline
from grpc_handler import BrainServiceHandler

async def test():
    print("1. Khởi tạo LLMClient & RAGPipeline...")
    llm = LLMClient(api_key=config.gemini_api_key, model=config.gemini_model)
    rag = RAGPipeline(
        qdrant_url=config.qdrant_url,
        qdrant_api_key=config.qdrant_api_key,
        collection=config.qdrant_collection,
    )
    
    print("2. Khởi tạo BrainServiceHandler...")
    handler = BrainServiceHandler(llm=llm, rag=rag)
    
    query = "Nghỉ phép năm bao nhiêu ngày?"
    print(f"\n3. Gửi query: {query}")
    print("-" * 50)
    
    async for chunk in handler.think(query, "test-session", []):
        if chunk.get("text"):
            print(chunk["text"], end="", flush=True)
        if chunk.get("timing") and not chunk.get("is_final"):
            timing_breakdown = chunk["timing"]
        if chunk.get("is_final"):
            print(f"\n\n[Timing Breakdown]")
            if timing_breakdown:
                for k, v in timing_breakdown.items():
                    print(f"  {k}: {v}ms")
            print(f"  total_ms: {chunk['timing'].get('total_ms', 'N/A')}ms")

if __name__ == "__main__":
    asyncio.run(test())
