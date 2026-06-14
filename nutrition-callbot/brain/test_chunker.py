"""
Smoke test for the current LLM stream chunker.

Chạy:
  python test_chunker.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.chunker import chunk_llm_stream


async def _mock_stream(parts):
    for text in parts:
        yield {"text": text, "is_final": False}


async def test_chunker():
    parts = [
        "Chào bạn, chế độ ăn cho người đái tháo đường cần ưu tiên thực phẩm giàu chất xơ. ",
        "Bạn nên chọn rau xanh, đạm nạc, cá, đậu phụ và tinh bột hấp thu chậm. ",
        "Hạn chế nước ngọt, bánh kẹo và không bỏ bữa nếu đang dùng thuốc điều trị.",
    ]

    chunks = [
        chunk async for chunk in chunk_llm_stream(_mock_stream(parts), min_size=40)
    ]

    print(f"Generated {len(chunks)} chunks")
    for index, chunk in enumerate(chunks, start=1):
        print(f"{index}. ({len(chunk)} chars) {chunk}")

    assert chunks, "Expected at least one chunk"
    assert "".join(chunks).replace("  ", " ").strip()


if __name__ == "__main__":
    asyncio.run(test_chunker())
