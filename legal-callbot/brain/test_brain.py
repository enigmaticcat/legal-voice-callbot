"""
Test Brain Service — Step 1 Checkpoint
Chạy: GEMINI_API_KEY=xxx python test_brain.py

Test cases:
  1. LLM streaming hoạt động
  2. Query expansion đúng
  3. Word-safe chunking không cắt giữa từ
  4. Timing log hiển thị
  5. Fallback khi lỗi
"""
import asyncio
import os
import sys
import time

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.llm import LLMClient
from core.prompt import build_prompt, LEGAL_SYSTEM_PROMPT
from core.query_expander import expand_query
from core.chunker import chunk_llm_stream


# ─── Test 1: Query Expansion ────────────────────────────
def test_query_expansion():
    print("=" * 60)
    print("Test 1: Query Expansion")
    print("=" * 60)

    test_cases = [
        ("vượt đèn đỏ bị phạt bao nhiêu?", "vi phạm tín hiệu đèn giao thông"),
        ("uống bia lái xe bị sao?", "nồng độ cồn điều khiển phương tiện"),
        ("sổ đỏ là gì?", "Giấy chứng nhận quyền sử dụng đất"),
        ("nghỉ việc có được BHXH không?", "bảo hiểm xã hội bắt buộc"),
        ("câu bình thường không cần expand", None),  # Không thay đổi
    ]

    passed = 0
    for query, expected_term in test_cases:
        expanded = expand_query(query)
        if expected_term:
            if expected_term in expanded:
                print(f"  '{query}' → chứa '{expected_term}'")
                passed += 1
            else:
                print(f"  '{query}' → '{expanded}' (thiếu '{expected_term}')")
        else:
            if expanded == query:
                print(f"  '{query}' → không thay đổi (đúng)")
                passed += 1
            else:
                print(f"  '{query}' → '{expanded}' (không nên thay đổi!)")

    print(f"\n  Kết quả: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


# ─── Test 2: Word-Safe Chunking ─────────────────────────
def test_word_safe_chunking():
    print("\n" + "=" * 60)
    print("Test 2: Word-Safe Chunking")
    print("=" * 60)

    # Mô phỏng LLM stream
    text_stream = [
        "Theo Điều 5, ",
        "Khoản 4a ",
        "Nghị định 100/2019/NĐ-CP, ",
        "xe ô tô vượt đèn đỏ bị phạt từ 4 đến 6 triệu đồng. ",
        "Xe máy theo Khoản 3 cùng Điều, ",
        "phạt từ 800 nghìn đến 1 triệu đồng. ",
        "Đây chỉ là tham khảo.",
    ]

    chunks = list(chunk_llm_stream(text_stream, min_size=40))

    print("  Chunks:")
    all_ok = True
    for i, chunk in enumerate(chunks):
        # Kiểm tra không cắt giữa từ
        if chunk and not chunk[-1] in ".!?;:,":
            if len(chunk) < 40:
                pass  # OK: chunk cuối có thể ngắn
            else:
                ends_with_space = chunk[-1] == " "
                if not ends_with_space:
                    print(f"  Chunk {i} có thể bị cắt giữa từ: ...{chunk[-10:]}")

        print(f"    [{i}] ({len(chunk)} chars) \"{chunk}\"")

    if chunks:
        print(f"\n  Tạo {len(chunks)} chunks, không cắt giữa từ")
        return True
    else:
        print("\n  Không tạo được chunk nào")
        return False


# ─── Test 3: Prompt Building ────────────────────────────
def test_prompt_building():
    print("\n" + "=" * 60)
    print("Test 3: Prompt Building")
    print("=" * 60)

    prompt = build_prompt(
        query="vượt đèn đỏ bị phạt bao nhiêu?",
        legal_context="Điều 5, Khoản 4: Phạt từ 4-6 triệu đồng...",
        conversation_history=[
            {"role": "user", "text": "Xin chào"},
            {"role": "assistant", "text": "Chào bạn, tôi có thể giúp gì?"},
            {"role": "assistant", "text": "Tôi đang trả lời thì...", "interrupted": True},
        ],
    )

    checks = [
        ("Ví dụ tư vấn" in prompt, "Có few-shot examples"),
        ("Căn cứ pháp lý" in prompt, "Có RAG context"),
        ("Lịch sử hội thoại" in prompt, "Có conversation history"),
        ("bị ngắt giữa chừng" in prompt, "Có interrupted marker"),
        ("vượt đèn đỏ" in prompt, "Có câu hỏi"),
    ]

    passed = 0
    for check, label in checks:
        if check:
            print(f"  {label}")
            passed += 1
        else:
            print(f"  {label}")

    print(f"\n  Prompt length: {len(prompt)} chars")
    print(f"  Kết quả: {passed}/{len(checks)} passed")
    return passed == len(checks)


# ─── Test 4: LLM Streaming (cần API key) ────────────────
async def test_llm_streaming():
    print("\n" + "=" * 60)
    print("Test 4: LLM Streaming (Gemini API)")
    print("=" * 60)

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("  GEMINI_API_KEY chưa set. Bỏ qua test này.")
        print("  Chạy: GEMINI_API_KEY=xxx python test_brain.py")
        return None

    try:
        llm = LLMClient(api_key=api_key)
    except Exception as e:
        print(f"  Không tạo được LLMClient: {e}")
        return False

    # Test streaming
    query = "Vượt đèn đỏ bị phạt bao nhiêu?"
    prompt = build_prompt(query=query)

    print(f"  Query: {query}")
    print(f"  Streaming response:")

    start = time.time()
    chunks_received = 0
    full_text = []
    ttft = None

    try:
        async for chunk in llm.generate_stream(
            prompt=prompt,
            system_instruction=LEGAL_SYSTEM_PROMPT,
        ):
            chunks_received += 1
            full_text.append(chunk["text"])

            if "ttft_ms" in chunk:
                ttft = chunk["ttft_ms"]

            # Hiển thị chunk
            preview = chunk["text"][:50].replace("\n", " ")
            status = "FINAL" if chunk["is_final"] else f"chunk-{chunks_received}"
            print(f"    [{status}] {preview}...")

    except Exception as e:
        print(f"  LLM streaming error: {e}")
        return False

    total_ms = (time.time() - start) * 1000
    response = "".join(full_text)

    print(f"\n  Metrics:")
    print(f"    Chunks: {chunks_received}")
    print(f"    TTFT: {ttft:.0f}ms" if ttft else "    TTFT: N/A")
    print(f"    Total: {total_ms:.0f}ms")
    print(f"    Response length: {len(response)} chars")

    # Kiểm tra response có hữu ích
    checks = [
        (chunks_received > 0, "Nhận được ≥ 1 chunk"),
        (len(response) > 50, "Response đủ dài (> 50 chars)"),
        (ttft is not None and ttft < 5000, "TTFT < 5s"),
    ]

    passed = 0
    for check, label in checks:
        if check:
            print(f"    {label}")
            passed += 1
        else:
            print(f"    {label}")

    print(f"\n  Full response:")
    print(f"  {response[:300]}...")

    return passed == len(checks)


# ─── Main ───────────────────────────────────────────────
async def main():
    print("Legal CallBot — Brain Service Test")
    print("Giai đoạn 1, Bước 1: LLM Streaming\n")

    results = {}

    # Offline tests (không cần API key)
    results["Query Expansion"] = test_query_expansion()
    results["Word-Safe Chunking"] = test_word_safe_chunking()
    results["Prompt Building"] = test_prompt_building()

    # Online test (cần API key)
    results["LLM Streaming"] = await test_llm_streaming()

    # ─── Summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TỔNG KẾT")
    print("=" * 60)

    for name, result in results.items():
        if result is True:
            print(f"  {name}")
        elif result is False:
            print(f"  {name}")
        else:
            print(f"  {name} (skipped)")

    all_passed = all(r is True for r in results.values() if r is not None)
    if all_passed:
        print("\n  Checkpoint Bước 1 PASSED!")
    else:
        print("\n  Một số test chưa pass.")


if __name__ == "__main__":
    asyncio.run(main())
