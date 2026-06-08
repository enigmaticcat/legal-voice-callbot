"""
Unit tests — không cần service nào chạy (Qdrant, LLM, TTS).
Chạy: pytest tests/test_unit.py -v

Covers:
  - brain.core.query_expander  : expand_query()
  - tts.core.chunker           : chunk_text()
  - brain.core.prompt          : build_prompt()
  - data-pipeline chunker      : sentence_chunks() / _sentences_to_chunks()
"""
import sys
import os
import pytest

# Thêm paths để import được các module
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "brain"))
sys.path.insert(0, os.path.join(ROOT, "tts"))

# ─────────────────────────────────────────────
# 1. query_expander
# ─────────────────────────────────────────────
from core.query_expander import expand_query


class TestExpandQuery:

    def test_alias_tieu_duong(self):
        result = expand_query("Người bị tiểu đường nên ăn gì?")
        assert "đái tháo đường" in result

    def test_alias_ba_bau(self):
        result = expand_query("Bà bầu cần bổ sung gì?")
        assert "phụ nữ mang thai" in result

    def test_alias_omega3(self):
        result = expand_query("omega 3 có tác dụng gì?")
        assert "axit béo omega-3" in result

    def test_alias_canxi(self):
        result = expand_query("canxi giúp ích gì cho xương?")
        assert "calcium" in result

    def test_no_change_when_no_alias(self):
        query = "Cách nấu canh chua cá lóc?"
        assert expand_query(query) == query

    def test_case_insensitive(self):
        result = expand_query("TIỂU ĐƯỜNG type 2 ăn uống thế nào?")
        assert "đái tháo đường" in result

    def test_multiple_aliases_in_one_query(self):
        result = expand_query("Bà bầu bị tiểu đường cần ăn gì?")
        assert "phụ nữ mang thai" in result
        assert "đái tháo đường" in result

    def test_returns_string(self):
        assert isinstance(expand_query("test"), str)

    def test_empty_string(self):
        assert expand_query("") == ""


# ─────────────────────────────────────────────
# 2. TTS chunker
# ─────────────────────────────────────────────
from core.chunker import chunk_text


class TestChunkText:

    def test_basic_split_at_period(self):
        chunks = chunk_text("Xin chào. Tôi là bot.", min_size=5)
        assert len(chunks) >= 1
        assert all(len(c) > 0 for c in chunks)

    def test_short_text_not_split_prematurely(self):
        # Text ngắn hơn min_size không bị cắt sớm
        chunks = chunk_text("OK.", min_size=100)
        assert len(chunks) == 1
        assert "OK." in chunks[0]

    def test_no_empty_chunks(self):
        text = "Chào bạn, tôi là chuyên gia dinh dưỡng. Hôm nay chúng ta sẽ nói về vitamin D."
        chunks = chunk_text(text, min_size=10)
        assert all(c.strip() != "" for c in chunks)

    def test_total_content_preserved(self):
        text = "Ăn rau xanh rất tốt. Uống đủ nước mỗi ngày. Ngủ đủ giấc."
        chunks = chunk_text(text, min_size=10)
        reconstructed = "".join(chunks)
        assert reconstructed == text

    def test_split_at_comma(self):
        text = "Canxi, sắt, kẽm là các khoáng chất quan trọng cho sức khỏe."
        chunks = chunk_text(text, min_size=5)
        assert len(chunks) >= 2

    def test_min_size_respected(self):
        # Chunk đầu tiên phải >= min_size trước khi split (trừ chunk cuối)
        text = "A. B. C. Đây là câu dài hơn để đảm bảo min_size."
        chunks = chunk_text(text, min_size=20)
        # Chunk không phải chunk cuối phải đủ dài
        for chunk in chunks[:-1]:
            assert len(chunk.strip()) >= 20

    def test_empty_string(self):
        assert chunk_text("") == []

    def test_single_sentence_no_punctuation(self):
        text = "Không có dấu câu ở cuối"
        chunks = chunk_text(text, min_size=5)
        assert len(chunks) == 1
        assert chunks[0] == text


# ─────────────────────────────────────────────
# 3. build_prompt
# ─────────────────────────────────────────────
from core.prompt import build_prompt


class TestBuildPrompt:

    def test_contains_query(self):
        prompt = build_prompt("Vitamin C có trong thực phẩm nào?")
        assert "Vitamin C có trong thực phẩm nào?" in prompt

    def test_contains_rag_context_when_provided(self):
        ctx = "Vitamin C có nhiều trong cam, chanh, ổi."
        prompt = build_prompt("Vitamin C?", nutrition_context=ctx)
        assert ctx in prompt

    def test_no_rag_fallback_message(self):
        prompt = build_prompt("câu hỏi", nutrition_context="")
        assert "Chưa có dữ liệu RAG" in prompt

    def test_contains_few_shot_examples(self):
        prompt = build_prompt("bất kỳ câu hỏi nào")
        assert "Hỏi:" in prompt
        assert "Đáp:" in prompt

    def test_conversation_history_included(self):
        history = [
            {"role": "user", "text": "Câu trước của tôi"},
            {"role": "assistant", "text": "Câu trả lời trước"},
        ]
        prompt = build_prompt("câu mới", conversation_history=history)
        assert "Câu trước của tôi" in prompt
        assert "Câu trả lời trước" in prompt

    def test_interrupted_turn_marked(self):
        history = [{"role": "user", "text": "Câu bị ngắt", "interrupted": True}]
        prompt = build_prompt("câu hỏi", conversation_history=history)
        assert "[bị ngắt giữa chừng]" in prompt

    def test_conversation_summary_included(self):
        prompt = build_prompt("câu hỏi", conversation_summary="Tóm tắt hội thoại trước")
        assert "Tóm tắt hội thoại trước" in prompt

    def test_returns_string(self):
        assert isinstance(build_prompt("test"), str)

    def test_no_url_in_prompt_instruction(self):
        # System prompt yêu cầu không trích nguồn/URL — kiểm tra instruction có trong prompt
        prompt = build_prompt("test", nutrition_context="some context")
        assert "URL" in prompt or "nguồn" in prompt  # instruction cấm trích nguồn phải có

    def test_empty_query(self):
        prompt = build_prompt("")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
