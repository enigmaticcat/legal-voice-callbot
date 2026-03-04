"""
LLM Wrapper — Gemini API Streaming
Gọi Gemini 2.0 Flash API và stream text response.
Sử dụng google-genai SDK (mới nhất).
"""
import asyncio
import logging
import time
from typing import AsyncGenerator

from google import genai
from google.genai import types

logger = logging.getLogger("brain.core.llm")


class LLMClient:
    """
    Wrapper cho Gemini API streaming.

    Config:
      - model: gemini-2.0-flash (nhanh, rẻ, đủ chính xác cho tư vấn pháp lý)
      - temperature: 0.3 (thấp cho pháp lý, giảm hallucination)
      - streaming: True
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required. Set env GEMINI_API_KEY.")

        self.model = model
        self.client = genai.Client(api_key=api_key)
        logger.info(f"LLM client initialized (model: {model})")

    async def generate_stream(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.3,
        max_output_tokens: int = 4096,
    ) -> AsyncGenerator[dict, None]:
        """
        Sinh text streaming từ Gemini API.

        Yields:
            {"text": str, "is_final": bool, "ttft_ms": float (chỉ chunk đầu)}
        """
        start_time = time.time()
        first_token = True

        logger.debug(f"Generating response for: {prompt[:80]}...")

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # Tắt thinking để giảm latency
        )
        if system_instruction:
            config.system_instruction = system_instruction

        try:
            # Gọi Gemini sync streaming trong thread pool
            collected, ttfc_ms = await asyncio.to_thread(
                self._sync_stream, prompt, config
            )

            for text in collected:
                chunk = {"text": text, "is_final": False}
                if first_token:
                    ttft = (time.time() - start_time) * 1000
                    chunk["ttft_ms"] = ttft
                    chunk["ttfc_ms"] = ttfc_ms  # Time to first LLM chunk
                    logger.info(f"LLM TTFT: {ttft:.0f}ms | TTFC: {ttfc_ms:.0f}ms")
                    first_token = False
                yield chunk

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            yield {
                "text": "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau.",
                "is_final": True,
                "error": True,
            }

    def _sync_stream(
        self, prompt: str, config: types.GenerateContentConfig
    ) -> tuple:
        """
        Gọi Gemini API sync (chạy trong thread pool).
        Trả (list of text strings, ttfc_ms).
        """
        collected = []
        ttfc_ms = None
        t0 = time.time()
        try:
            response = self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=config,
            )
            for chunk in response:
                if chunk.text:
                    if ttfc_ms is None:
                        ttfc_ms = (time.time() - t0) * 1000
                        logger.info(f"LLM TTFC (first chunk from Gemini): {ttfc_ms:.0f}ms")
                    collected.append(chunk.text)
        except Exception as e:
            logger.error(f"Sync stream error: {e}")
            raise
        return collected, ttfc_ms or 0

    async def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.3,
    ) -> str:
        """
        Non-streaming: trả full response text.
        Dùng cho test hoặc khi không cần streaming.
        """
        full_text = []
        async for chunk in self.generate_stream(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
        ):
            full_text.append(chunk["text"])
        return "".join(full_text)
