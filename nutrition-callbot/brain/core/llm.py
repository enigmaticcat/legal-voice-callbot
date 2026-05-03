"""
LLM Wrapper — Gemini API hoặc OpenAI-compatible (vLLM, Ollama)

Chọn backend qua env LLM_BACKEND:
  LLM_BACKEND=gemini   → GeminiLLMClient  (default)
  LLM_BACKEND=openai   → OpenAILLMClient  (vLLM / Ollama / LM Studio)

LLMClient = factory function trả về đúng client theo config.
"""
import asyncio
import logging
import time
from typing import AsyncGenerator
from google import genai
from google.genai import types

logger = logging.getLogger("brain.core.llm")


# ── Gemini ────────────────────────────────────────────────────────────────────

class GeminiLLMClient:

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
            response = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=config,
            )

            ttfc_ms = None
            t0 = time.time()

            async for chunk in response:
                if chunk.text:
                    if ttfc_ms is None:
                        ttfc_ms = (time.time() - t0) * 1000
                        logger.info(f"LLM TTFC (first chunk from Gemini): {ttfc_ms:.0f}ms")

                    result = {"text": chunk.text, "is_final": False}
                    if first_token:
                        ttft = (time.time() - start_time) * 1000
                        result["ttft_ms"] = ttft
                        result["ttfc_ms"] = ttfc_ms
                        logger.info(f"LLM TTFT: {ttft:.0f}ms | TTFC: {ttfc_ms:.0f}ms")
                        first_token = False
                    yield result

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            yield {
                "text": "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau.",
                "is_final": True,
                "error": True,
            }

    async def generate(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.3,
    ) -> str:
        full_text = []
        async for chunk in self.generate_stream(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
        ):
            full_text.append(chunk["text"])
        return "".join(full_text)


# ── OpenAI-compatible (vLLM / Ollama) ────────────────────────────────────────

class OpenAILLMClient:

    def __init__(self, base_url: str, model: str, api_key: str = "dummy"):
        from openai import AsyncOpenAI
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        logger.info(f"OpenAI-compat LLM: {model} @ {base_url}")

    async def generate_stream(
        self,
        prompt: str,
        system_instruction: str = "",
        temperature: float = 0.3,
        max_output_tokens: int = 512,
    ) -> AsyncGenerator[dict, None]:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        t0 = time.time()
        ttfc_ms = None
        first_token = True

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
                stream=True,
            )

            async for chunk in stream:
                text = chunk.choices[0].delta.content
                if not text:
                    continue

                if ttfc_ms is None:
                    ttfc_ms = (time.time() - t0) * 1000

                result = {"text": text, "is_final": False}
                if first_token:
                    result["ttfc_ms"] = ttfc_ms
                    first_token = False
                yield result

        except Exception as e:
            logger.error(f"OpenAI-compat LLM error: {e}")
            yield {
                "text": "Xin loi, he thong dang gap su co. Vui long thu lai sau.",
                "is_final": True,
                "error": True,
            }

    async def generate(self, prompt: str, system_instruction: str = "", temperature: float = 0.3) -> str:
        full_text = []
        async for chunk in self.generate_stream(prompt, system_instruction, temperature):
            full_text.append(chunk["text"])
        return "".join(full_text)


# ── Factory ───────────────────────────────────────────────────────────────────

def LLMClient(api_key: str = "", model: str = "gemini-2.5-flash",
              backend: str = None, base_url: str = None):
    """
    Factory: trả GeminiLLMClient hoặc OpenAILLMClient tuỳ LLM_BACKEND env.
      LLM_BACKEND=gemini  (default)
      LLM_BACKEND=openai  + LLM_BASE_URL + LLM_MODEL
    """
    import os
    b = backend or os.getenv("LLM_BACKEND", "gemini")
    if b == "openai":
        url   = base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
        mdl   = os.getenv("LLM_MODEL", model)
        key   = os.getenv("OPENAI_API_KEY", "dummy")
        return OpenAILLMClient(base_url=url, model=mdl, api_key=key)
    return GeminiLLMClient(api_key=api_key, model=model)
