"""Local LLM client using an OpenAI-compatible inference server."""
import logging
import time
from typing import AsyncGenerator

logger = logging.getLogger("brain.core.llm")


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
                "text": "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau.",
                "is_final": True,
                "error": True,
            }

    async def generate(self, prompt: str, system_instruction: str = "", temperature: float = 0.3) -> str:
        full_text = []
        async for chunk in self.generate_stream(prompt, system_instruction, temperature):
            full_text.append(chunk["text"])
        return "".join(full_text)


def LLMClient(
    api_key: str = "local",
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    base_url: str = None,
):
    """Create a client for the locally hosted Qwen inference server."""
    import os
    url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    model_name = os.getenv("LLM_MODEL", model)
    local_key = os.getenv("LLM_API_KEY", api_key)
    return OpenAILLMClient(base_url=url, model=model_name, api_key=local_key)
