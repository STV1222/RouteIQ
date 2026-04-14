"""
DeepSeek provider adapter — uses OpenAI SDK with DeepSeek base URL.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from src.config import settings
from src.providers.base import LLMResponse
from src.providers.openai_provider import OpenAIProvider

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Map RouteIQ model_ids to DeepSeek API model strings
_MODEL_MAP: dict[str, str] = {
    "deepseek-v3": "deepseek-chat",
}


class DeepSeekProvider(OpenAIProvider):
    """
    DeepSeek uses the OpenAI-compatible API, so we simply subclass
    OpenAIProvider and point it at the DeepSeek base URL.
    """

    provider_name = "deepseek"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(
            api_key=api_key or settings.deepseek_api_key,
            base_url=_DEEPSEEK_BASE_URL,
        )

    async def call(
        self,
        messages: list[dict[str, Any]],
        model_id: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        api_model = _MODEL_MAP.get(model_id, model_id)
        return await super().call(
            messages=messages,
            model_id=api_model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **kwargs,
        )

    async def call_stream(
        self,
        messages: list[dict[str, Any]],
        model_id: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        api_model = _MODEL_MAP.get(model_id, model_id)
        async for chunk in super().call_stream(
            messages=messages,
            model_id=api_model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        ):
            yield chunk
