"""
OpenRouter provider adapter.

OpenRouter exposes an OpenAI-compatible API at https://openrouter.ai/api/v1,
so we subclass OpenAIProvider and only override the model ID mapping.

One API key covers OpenAI, Anthropic, DeepSeek, Google, and 200+ other models.
Set USE_OPENROUTER=true and OPENROUTER_API_KEY in .env to enable.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from src.config import settings
from src.providers.base import LLMResponse
from src.providers.openai_provider import OpenAIProvider

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Maps RouteIQ internal model_ids → OpenRouter model slugs
# Browse the full catalogue at https://openrouter.ai/models
OPENROUTER_MODEL_MAP: dict[str, str] = {
    "gpt-5-nano":           "openai/gpt-4o-mini",
    "gemini-2.5-flash-lite":"google/gemini-2.5-flash-lite-preview-06-17",
    "claude-haiku-4-5":     "anthropic/claude-haiku-4-5",
    "deepseek-v3":          "deepseek/deepseek-chat",
    "gpt-5.4":              "openai/gpt-4o",
    "claude-sonnet-4-6":    "anthropic/claude-sonnet-4-6",
    "claude-opus-4-6":      "anthropic/claude-opus-4-6",
}


class OpenRouterProvider(OpenAIProvider):
    """
    Single-key provider that routes all models through OpenRouter.
    Inherits all streaming / non-streaming logic from OpenAIProvider.
    """

    provider_name = "openrouter"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(
            api_key=api_key or settings.openrouter_api_key,
            base_url=_OPENROUTER_BASE_URL,
        )

    def _resolve_model(self, model_id: str) -> str:
        """Translate a RouteIQ model_id to an OpenRouter slug."""
        return OPENROUTER_MODEL_MAP.get(model_id, model_id)

    async def call(
        self,
        messages: list[dict[str, Any]],
        model_id: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        return await super().call(
            messages=messages,
            model_id=self._resolve_model(model_id),
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
        async for chunk in super().call_stream(
            messages=messages,
            model_id=self._resolve_model(model_id),
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        ):
            yield chunk
