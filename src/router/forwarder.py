"""
Request forwarder — dispatches to the correct provider and handles fallback.

Supports both streaming and non-streaming responses.  On provider failure,
automatically retries with the next cheapest capable model (up to
settings.max_fallback_attempts times).
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Optional

from src.config import settings
from src.providers.base import BaseProvider, LLMResponse
from src.providers.openai_provider import OpenAIProvider
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.deepseek_provider import DeepSeekProvider
from src.providers.gemini_provider import GeminiProvider
from src.providers.openrouter_provider import OpenRouterProvider
from src.router.classifier import ComplexityResult
from src.router.scorer import ModelSelection, get_fallback_models

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry — instantiated once at module load
#
# When USE_OPENROUTER=true every provider key maps to the single
# OpenRouterProvider instance, so only OPENROUTER_API_KEY is required.
# ---------------------------------------------------------------------------

def _build_registry() -> dict[str, BaseProvider]:
    if settings.use_openrouter:
        logger.info("OpenRouter mode enabled — routing all models through openrouter.ai")
        _or = OpenRouterProvider()
        return {
            "openai":      _or,
            "anthropic":   _or,
            "deepseek":    _or,
            "gemini":      _or,
            "openrouter":  _or,
        }
    return {
        "openai":    OpenAIProvider(),
        "anthropic": AnthropicProvider(),
        "deepseek":  DeepSeekProvider(),
        "gemini":    GeminiProvider(),
    }

_PROVIDERS: dict[str, BaseProvider] = _build_registry()


def get_provider(provider_name: str) -> BaseProvider:
    provider = _PROVIDERS.get(provider_name)
    if provider is None:
        raise ValueError(f"Unknown provider: '{provider_name}'")
    return provider


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def forward(
    messages: list[dict[str, Any]],
    selection: ModelSelection,
    complexity: ComplexityResult,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    stream: bool = False,
    extra_kwargs: Optional[dict[str, Any]] = None,
) -> LLMResponse:
    """
    Forward a non-streaming request to the selected provider.
    Automatically falls back to the next cheapest capable model on failure.
    """
    kwargs = extra_kwargs or {}
    attempted: list[str] = []

    # Build the ordered list of candidates: selected first, then fallbacks
    candidates = [(selection.model_id, selection.provider)] + [
        (m.model_id, m.provider)
        for m in get_fallback_models(complexity, exclude=[selection.model_id])
    ]

    for model_id, provider_name in candidates[: settings.max_fallback_attempts + 1]:
        attempted.append(model_id)
        provider = get_provider(provider_name)
        try:
            start = time.monotonic()
            response = await provider.call(
                messages=messages,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                **kwargs,
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "Provider call succeeded model=%s provider=%s latency_ms=%d",
                model_id, provider_name, elapsed_ms,
            )
            return response
        except Exception as exc:
            logger.warning(
                "Provider call failed model=%s provider=%s error=%s — trying fallback",
                model_id, provider_name, exc,
            )
            if model_id == candidates[-1][0]:
                raise RuntimeError(
                    f"All providers failed. Attempted: {attempted}. Last error: {exc}"
                ) from exc

    raise RuntimeError(f"All providers failed. Attempted: {attempted}")


async def forward_stream(
    messages: list[dict[str, Any]],
    selection: ModelSelection,
    complexity: ComplexityResult,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    extra_kwargs: Optional[dict[str, Any]] = None,
) -> AsyncIterator[str]:
    """
    Forward a streaming request.  Yields SSE lines.

    Falls back to the next capable model if the primary provider raises
    before yielding any chunks.  Once streaming has begun we cannot
    transparently retry, so we let remaining errors propagate.
    """
    kwargs = extra_kwargs or {}
    attempted: list[str] = []

    candidates = [(selection.model_id, selection.provider)] + [
        (m.model_id, m.provider)
        for m in get_fallback_models(complexity, exclude=[selection.model_id])
    ]

    for model_id, provider_name in candidates[: settings.max_fallback_attempts + 1]:
        attempted.append(model_id)
        provider = get_provider(provider_name)
        try:
            gen = provider.call_stream(
                messages=messages,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            # Async generators need to be iterated; wrap in a helper so we
            # can distinguish "failed before first yield" from mid-stream errors.
            async for chunk in gen:
                yield chunk
            return  # success — done
        except Exception as exc:
            logger.warning(
                "Streaming provider failed model=%s provider=%s error=%s — trying fallback",
                model_id, provider_name, exc,
            )
            if model_id == candidates[-1][0]:
                raise RuntimeError(
                    f"All streaming providers failed. Attempted: {attempted}. Last error: {exc}"
                ) from exc
