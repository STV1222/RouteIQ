"""
Abstract base class for all LLM providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional


@dataclass
class LLMResponse:
    id: str
    model: str
    content: str                        # full text of the completion
    role: str = "assistant"
    finish_reason: Optional[str] = "stop"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_calls: Optional[list[dict[str, Any]]] = None
    raw: Optional[dict[str, Any]] = None   # full upstream response for pass-through


class BaseProvider(ABC):
    """Common interface every provider adapter must implement."""

    provider_name: str = ""

    @abstractmethod
    async def call(
        self,
        messages: list[dict[str, Any]],
        model_id: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        stream: bool,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Non-streaming call.  Must return a fully resolved LLMResponse.
        """
        ...

    @abstractmethod
    async def call_stream(
        self,
        messages: list[dict[str, Any]],
        model_id: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Streaming call.  Must yield raw SSE lines (``data: {...}\\n\\n``)
        exactly as the OpenAI streaming spec requires, so the gateway
        can forward them byte-for-byte to the client.
        """
        ...
