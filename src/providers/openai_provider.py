"""
OpenAI provider adapter (GPT-5 Nano, GPT-5.4, etc.)
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI

from src.config import settings
from src.providers.base import BaseProvider, LLMResponse


class OpenAIProvider(BaseProvider):
    provider_name = "openai"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key or settings.openai_api_key,
            **({"base_url": base_url} if base_url else {}),
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
        params: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": False,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        params.update({k: v for k, v in kwargs.items() if v is not None})

        resp = await self._client.chat.completions.create(**params)

        choice = resp.choices[0]
        msg = choice.message
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        return LLMResponse(
            id=resp.id,
            model=resp.model,
            content=msg.content or "",
            role=msg.role,
            finish_reason=choice.finish_reason,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            total_tokens=resp.usage.total_tokens if resp.usage else 0,
            tool_calls=tool_calls,
            raw=resp.model_dump(),
        )

    async def call_stream(
        self,
        messages: list[dict[str, Any]],
        model_id: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        params: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": True,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        params.update({k: v for k, v in kwargs.items() if v is not None})

        async with await self._client.chat.completions.create(**params) as stream:
            async for chunk in stream:
                yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
