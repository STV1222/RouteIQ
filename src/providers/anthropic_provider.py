"""
Anthropic provider adapter (Claude Haiku 4.5, Sonnet 4.6, Opus 4.6).

Handles the OpenAI → Anthropic message format conversion, including:
  - system message extraction
  - tool_calls ↔ tool_use block conversion
  - streaming SSE re-encoding to OpenAI format
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Optional

import anthropic

from src.config import settings
from src.providers.base import BaseProvider, LLMResponse

# Map RouteIQ model_ids to Anthropic API model strings
_MODEL_MAP: dict[str, str] = {
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-opus-4-6": "claude-opus-4-6",
}
_DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(BaseProvider):
    provider_name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or settings.anthropic_api_key
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def call(
        self,
        messages: list[dict[str, Any]],
        model_id: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        system, anthropic_messages = _convert_messages(messages)
        anthropic_model = _MODEL_MAP.get(model_id, model_id)

        params: dict[str, Any] = {
            "model": anthropic_model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
        }
        if system:
            params["system"] = system
        if temperature is not None:
            params["temperature"] = temperature

        # Forward tool definitions if present
        if kwargs.get("tools"):
            params["tools"] = _convert_tools(kwargs["tools"])

        resp = await self._client.messages.create(**params)

        content_text = ""
        tool_calls = None
        for block in resp.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                tool_calls = tool_calls or []
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

        return LLMResponse(
            id=resp.id,
            model=resp.model,
            content=content_text,
            role="assistant",
            finish_reason=_map_stop_reason(resp.stop_reason),
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
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
        system, anthropic_messages = _convert_messages(messages)
        anthropic_model = _MODEL_MAP.get(model_id, model_id)
        call_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        params: dict[str, Any] = {
            "model": anthropic_model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
        }
        if system:
            params["system"] = system
        if temperature is not None:
            params["temperature"] = temperature

        # Yield an initial role chunk (OpenAI convention)
        role_chunk = {
            "id": call_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"

        async with self._client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                chunk = {
                    "id": call_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

        # Final done chunk
        done_chunk = {
            "id": call_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done_chunk)}\n\n"
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _convert_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Split OpenAI-style messages into (system_prompt, anthropic_messages).
    Anthropic does not accept system messages inside the messages array.
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            continue

        # Map OpenAI tool role → Anthropic tool_result user message
        if role == "tool":
            converted.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": content if isinstance(content, str) else json.dumps(content),
                    }
                ],
            })
            continue

        # Map assistant tool_calls → Anthropic tool_use blocks
        if role == "assistant" and msg.get("tool_calls"):
            blocks: list[dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in msg["tool_calls"]:
                blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": json.loads(tc["function"].get("arguments", "{}")),
                })
            converted.append({"role": "assistant", "content": blocks})
            continue

        # Standard text message
        if isinstance(content, list):
            # Multimodal — pass through (Anthropic format is compatible enough)
            converted.append({"role": role, "content": content})
        else:
            converted.append({"role": role, "content": content or ""})

    return "\n\n".join(system_parts), converted


def _convert_tools(openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tool definitions to Anthropic tool format."""
    result = []
    for tool in openai_tools:
        fn = tool.get("function", tool)
        result.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


def _map_stop_reason(reason: str | None) -> str:
    mapping = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }
    return mapping.get(reason or "", "stop")
