"""
Google Gemini provider adapter (Gemini 2.5 Flash-Lite, Flash).

Uses google-generativeai SDK and re-encodes responses to OpenAI format.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Optional

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from src.config import settings
from src.providers.base import BaseProvider, LLMResponse

# Map RouteIQ model_ids to Gemini API model names
_MODEL_MAP: dict[str, str] = {
    "gemini-2.5-flash-lite": "gemini-2.5-flash-8b",
    "gemini-2.5-flash":      "gemini-2.5-flash",
}

_DEFAULT_MAX_TOKENS = 4096


class GeminiProvider(BaseProvider):
    provider_name = "gemini"

    def __init__(self, api_key: str | None = None) -> None:
        genai.configure(api_key=api_key or settings.google_api_key)

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
        gemini_model_name = _MODEL_MAP.get(model_id, model_id)
        system_instruction, gemini_contents = _convert_messages(messages)

        gen_config = GenerationConfig(
            max_output_tokens=max_tokens or _DEFAULT_MAX_TOKENS,
            **({"temperature": temperature} if temperature is not None else {}),
        )

        model = genai.GenerativeModel(
            model_name=gemini_model_name,
            system_instruction=system_instruction or None,
            generation_config=gen_config,
        )

        resp = await model.generate_content_async(gemini_contents)

        text = resp.text if hasattr(resp, "text") else ""
        call_id = f"chatcmpl-{uuid.uuid4().hex}"

        usage = resp.usage_metadata if hasattr(resp, "usage_metadata") else None
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        return LLMResponse(
            id=call_id,
            model=gemini_model_name,
            content=text,
            role="assistant",
            finish_reason=_map_finish_reason(resp),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    async def call_stream(
        self,
        messages: list[dict[str, Any]],
        model_id: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        gemini_model_name = _MODEL_MAP.get(model_id, model_id)
        system_instruction, gemini_contents = _convert_messages(messages)
        call_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        gen_config = GenerationConfig(
            max_output_tokens=max_tokens or _DEFAULT_MAX_TOKENS,
            **({"temperature": temperature} if temperature is not None else {}),
        )

        model = genai.GenerativeModel(
            model_name=gemini_model_name,
            system_instruction=system_instruction or None,
            generation_config=gen_config,
        )

        # Initial role chunk
        role_chunk = {
            "id": call_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"

        async for chunk in await model.generate_content_async(
            gemini_contents, stream=True
        ):
            text = chunk.text if hasattr(chunk, "text") else ""
            if text:
                data = {
                    "id": call_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(data)}\n\n"

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
    """Convert OpenAI messages to Gemini content format."""
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            continue

        # Gemini uses "user" / "model" roles
        gemini_role = "model" if role == "assistant" else "user"

        if isinstance(content, str):
            contents.append({"role": gemini_role, "parts": [{"text": content}]})
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append({"text": block.get("text", "")})
                    # Image blocks could be added here
            contents.append({"role": gemini_role, "parts": parts})

    return "\n\n".join(system_parts), contents


def _map_finish_reason(resp: Any) -> str:
    try:
        reason = resp.candidates[0].finish_reason
        mapping = {1: "stop", 2: "length", 3: "content_filter"}
        return mapping.get(int(reason), "stop")
    except Exception:
        return "stop"
