"""
Pydantic models for incoming RouteIQ / OpenAI-compatible requests.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

class TextContentPart(BaseModel):
    type: str = "text"
    text: str


class ImageUrlDetail(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class ImageContentPart(BaseModel):
    type: str = "image_url"
    image_url: ImageUrlDetail


ContentPart = Union[TextContentPart, ImageContentPart]


class Message(BaseModel):
    role: str                                               # system | user | assistant | tool
    content: Union[str, list[ContentPart], None] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Function / tool definitions (pass-through to provider)
# ---------------------------------------------------------------------------

class FunctionDef(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


class ToolDef(BaseModel):
    type: str = "function"
    function: FunctionDef


# ---------------------------------------------------------------------------
# Main request schema — mirrors OpenAI /v1/chat/completions
# ---------------------------------------------------------------------------

class RouteIQRequest(BaseModel):
    # Core fields
    messages: list[Message] = Field(..., min_length=1)
    model: Optional[str] = None             # client's preferred / default model
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    response_format: Optional[dict[str, Any]] = None
    tools: Optional[list[ToolDef]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None

    # RouteIQ-specific overrides (optional, ignored if not provided)
    routeiq_force_model: Optional[str] = Field(
        default=None,
        description="Force RouteIQ to use this exact model, bypassing classifier.",
    )
    routeiq_min_complexity: Optional[int] = Field(
        default=None,
        ge=0, le=100,
        description="Artificially raise the complexity floor for this request.",
    )

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def _validate_messages_content(self) -> "RouteIQRequest":
        for msg in self.messages:
            if msg.content is None and msg.tool_calls is None:
                raise ValueError(
                    f"Message with role '{msg.role}' must have content or tool_calls."
                )
        return self

    def to_openai_dict(self) -> dict[str, Any]:
        """Serialise to a dict suitable for passing to the OpenAI SDK."""
        data: dict[str, Any] = {
            "messages": [m.model_dump(exclude_none=True) for m in self.messages],
            "stream": self.stream,
        }
        optional_fields = [
            "max_tokens", "temperature", "top_p", "n", "stop",
            "presence_penalty", "frequency_penalty", "logit_bias",
            "user", "seed", "response_format", "tools", "tool_choice",
        ]
        for f in optional_fields:
            v = getattr(self, f, None)
            if v is not None:
                data[f] = v if not isinstance(v, BaseModel) else v.model_dump(exclude_none=True)
        if self.tools:
            data["tools"] = [t.model_dump(exclude_none=True) for t in self.tools]
        return data
