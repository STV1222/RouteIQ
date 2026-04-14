"""
Pydantic models for RouteIQ responses.

RouteIQ wraps the upstream LLM response in an OpenAI-compatible envelope
and adds savings / routing metadata.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# OpenAI-compatible sub-models
# ---------------------------------------------------------------------------

class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class ChoiceMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    function_call: Optional[FunctionCall] = None

    model_config = {"extra": "allow"}


class Choice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None

    model_config = {"extra": "allow"}


class UsageStats(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Streaming delta models
# ---------------------------------------------------------------------------

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None

    model_config = {"extra": "allow"}


class StreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Main response schema
# ---------------------------------------------------------------------------

class RouteIQResponse(BaseModel):
    # OpenAI-compatible core
    id: str
    object: str = "chat.completion"
    created: int
    model: str                          # the model RouteIQ actually used
    choices: list[Choice]
    usage: Optional[UsageStats] = None
    system_fingerprint: Optional[str] = None

    # RouteIQ metadata — included in response body AND headers
    routeiq_model_used: str = Field(description="Model selected by RouteIQ.")
    routeiq_provider: str = Field(description="Provider of the selected model.")
    routeiq_savings_usd: float = Field(description="USD saved vs. default model.")
    routeiq_complexity_score: int = Field(description="Complexity score 0-100.")
    routeiq_complexity_category: str = Field(description="Task category detected.")
    routeiq_cache_hit: bool = Field(default=False)
    routeiq_default_model: str = Field(description="Model savings are compared against.")
    routeiq_estimated_cost_usd: float = Field(description="Estimated cost for this request.")

    model_config = {"extra": "allow"}

    def to_openai_dict(self) -> dict[str, Any]:
        """Return an OpenAI-compatible dict (routeiq_* fields excluded)."""
        d = self.model_dump(exclude_none=True)
        routeiq_keys = [k for k in d if k.startswith("routeiq_")]
        for k in routeiq_keys:
            del d[k]
        return d

    def response_headers(self) -> dict[str, str]:
        return {
            "X-RouteIQ-Model": self.routeiq_model_used,
            "X-RouteIQ-Provider": self.routeiq_provider,
            "X-RouteIQ-Savings": f"{self.routeiq_savings_usd:.6f}",
            "X-RouteIQ-Complexity": str(self.routeiq_complexity_score),
            "X-RouteIQ-Category": self.routeiq_complexity_category,
            "X-RouteIQ-Cache-Hit": str(self.routeiq_cache_hit).lower(),
            "X-RouteIQ-Estimated-Cost": f"{self.routeiq_estimated_cost_usd:.6f}",
        }
