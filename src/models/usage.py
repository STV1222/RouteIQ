"""
Pydantic model for usage log records stored in DynamoDB.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class UsageRecord(BaseModel):
    # DynamoDB primary key
    api_key: str                        # pk
    timestamp_request_id: str          # sk  →  "2024-01-15T10:30:00Z#<uuid>"

    # Routing metadata
    model_used: str
    provider: str
    complexity_score: int
    complexity_category: str

    # Token counts (actual, from provider response)
    input_tokens: int
    output_tokens: int
    total_tokens: int

    # Cost / savings (USD, 6 decimal places)
    actual_cost_usd: float
    savings_usd: float
    estimated_cost_usd: float

    # Request metadata
    cache_hit: bool = False
    latency_ms: int = 0
    stream: bool = False
    default_model: str = ""
    user_id: Optional[str] = None

    model_config = {"extra": "allow"}

    def to_dynamo_item(self) -> dict:
        """Serialise to a flat dict of DynamoDB attribute values (strings/numbers)."""
        return {
            "pk": self.api_key,
            "sk": self.timestamp_request_id,
            "model_used": self.model_used,
            "provider": self.provider,
            "complexity_score": self.complexity_score,
            "complexity_category": self.complexity_category,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "actual_cost_usd": str(round(self.actual_cost_usd, 6)),
            "savings_usd": str(round(self.savings_usd, 6)),
            "estimated_cost_usd": str(round(self.estimated_cost_usd, 6)),
            "cache_hit": self.cache_hit,
            "latency_ms": self.latency_ms,
            "stream": self.stream,
            "default_model": self.default_model,
            **({"user_id": self.user_id} if self.user_id else {}),
        }


class ApiKeyRecord(BaseModel):
    api_key: str                        # pk
    user_id: str
    monthly_budget_usd: float = 0.0    # 0 = unlimited
    spend_this_month_usd: float = 0.0
    created_at: str = ""
    is_active: bool = True
    default_model: Optional[str] = None

    model_config = {"extra": "allow"}
