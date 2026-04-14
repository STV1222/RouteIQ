"""
RouteIQ configuration — model cost table, environment settings, and constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Model descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    provider: str                   # "openai" | "anthropic" | "deepseek" | "gemini"
    input_cost_per_1m: float        # USD per 1 million input tokens
    output_cost_per_1m: float       # USD per 1 million output tokens
    max_complexity: int             # 0-100; requests scored above this are NOT sent here
    context_window: int             # max input tokens
    display_name: str = ""

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        cost = (input_tokens / 1_000_000 * self.input_cost_per_1m) + (
            output_tokens / 1_000_000 * self.output_cost_per_1m
        )
        return round(cost, 6)


# ---------------------------------------------------------------------------
# Model cost table — ordered cheapest → most expensive
# ---------------------------------------------------------------------------

MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        model_id="gpt-5-nano",
        provider="openai",
        input_cost_per_1m=0.05,
        output_cost_per_1m=0.40,
        max_complexity=35,
        context_window=128_000,
        display_name="GPT-5 Nano",
    ),
    ModelSpec(
        model_id="gemini-2.5-flash-lite",
        provider="gemini",
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.40,
        max_complexity=40,
        context_window=1_000_000,
        display_name="Gemini 2.5 Flash-Lite",
    ),
    ModelSpec(
        model_id="claude-haiku-4-5",
        provider="anthropic",
        input_cost_per_1m=1.00,
        output_cost_per_1m=5.00,
        max_complexity=60,
        context_window=200_000,
        display_name="Claude Haiku 4.5",
    ),
    ModelSpec(
        model_id="deepseek-v3",
        provider="deepseek",
        input_cost_per_1m=0.28,
        output_cost_per_1m=0.42,
        max_complexity=75,
        context_window=64_000,
        display_name="DeepSeek V3.2",
    ),
    ModelSpec(
        model_id="gpt-5.4",
        provider="openai",
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
        max_complexity=88,
        context_window=128_000,
        display_name="GPT-5.4",
    ),
    ModelSpec(
        model_id="claude-sonnet-4-6",
        provider="anthropic",
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        max_complexity=95,
        context_window=200_000,
        display_name="Claude Sonnet 4.6",
    ),
    ModelSpec(
        model_id="claude-opus-4-6",
        provider="anthropic",
        input_cost_per_1m=5.00,
        output_cost_per_1m=25.00,
        max_complexity=100,
        context_window=200_000,
        display_name="Claude Opus 4.6",
    ),
]

# Fast lookup by model_id
MODEL_MAP: dict[str, ModelSpec] = {m.model_id: m for m in MODEL_SPECS}

# Default model used for savings comparison when client doesn't specify one
DEFAULT_MODEL_ID: str = _env("DEFAULT_MODEL", "gpt-5.4")


# ---------------------------------------------------------------------------
# Settings singleton
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    # LLM provider keys
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    deepseek_api_key: str = field(default_factory=lambda: _env("DEEPSEEK_API_KEY"))
    google_api_key: str = field(default_factory=lambda: _env("GOOGLE_API_KEY"))

    # OpenRouter — single key covering all providers
    # Set USE_OPENROUTER=true to route everything through OpenRouter instead of
    # individual provider APIs.  Only OPENROUTER_API_KEY is required in that mode.
    openrouter_api_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY"))
    use_openrouter: bool = field(
        default_factory=lambda: _env("USE_OPENROUTER", "false").lower() == "true"
    )

    # Infrastructure
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379"))
    redis_ttl_seconds: int = field(default_factory=lambda: _env_int("REDIS_TTL_SECONDS", 300))
    dynamodb_table_usage: str = field(default_factory=lambda: _env("DYNAMODB_TABLE_USAGE", "routeiq_usage"))
    dynamodb_table_keys: str = field(default_factory=lambda: _env("DYNAMODB_TABLE_KEYS", "routeiq_api_keys"))
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))
    dynamodb_endpoint_url: Optional[str] = field(
        default_factory=lambda: _env("DYNAMODB_ENDPOINT_URL") or None
    )

    # App behaviour
    default_model: str = field(default_factory=lambda: _env("DEFAULT_MODEL", "gpt-5.4"))
    routeiq_env: str = field(default_factory=lambda: _env("ROUTEIQ_ENV", "development"))
    max_fallback_attempts: int = field(default_factory=lambda: _env_int("MAX_FALLBACK_ATTEMPTS", 3))

    # Dev / bypass flags
    skip_auth: bool = field(default_factory=lambda: _env("SKIP_AUTH", "false").lower() == "true")
    skip_cache: bool = field(default_factory=lambda: _env("SKIP_CACHE", "false").lower() == "true")


# Module-level singleton — import this everywhere
settings = Settings()
