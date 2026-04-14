"""
Model selection scoring engine.

Given a ComplexityResult, selects the cheapest model whose max_complexity
is >= the request's complexity score, then computes savings vs. the
user's default (or the system default).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import MODEL_MAP, MODEL_SPECS, DEFAULT_MODEL_ID, ModelSpec
from src.router.classifier import ComplexityResult


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSelection:
    model_id: str
    provider: str
    display_name: str
    estimated_cost_usd: float       # projected cost for this request
    savings_vs_default_usd: float   # how much cheaper vs. user's default model
    default_model_id: str           # the model we compared against


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_model(
    complexity: ComplexityResult,
    user_default: str | None = None,
) -> ModelSelection:
    """
    Select the cheapest model that can handle *complexity*.

    Args:
        complexity: Result from classifier.classify().
        user_default: The model_id the user would have used without RouteIQ.
                      Falls back to DEFAULT_MODEL_ID from config.

    Returns:
        ModelSelection with chosen model and savings metadata.

    Raises:
        RuntimeError: If no model in the cost table can handle the complexity
                      (should never happen — claude-opus-4-6 has max_complexity=100).
    """
    default_id = _resolve_default(user_default)
    candidates = _eligible_models(complexity.score)

    if not candidates:
        raise RuntimeError(
            f"No model can handle complexity score {complexity.score}. "
            "This should never happen — check the MODEL_SPECS table."
        )

    # Pick the candidate with the lowest estimated cost
    selected = min(candidates, key=lambda m: m.estimate_cost(
        complexity.input_tokens, complexity.estimated_output_tokens
    ))

    selected_cost = selected.estimate_cost(
        complexity.input_tokens, complexity.estimated_output_tokens
    )
    default_cost = _cost_for_model(default_id, complexity)
    savings = round(max(0.0, default_cost - selected_cost), 6)

    return ModelSelection(
        model_id=selected.model_id,
        provider=selected.provider,
        display_name=selected.display_name,
        estimated_cost_usd=round(selected_cost, 6),
        savings_vs_default_usd=savings,
        default_model_id=default_id,
    )


def get_fallback_models(
    complexity: ComplexityResult,
    exclude: list[str],
) -> list[ModelSpec]:
    """
    Return eligible models ordered cheapest-first, excluding the given model_ids.
    Used by the forwarder to retry on provider failure.
    """
    candidates = _eligible_models(complexity.score)
    filtered = [m for m in candidates if m.model_id not in exclude]
    return sorted(
        filtered,
        key=lambda m: m.estimate_cost(
            complexity.input_tokens, complexity.estimated_output_tokens
        ),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_default(user_default: str | None) -> str:
    """Return a valid model_id for savings comparison."""
    if user_default and user_default in MODEL_MAP:
        return user_default
    return DEFAULT_MODEL_ID


def _eligible_models(complexity_score: int) -> list[ModelSpec]:
    """All models whose max_complexity >= complexity_score, cheapest first."""
    return [m for m in MODEL_SPECS if m.max_complexity >= complexity_score]


def _cost_for_model(model_id: str, complexity: ComplexityResult) -> float:
    """Estimate cost for a specific model, falling back to most expensive if unknown."""
    spec = MODEL_MAP.get(model_id)
    if spec is None:
        # Unknown model — use the most expensive as a conservative baseline
        spec = MODEL_SPECS[-1]
    return spec.estimate_cost(complexity.input_tokens, complexity.estimated_output_tokens)
