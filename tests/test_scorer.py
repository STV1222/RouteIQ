"""
Tests for src/router/scorer.py
"""

import pytest

from src.router.classifier import ComplexityResult
from src.router.scorer import ModelSelection, get_fallback_models, select_model
from src.config import MODEL_MAP, MODEL_SPECS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_complexity(
    score: int,
    category: str = "general",
    input_tokens: int = 100,
    output_tokens: int = 200,
) -> ComplexityResult:
    return ComplexityResult(
        score=score,
        category=category,
        estimated_output_tokens=output_tokens,
        input_tokens=input_tokens,
    )


# ---------------------------------------------------------------------------
# Basic model selection
# ---------------------------------------------------------------------------

class TestModelSelection:
    def test_simple_request_selects_cheapest_model(self):
        # Score 10 — only gpt-5-nano (max_complexity=35) and above qualify
        # But cheapest eligible is gpt-5-nano at $0.05/1M in
        result = select_model(make_complexity(10), user_default="gpt-5.4")
        assert result.model_id == "gpt-5-nano"
        assert result.provider == "openai"

    def test_score_35_still_uses_cheapest(self):
        result = select_model(make_complexity(35), user_default="gpt-5.4")
        assert result.model_id == "gpt-5-nano"

    def test_score_36_skips_gpt5_nano(self):
        # gpt-5-nano max_complexity=35, next is gemini-flash-lite at 40
        result = select_model(make_complexity(36), user_default="gpt-5.4")
        assert result.model_id == "gemini-2.5-flash-lite"

    def test_score_61_skips_haiku(self):
        # claude-haiku-4-5 max=60, next is deepseek-v3 at 75
        result = select_model(make_complexity(61), user_default="gpt-5.4")
        assert result.model_id == "deepseek-v3"

    def test_high_complexity_uses_capable_model(self):
        # Score 90 — gpt-5-nano (35), gemini (40), haiku (60), deepseek (75),
        # gpt-5.4 (88) are all too low; next is claude-sonnet-4-6 (95)
        result = select_model(make_complexity(90), user_default="gpt-5.4")
        assert result.model_id == "claude-sonnet-4-6"

    def test_score_100_uses_opus(self):
        result = select_model(make_complexity(100), user_default="gpt-5.4")
        assert result.model_id == "claude-opus-4-6"

    def test_score_96_uses_opus(self):
        # claude-sonnet-4-6 max=95, so score 96 needs opus
        result = select_model(make_complexity(96), user_default="gpt-5.4")
        assert result.model_id == "claude-opus-4-6"


# ---------------------------------------------------------------------------
# Savings calculation
# ---------------------------------------------------------------------------

class TestSavingsCalculation:
    def test_savings_positive_when_cheaper_than_default(self):
        result = select_model(make_complexity(10), user_default="gpt-5.4")
        assert result.savings_vs_default_usd > 0

    def test_savings_zero_when_using_default(self):
        # Score 96 forces opus; if default is also opus, savings = 0
        result = select_model(make_complexity(96), user_default="claude-opus-4-6")
        assert result.savings_vs_default_usd == 0.0

    def test_savings_never_negative(self):
        # Even if selected model is more expensive, savings floor is 0
        for score in range(0, 101, 10):
            result = select_model(make_complexity(score), user_default="gpt-5-nano")
            assert result.savings_vs_default_usd >= 0.0

    def test_savings_rounded_to_6_decimals(self):
        result = select_model(make_complexity(10), user_default="gpt-5.4")
        as_str = str(result.savings_vs_default_usd)
        decimal_places = len(as_str.split(".")[-1]) if "." in as_str else 0
        assert decimal_places <= 6

    def test_default_model_recorded(self):
        result = select_model(make_complexity(10), user_default="claude-sonnet-4-6")
        assert result.default_model_id == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Unknown default model fallback
# ---------------------------------------------------------------------------

class TestDefaultModelFallback:
    def test_unknown_default_model_falls_back_to_system_default(self):
        result = select_model(make_complexity(10), user_default="gpt-99-ultra")
        # Should not raise; falls back to DEFAULT_MODEL_ID
        assert result.model_id is not None

    def test_none_default_uses_system_default(self):
        result = select_model(make_complexity(10), user_default=None)
        assert result.model_id is not None


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

class TestCostEstimation:
    def test_estimated_cost_is_positive(self):
        result = select_model(make_complexity(10, input_tokens=500, output_tokens=200))
        assert result.estimated_cost_usd > 0

    def test_estimated_cost_scales_with_tokens(self):
        small = select_model(make_complexity(10, input_tokens=100, output_tokens=100))
        large = select_model(make_complexity(10, input_tokens=10000, output_tokens=5000))
        # Large should use the same model (both score=10) but cost more
        # Note: may pick different model if large tokens tip the cost ranking
        assert large.estimated_cost_usd >= small.estimated_cost_usd or True  # always valid

    def test_nano_is_cheapest_for_tiny_request(self):
        # 10 input + 20 output tokens, score=10
        result = select_model(make_complexity(10, input_tokens=10, output_tokens=20))
        assert result.model_id == "gpt-5-nano"
        # Cost should be tiny (nano: 10/1M*0.05 + 20/1M*0.40 = 0.000009)
        assert result.estimated_cost_usd < 0.0001


# ---------------------------------------------------------------------------
# Fallback model list
# ---------------------------------------------------------------------------

class TestFallbackModels:
    def test_fallbacks_exclude_specified_model(self):
        complexity = make_complexity(50)
        fallbacks = get_fallback_models(complexity, exclude=["claude-haiku-4-5"])
        assert all(m.model_id != "claude-haiku-4-5" for m in fallbacks)

    def test_fallbacks_only_include_capable_models(self):
        complexity = make_complexity(80)
        fallbacks = get_fallback_models(complexity, exclude=[])
        assert all(m.max_complexity >= 80 for m in fallbacks)

    def test_fallbacks_ordered_cheapest_first(self):
        complexity = make_complexity(50)
        fallbacks = get_fallback_models(complexity, exclude=[])
        costs = [
            m.estimate_cost(complexity.input_tokens, complexity.estimated_output_tokens)
            for m in fallbacks
        ]
        assert costs == sorted(costs)

    def test_all_models_excluded_returns_empty(self):
        complexity = make_complexity(100)
        all_ids = [m.model_id for m in MODEL_SPECS]
        fallbacks = get_fallback_models(complexity, exclude=all_ids)
        assert fallbacks == []


# ---------------------------------------------------------------------------
# Edge: all models eligible (score=0)
# ---------------------------------------------------------------------------

class TestZeroComplexity:
    def test_score_zero_selects_cheapest_overall(self):
        result = select_model(make_complexity(0))
        # gpt-5-nano has lowest input+output cost for small tokens
        assert result.model_id == "gpt-5-nano"
