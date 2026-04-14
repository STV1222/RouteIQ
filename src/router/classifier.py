"""
Request complexity classifier — pure Python, zero external calls, <5ms target.

Classifies an incoming messages array into a complexity score (0-100) and
a task category used downstream by the scorer to select the cheapest capable model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComplexityResult:
    score: int                      # 0-100
    category: str                   # e.g. "coding", "reasoning", "general"
    estimated_output_tokens: int    # rough estimate used for cost projection
    input_tokens: int               # counted from the messages


# ---------------------------------------------------------------------------
# Keyword → (category, base_score) table
# Evaluated in order; first match wins.
# ---------------------------------------------------------------------------

_CATEGORY_RULES: list[tuple[list[str], str, int]] = [
    # (keywords, category, base_score)
    (
        ["agent", "tool use", "tool_use", "function call", "plan and execute",
         "execute a plan", "use tools", "autonomous"],
        "agentic", 85,
    ),
    (
        ["reason step by step", "think step by step", "chain of thought",
         "analyze", "analyse", "explain why", "compare and contrast",
         "evaluate the trade", "pros and cons", "critically assess"],
        "reasoning", 80,
    ),
    (
        ["write a function", "implement", "debug", "fix the bug", "refactor",
         "write code", "write a class", "write a script", "write a program",
         "code snippet", "python ", "javascript ", "typescript ", "golang ",
         "java ", "c++ ", "rust ", "sql query", "regex", "algorithm",
         "data structure", "unit test", "write tests"],
        "coding", 70,
    ),
    (
        ["summarize", "summarise", "tldr", "tl;dr", "brief summary",
         "give me a summary", "key points", "main points", "overview of"],
        "summarization", 20,
    ),
    (
        ["translate", "translation", "in french", "in spanish", "in german",
         "in japanese", "in chinese", "in arabic", "in portuguese"],
        "translation", 15,
    ),
    (
        ["classify", "categorize", "categorise", "sentiment", "label this",
         "is this positive", "is this negative", "tag this"],
        "classification", 25,
    ),
]

# Fallback
_DEFAULT_CATEGORY = "general"
_DEFAULT_BASE_SCORE = 40

# Output token estimates by category (rough heuristics)
_OUTPUT_TOKEN_ESTIMATES: dict[str, int] = {
    "agentic":        800,
    "reasoning":      600,
    "coding":         500,
    "summarization":  200,
    "translation":    300,
    "classification":  50,
    "general":        250,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(messages: list[dict[str, Any]]) -> ComplexityResult:
    """
    Classify a messages array without making any external calls.

    Args:
        messages: OpenAI-style message list, e.g.
                  [{"role": "user", "content": "..."}, ...]

    Returns:
        ComplexityResult with score, category, token estimates.
    """
    text = _extract_text(messages)
    input_tokens = _count_tokens(text)
    lower = text.lower()

    category, base_score = _match_category(lower)

    # Boost for long inputs (more context → harder task)
    score = base_score
    if input_tokens > 2_000:
        score += 10
    if input_tokens > 6_000:
        score += 15

    score = min(score, 100)

    estimated_output = _OUTPUT_TOKEN_ESTIMATES.get(category, 250)
    # Scale output estimate for very long inputs (likely longer replies expected)
    if input_tokens > 4_000:
        estimated_output = int(estimated_output * 1.5)

    return ComplexityResult(
        score=score,
        category=category,
        estimated_output_tokens=estimated_output,
        input_tokens=input_tokens,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_text(messages: list[dict[str, Any]]) -> str:
    """Concatenate all message content into a single string for analysis."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # Vision / multimodal messages: pull text blocks only
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return " ".join(parts)


def _count_tokens(text: str) -> int:
    """Rough token count: characters / 4 (GPT-family rule of thumb)."""
    return max(1, len(text) // 4)


def _match_category(lower_text: str) -> tuple[str, int]:
    """Return (category, base_score) for the first matching rule."""
    for keywords, category, score in _CATEGORY_RULES:
        for kw in keywords:
            # Use word-boundary-aware search to avoid false positives
            if _keyword_present(kw, lower_text):
                return category, score
    return _DEFAULT_CATEGORY, _DEFAULT_BASE_SCORE


def _keyword_present(keyword: str, text: str) -> bool:
    """
    Check whether *keyword* appears as a meaningful phrase in *text*.
    Single-word keywords use word boundaries; multi-word phrases use substring match.
    """
    if " " in keyword:
        return keyword in text
    # Wrap single words in word-boundary assertion
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))
