"""
Tests for src/router/classifier.py

All tests are pure-Python — no external calls, no fixtures needed.
"""

import pytest

from src.router.classifier import ComplexityResult, classify, _count_tokens, _extract_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def msg(text: str, role: str = "user") -> dict:
    return {"role": role, "content": text}


def msgs(*texts: str) -> list[dict]:
    return [msg(t) for t in texts]


# ---------------------------------------------------------------------------
# Category detection
# ---------------------------------------------------------------------------

class TestCategoryDetection:
    def test_summarization_keyword_summarize(self):
        result = classify(msgs("Please summarize this document for me."))
        assert result.category == "summarization"
        assert result.score == 20

    def test_summarization_keyword_tldr(self):
        result = classify(msgs("Give me a tldr of this article."))
        assert result.category == "summarization"

    def test_summarization_keyword_brief(self):
        result = classify(msgs("Write a brief summary of the quarterly report."))
        assert result.category == "summarization"

    def test_translation(self):
        result = classify(msgs("Translate the following text to French: hello world"))
        assert result.category == "translation"
        assert result.score == 15

    def test_translation_in_french(self):
        result = classify(msgs("Say this in french: good morning"))
        assert result.category == "translation"

    def test_classification_sentiment(self):
        result = classify(msgs("Classify the sentiment of this review as positive or negative."))
        assert result.category == "classification"
        assert result.score == 25

    def test_classification_categorize(self):
        result = classify(msgs("Categorize these support tickets by priority."))
        assert result.category == "classification"

    def test_coding_function(self):
        result = classify(msgs("Write a function in Python that calculates the Fibonacci sequence."))
        assert result.category == "coding"
        assert result.score == 70

    def test_coding_implement(self):
        result = classify(msgs("Implement a binary search tree with insert and delete operations."))
        assert result.category == "coding"

    def test_coding_debug(self):
        result = classify(msgs("Debug this Python code: def foo(): return None"))
        assert result.category == "coding"

    def test_reasoning_step_by_step(self):
        result = classify(msgs("Think step by step about how to solve this problem."))
        assert result.category == "reasoning"
        assert result.score == 80

    def test_reasoning_analyze(self):
        result = classify(msgs("Analyze the trade-offs between SQL and NoSQL databases."))
        assert result.category == "reasoning"

    def test_reasoning_explain_why(self):
        result = classify(msgs("Explain why the sky is blue."))
        assert result.category == "reasoning"

    def test_agentic_agent(self):
        result = classify(msgs("You are an autonomous agent. Plan a marketing campaign."))
        assert result.category == "agentic"
        assert result.score == 85

    def test_agentic_tool_use(self):
        result = classify(msgs("Use tools to gather data and execute the following tasks."))
        assert result.category == "agentic"

    def test_general_fallback(self):
        result = classify(msgs("Hello, how are you doing today?"))
        assert result.category == "general"
        assert result.score == 40


# ---------------------------------------------------------------------------
# Score range validation
# ---------------------------------------------------------------------------

class TestScoreRanges:
    def test_score_never_below_zero(self):
        result = classify(msgs("hi"))
        assert result.score >= 0

    def test_score_never_above_100(self):
        long_text = "analyze " + ("x " * 10000)
        result = classify(msgs(long_text))
        assert result.score <= 100

    def test_score_capped_at_100_for_agentic_long(self):
        # agentic base=85, >2000 tokens +10, >6000 tokens +15 → 110, capped at 100
        # 10000 words * 2 chars = 20000 chars → 5000 tokens (just under 6000)
        # Use enough text to cross the 6000 token threshold (6000*4=24000 chars)
        long_text = "agent " + ("a " * 12000)
        result = classify(msgs(long_text))
        assert result.score == 100

    def test_token_boost_over_2000(self):
        # 2001 tokens ≈ 8004 chars
        medium_text = "summarize " + ("word " * 2000)
        result = classify(msgs(medium_text))
        assert result.score == 30  # 20 (summarization) + 10 (>2000 tokens)

    def test_token_boost_over_6000(self):
        # 6001 tokens ≈ 24004 chars
        long_text = "summarize " + ("word " * 6000)
        result = classify(msgs(long_text))
        assert result.score == 45  # 20 + 10 + 15


# ---------------------------------------------------------------------------
# Output token estimates
# ---------------------------------------------------------------------------

class TestOutputTokenEstimates:
    def test_coding_has_higher_estimate_than_classification(self):
        coding = classify(msgs("Implement a sorting algorithm."))
        classification = classify(msgs("Classify this text as spam or ham."))
        assert coding.estimated_output_tokens > classification.estimated_output_tokens

    def test_agentic_has_highest_estimate(self):
        agentic = classify(msgs("You are an agent. Plan and execute the task."))
        general = classify(msgs("Tell me about Paris."))
        assert agentic.estimated_output_tokens > general.estimated_output_tokens

    def test_long_input_scales_output_estimate(self):
        short = classify(msgs("Summarize this."))
        long_text = "summarize " + ("word " * 4100)
        long = classify(msgs(long_text))
        assert long.estimated_output_tokens > short.estimated_output_tokens


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

class TestTokenCounting:
    def test_count_tokens_basic(self):
        assert _count_tokens("hello world") == 2   # 11 chars / 4 = 2

    def test_count_tokens_minimum_one(self):
        assert _count_tokens("") == 1

    def test_input_tokens_reported(self):
        text = "x " * 400   # 800 chars → 200 tokens
        result = classify(msgs(text))
        assert result.input_tokens == 200

    def test_multipart_messages_concatenated(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Summarize this article."},
        ]
        result = classify(messages)
        assert result.category == "summarization"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_content_string(self):
        result = classify([{"role": "user", "content": ""}])
        assert result.category == "general"
        assert result.score == 40

    def test_multimodal_message_text_blocks(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this image."},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ]
        result = classify(messages)
        assert result.category == "summarization"

    def test_multiple_messages_highest_category_wins(self):
        # The agentic keyword appears in the second message
        messages = [
            {"role": "system", "content": "You are an autonomous agent."},
            {"role": "user", "content": "Plan and execute the task."},
        ]
        result = classify(messages)
        assert result.category == "agentic"

    def test_case_insensitive_matching(self):
        result = classify(msgs("SUMMARIZE THIS DOCUMENT"))
        assert result.category == "summarization"
