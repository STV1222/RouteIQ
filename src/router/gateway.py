"""
Gateway — main orchestration layer for every RouteIQ request.

Flow:
  validate API key → cache lookup → classify → score → forward → cache set
  → log usage → return RouteIQResponse
"""

from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator, Optional

from src.cache import redis_cache
from src.config import MODEL_MAP, settings
from src.models.request import RouteIQRequest
from src.models.response import Choice, ChoiceMessage, RouteIQResponse, UsageStats
from src.models.usage import UsageRecord
from src.router import classifier, forwarder
from src.router.scorer import ModelSelection, select_model
from src.storage import dynamo


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def route_request(
    request: RouteIQRequest,
    api_key: str,
    api_key_record: Any = None,          # pre-fetched ApiKeyRecord (from auth middleware)
) -> RouteIQResponse:
    """
    Non-streaming path.  Returns a fully resolved RouteIQResponse.
    """
    request_id = uuid.uuid4().hex
    start_ms = _now_ms()

    # 1. Resolve user's default model (key record > request field > system default)
    user_default = _resolve_default(request, api_key_record)

    # 2. Cache lookup
    messages_dicts = [m.model_dump(exclude_none=True) for m in request.messages]
    cache_key = redis_cache.make_cache_key(messages_dicts, user_default)
    cached = await redis_cache.get(cache_key)
    if cached is not None:
        resp = RouteIQResponse(**cached)
        resp = resp.model_copy(update={"routeiq_cache_hit": True})
        # Still log the cache hit to DynamoDB
        await _log(
            api_key=api_key,
            request_id=request_id,
            response=resp,
            complexity_score=resp.routeiq_complexity_score,
            category=resp.routeiq_complexity_category,
            cache_hit=True,
            stream=False,
            latency_ms=_now_ms() - start_ms,
            user_default=user_default,
            api_key_record=api_key_record,
        )
        return resp

    # 3. Classify
    complexity = classifier.classify(messages_dicts)

    # Allow caller to force a higher complexity floor
    if request.routeiq_min_complexity is not None:
        from src.router.classifier import ComplexityResult
        effective_score = max(complexity.score, request.routeiq_min_complexity)
        complexity = ComplexityResult(
            score=effective_score,
            category=complexity.category,
            estimated_output_tokens=complexity.estimated_output_tokens,
            input_tokens=complexity.input_tokens,
        )

    # 4. Score / select model
    if request.routeiq_force_model and request.routeiq_force_model in MODEL_MAP:
        spec = MODEL_MAP[request.routeiq_force_model]
        selection = ModelSelection(
            model_id=spec.model_id,
            provider=spec.provider,
            display_name=spec.display_name,
            estimated_cost_usd=spec.estimate_cost(
                complexity.input_tokens, complexity.estimated_output_tokens
            ),
            savings_vs_default_usd=0.0,
            default_model_id=user_default,
        )
    else:
        selection = select_model(complexity, user_default=user_default)

    # 5. Forward to provider
    extra = _build_extra_kwargs(request)
    llm_response = await forwarder.forward(
        messages=messages_dicts,
        selection=selection,
        complexity=complexity,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=False,
        extra_kwargs=extra,
    )

    # 6. Build RouteIQResponse
    resp = _build_response(
        llm_response=llm_response,
        selection=selection,
        complexity=complexity,
        cache_hit=False,
    )

    # 7. Cache the response
    await redis_cache.set(cache_key, resp.model_dump())

    # 8. Log usage
    latency = _now_ms() - start_ms
    await _log(
        api_key=api_key,
        request_id=request_id,
        response=resp,
        complexity_score=complexity.score,
        category=complexity.category,
        cache_hit=False,
        stream=False,
        latency_ms=latency,
        user_default=user_default,
        api_key_record=api_key_record,
        actual_tokens=llm_response.total_tokens,
        input_tokens=llm_response.input_tokens,
        output_tokens=llm_response.output_tokens,
    )

    return resp


async def route_request_stream(
    request: RouteIQRequest,
    api_key: str,
    api_key_record: Any = None,
) -> AsyncIterator[str]:
    """
    Streaming path.  Yields SSE lines and then logs usage.
    Note: cache is checked but not populated for streaming responses
    (the full body isn't available until the stream is exhausted).
    """
    request_id = uuid.uuid4().hex
    start_ms = _now_ms()

    user_default = _resolve_default(request, api_key_record)
    messages_dicts = [m.model_dump(exclude_none=True) for m in request.messages]

    # Cache check for streaming (rare hit, but worth it)
    cache_key = redis_cache.make_cache_key(messages_dicts, user_default)
    cached = await redis_cache.get(cache_key)
    if cached is not None:
        resp = RouteIQResponse(**cached)
        # Re-stream cached content as SSE
        import json, time as _time
        chunk = {
            "id": resp.id,
            "object": "chat.completion.chunk",
            "created": resp.created,
            "model": resp.model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": resp.choices[0].message.content or ""},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        done = chunk.copy()
        done["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"
        return

    complexity = classifier.classify(messages_dicts)
    selection = select_model(complexity, user_default=user_default)
    extra = _build_extra_kwargs(request)

    async for chunk in forwarder.forward_stream(
        messages=messages_dicts,
        selection=selection,
        complexity=complexity,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        extra_kwargs=extra,
    ):
        yield chunk

    # Log best-effort usage after stream completes (no actual token counts available)
    latency = _now_ms() - start_ms
    await _log_stream_usage(
        api_key=api_key,
        request_id=request_id,
        selection=selection,
        complexity=complexity,
        latency_ms=latency,
        user_default=user_default,
        api_key_record=api_key_record,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_default(request: RouteIQRequest, api_key_record: Any) -> str:
    if request.model:
        return request.model
    if api_key_record and getattr(api_key_record, "default_model", None):
        return api_key_record.default_model
    return settings.default_model


def _build_extra_kwargs(request: RouteIQRequest) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if request.tools:
        extra["tools"] = [t.model_dump(exclude_none=True) for t in request.tools]
    if request.tool_choice is not None:
        extra["tool_choice"] = request.tool_choice
    if request.top_p is not None:
        extra["top_p"] = request.top_p
    if request.stop is not None:
        extra["stop"] = request.stop
    if request.presence_penalty is not None:
        extra["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None:
        extra["frequency_penalty"] = request.frequency_penalty
    if request.seed is not None:
        extra["seed"] = request.seed
    if request.response_format is not None:
        extra["response_format"] = request.response_format
    return extra


def _build_response(
    llm_response: Any,
    selection: ModelSelection,
    complexity: Any,
    cache_hit: bool,
) -> RouteIQResponse:
    import time as _time
    from src.providers.base import LLMResponse

    raw = llm_response.raw or {}
    resp_id = llm_response.id or f"chatcmpl-{uuid.uuid4().hex}"

    tool_calls = None
    if llm_response.tool_calls:
        from src.models.response import ToolCall, FunctionCall
        tool_calls = [
            ToolCall(
                id=tc["id"],
                type=tc.get("type", "function"),
                function=FunctionCall(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in llm_response.tool_calls
        ]

    choice_msg = ChoiceMessage(
        role=llm_response.role,
        content=llm_response.content or None,
        tool_calls=tool_calls,
    )
    choice = Choice(
        index=0,
        message=choice_msg,
        finish_reason=llm_response.finish_reason,
    )

    actual_cost = MODEL_MAP[selection.model_id].estimate_cost(
        llm_response.input_tokens or complexity.input_tokens,
        llm_response.output_tokens or complexity.estimated_output_tokens,
    ) if selection.model_id in MODEL_MAP else selection.estimated_cost_usd

    return RouteIQResponse(
        id=resp_id,
        object="chat.completion",
        created=raw.get("created") or int(_time.time()),
        model=selection.model_id,
        choices=[choice],
        usage=UsageStats(
            prompt_tokens=llm_response.input_tokens,
            completion_tokens=llm_response.output_tokens,
            total_tokens=llm_response.total_tokens,
        ) if llm_response.total_tokens else None,
        routeiq_model_used=selection.model_id,
        routeiq_provider=selection.provider,
        routeiq_savings_usd=selection.savings_vs_default_usd,
        routeiq_complexity_score=complexity.score,
        routeiq_complexity_category=complexity.category,
        routeiq_cache_hit=cache_hit,
        routeiq_default_model=selection.default_model_id,
        routeiq_estimated_cost_usd=round(actual_cost, 6),
    )


async def _log(
    api_key: str,
    request_id: str,
    response: RouteIQResponse,
    complexity_score: int,
    category: str,
    cache_hit: bool,
    stream: bool,
    latency_ms: int,
    user_default: str,
    api_key_record: Any,
    actual_tokens: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    usage = response.usage
    record = UsageRecord(
        api_key=api_key,
        timestamp_request_id=dynamo.make_sk(request_id),
        model_used=response.routeiq_model_used,
        provider=response.routeiq_provider,
        complexity_score=complexity_score,
        complexity_category=category,
        input_tokens=input_tokens or (usage.prompt_tokens if usage else 0),
        output_tokens=output_tokens or (usage.completion_tokens if usage else 0),
        total_tokens=actual_tokens or (usage.total_tokens if usage else 0),
        actual_cost_usd=response.routeiq_estimated_cost_usd,
        savings_usd=response.routeiq_savings_usd,
        estimated_cost_usd=response.routeiq_estimated_cost_usd,
        cache_hit=cache_hit,
        latency_ms=latency_ms,
        stream=stream,
        default_model=user_default,
        user_id=getattr(api_key_record, "user_id", None),
    )
    await dynamo.log_usage(record)
    if not cache_hit:
        await dynamo.increment_spend(api_key, response.routeiq_estimated_cost_usd)


async def _log_stream_usage(
    api_key: str,
    request_id: str,
    selection: ModelSelection,
    complexity: Any,
    latency_ms: int,
    user_default: str,
    api_key_record: Any,
) -> None:
    record = UsageRecord(
        api_key=api_key,
        timestamp_request_id=dynamo.make_sk(request_id),
        model_used=selection.model_id,
        provider=selection.provider,
        complexity_score=complexity.score,
        complexity_category=complexity.category,
        input_tokens=complexity.input_tokens,
        output_tokens=complexity.estimated_output_tokens,
        total_tokens=complexity.input_tokens + complexity.estimated_output_tokens,
        actual_cost_usd=selection.estimated_cost_usd,
        savings_usd=selection.savings_vs_default_usd,
        estimated_cost_usd=selection.estimated_cost_usd,
        cache_hit=False,
        latency_ms=latency_ms,
        stream=True,
        default_model=user_default,
        user_id=getattr(api_key_record, "user_id", None),
    )
    await dynamo.log_usage(record)
    await dynamo.increment_spend(api_key, selection.estimated_cost_usd)


def _now_ms() -> int:
    return int(time.monotonic() * 1000)
