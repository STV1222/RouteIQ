"""
FastAPI route definitions — OpenAI-compatible endpoints + RouteIQ meta-endpoints.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from src.api.auth import validate_api_key
from src.api.budget import budget_remaining, check_budget
from src.cache import redis_cache
from src.models.request import RouteIQRequest
from src.models.usage import ApiKeyRecord
from src.router import gateway
from src.storage import dynamo

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /v1/chat/completions  — OpenAI-compatible drop-in
# ---------------------------------------------------------------------------

@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: RouteIQRequest,
    key_record: ApiKeyRecord = Depends(validate_api_key),
) -> Any:
    # Budget guard
    check_budget(key_record)

    api_key = key_record.api_key

    if body.stream:
        # Streaming response
        async def sse_generator():
            async for chunk in gateway.route_request_stream(
                request=body,
                api_key=api_key,
                api_key_record=key_record,
            ):
                yield chunk

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming
    resp = await gateway.route_request(
        request=body,
        api_key=api_key,
        api_key_record=key_record,
    )

    return JSONResponse(
        content=resp.model_dump(exclude_none=True),
        headers=resp.response_headers(),
    )


# ---------------------------------------------------------------------------
# GET /v1/usage
# ---------------------------------------------------------------------------

@router.get("/v1/usage")
async def get_usage(
    key_record: ApiKeyRecord = Depends(validate_api_key),
    limit: int = 50,
) -> Any:
    records = await dynamo.get_usage_stats(key_record.api_key, limit=limit)
    return {
        "api_key": key_record.api_key[:8] + "...",
        "records": records,
        "count": len(records),
        "monthly_budget_usd": key_record.monthly_budget_usd,
        "spend_this_month_usd": key_record.spend_this_month_usd,
        "budget_remaining_usd": budget_remaining(key_record),
    }


# ---------------------------------------------------------------------------
# GET /v1/savings
# ---------------------------------------------------------------------------

@router.get("/v1/savings")
async def get_savings(
    key_record: ApiKeyRecord = Depends(validate_api_key),
) -> Any:
    summary = await dynamo.get_savings_summary(key_record.api_key)
    return {
        "api_key": key_record.api_key[:8] + "...",
        **summary,
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health")
async def health_check() -> Any:
    redis_ok = await redis_cache.ping()
    return {
        "status": "ok",
        "redis": "ok" if redis_ok else "unavailable",
        "version": "1.0.0",
    }


# ---------------------------------------------------------------------------
# GET /v1/models  (bonus — lists available RouteIQ models)
# ---------------------------------------------------------------------------

@router.get("/v1/models")
async def list_models(
    _: ApiKeyRecord = Depends(validate_api_key),
) -> Any:
    from src.config import MODEL_SPECS
    return {
        "object": "list",
        "data": [
            {
                "id": m.model_id,
                "object": "model",
                "provider": m.provider,
                "display_name": m.display_name,
                "max_complexity": m.max_complexity,
                "input_cost_per_1m_usd": m.input_cost_per_1m,
                "output_cost_per_1m_usd": m.output_cost_per_1m,
            }
            for m in MODEL_SPECS
        ],
    }
