"""
Redis cache — prompt hashing + get/set with silent failure semantics.

Cache failures MUST NOT crash the request pipeline; they are logged to
stderr and the caller falls through to a live LLM call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from typing import Any, Optional

import redis.asyncio as aioredis

from src.config import settings

logger = logging.getLogger(__name__)

# Module-level client — created on first use
_client: Optional[aioredis.Redis] = None


def _get_client() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
    return _client


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def make_cache_key(
    messages: list[dict[str, Any]],
    model_preference: str | None = None,
) -> str:
    """
    Deterministic SHA-256 key for a (messages, model_preference) pair.
    sort_keys=True ensures order-independent hashing of dict fields.
    """
    payload = json.dumps(messages, sort_keys=True) + (model_preference or "")
    return "riq:" + hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Public API — all failures are silenced
# ---------------------------------------------------------------------------

async def get(cache_key: str) -> Optional[dict[str, Any]]:
    """Return the cached response dict, or None on miss / error."""
    if settings.skip_cache:
        return None
    try:
        client = _get_client()
        raw = await client.get(cache_key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        print(f"[RouteIQ] Redis GET error (silenced): {exc}", file=sys.stderr)
        return None


async def set(
    cache_key: str,
    response: dict[str, Any],
    ttl: int | None = None,
) -> None:
    """Store response in Redis. Silently skips on error."""
    if settings.skip_cache:
        return
    try:
        client = _get_client()
        effective_ttl = ttl if ttl is not None else settings.redis_ttl_seconds
        await client.set(cache_key, json.dumps(response), ex=effective_ttl)
    except Exception as exc:
        print(f"[RouteIQ] Redis SET error (silenced): {exc}", file=sys.stderr)


async def delete(cache_key: str) -> None:
    """Delete a cache entry. Silently skips on error."""
    try:
        client = _get_client()
        await client.delete(cache_key)
    except Exception as exc:
        print(f"[RouteIQ] Redis DELETE error (silenced): {exc}", file=sys.stderr)


async def ping() -> bool:
    """Health check — returns True if Redis is reachable."""
    try:
        client = _get_client()
        return await client.ping()
    except Exception:
        return False
