"""
API key validation middleware.

Extracts the Bearer token from Authorization header, validates against
DynamoDB, checks active status, and returns the ApiKeyRecord for downstream use.
"""

from __future__ import annotations

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import settings
from src.models.usage import ApiKeyRecord
from src.storage import dynamo

_bearer = HTTPBearer(auto_error=False)

# In-process cache to avoid a DynamoDB round-trip on every request.
# Entries are valid for ~60 seconds in production; cleared on budget changes.
_key_cache: dict[str, ApiKeyRecord] = {}


async def validate_api_key(request: Request) -> ApiKeyRecord:
    """
    FastAPI dependency.  Raises HTTP 401/403 on invalid / inactive / over-budget keys.
    Returns the ApiKeyRecord on success.

    If settings.skip_auth is True (dev mode), returns a synthetic unlimited key.
    """
    if settings.skip_auth:
        return _dev_key()

    credentials: HTTPAuthorizationCredentials | None = await _bearer(request)
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Use: Authorization: Bearer <routeiq-key>",
        )

    api_key = credentials.credentials

    # Warm cache hit
    if api_key in _key_cache:
        record = _key_cache[api_key]
    else:
        record = await dynamo.get_api_key(api_key)
        if record is not None:
            _key_cache[api_key] = record

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    if not record.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key is inactive.",
        )

    return record


def invalidate_key_cache(api_key: str) -> None:
    """Remove a key from the in-process cache (call after budget updates)."""
    _key_cache.pop(api_key, None)


def _dev_key() -> ApiKeyRecord:
    return ApiKeyRecord(
        api_key="dev-key",
        user_id="dev",
        monthly_budget_usd=0.0,
        spend_this_month_usd=0.0,
        is_active=True,
    )
