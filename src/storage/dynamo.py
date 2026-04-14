"""
DynamoDB storage layer — usage logs and API key management.

Write failures are SILENT (logged to stderr, never raised).
Reads raise on critical errors (e.g. key validation) so callers can 401.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from src.config import settings
from src.models.usage import ApiKeyRecord, UsageRecord

# ---------------------------------------------------------------------------
# DynamoDB resource (lazy init)
# ---------------------------------------------------------------------------

_dynamodb: Any = None


def _get_dynamodb() -> Any:
    global _dynamodb
    if _dynamodb is None:
        kwargs: dict[str, Any] = {"region_name": settings.aws_region}
        if settings.dynamodb_endpoint_url:
            # DynamoDB Local doesn't validate credentials but boto3 still requires them
            kwargs["endpoint_url"] = settings.dynamodb_endpoint_url
            kwargs["aws_access_key_id"] = "local"
            kwargs["aws_secret_access_key"] = "local"
        _dynamodb = boto3.resource("dynamodb", **kwargs)
    return _dynamodb


def _usage_table():
    return _get_dynamodb().Table(settings.dynamodb_table_usage)


def _keys_table():
    return _get_dynamodb().Table(settings.dynamodb_table_keys)


# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------

async def get_api_key(api_key: str) -> Optional[ApiKeyRecord]:
    """
    Fetch an API key record from DynamoDB.
    Returns None if the key does not exist.
    Raises on DynamoDB connectivity errors so auth can fail closed.
    """
    resp = _keys_table().get_item(Key={"pk": api_key})
    item = resp.get("Item")
    if item is None:
        return None
    return ApiKeyRecord(
        api_key=item["pk"],
        user_id=item.get("user_id", ""),
        monthly_budget_usd=float(item.get("monthly_budget_usd", 0)),
        spend_this_month_usd=float(item.get("spend_this_month_usd", 0)),
        created_at=item.get("created_at", ""),
        is_active=bool(item.get("is_active", True)),
        default_model=item.get("default_model"),
    )


async def increment_spend(api_key: str, amount_usd: float) -> None:
    """Atomically add *amount_usd* to spend_this_month_usd. Silent on error."""
    try:
        _keys_table().update_item(
            Key={"pk": api_key},
            UpdateExpression="ADD spend_this_month_usd :amt",
            ExpressionAttributeValues={":amt": _decimal(amount_usd)},
        )
    except Exception as exc:
        print(f"[RouteIQ] DynamoDB increment_spend error (silenced): {exc}", file=sys.stderr)


def _decimal(value: float):
    """Convert float to Decimal for DynamoDB."""
    from decimal import Decimal
    return Decimal(str(round(value, 6)))


# ---------------------------------------------------------------------------
# Usage logging
# ---------------------------------------------------------------------------

async def log_usage(record: UsageRecord) -> None:
    """Write a usage record to DynamoDB. Silent on error."""
    try:
        item = record.to_dynamo_item()
        # DynamoDB requires Decimal for numeric types
        from decimal import Decimal
        numeric_keys = [
            "complexity_score", "input_tokens", "output_tokens",
            "total_tokens", "latency_ms",
        ]
        str_numeric_keys = ["actual_cost_usd", "savings_usd", "estimated_cost_usd"]
        for k in numeric_keys:
            if k in item:
                item[k] = int(item[k])
        for k in str_numeric_keys:
            if k in item:
                item[k] = Decimal(item[k])
        _usage_table().put_item(Item=item)
    except Exception as exc:
        print(f"[RouteIQ] DynamoDB log_usage error (silenced): {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Usage queries
# ---------------------------------------------------------------------------

async def get_usage_stats(api_key: str, limit: int = 100) -> list[dict[str, Any]]:
    """Return recent usage records for an API key."""
    resp = _usage_table().query(
        KeyConditionExpression=Key("pk").eq(api_key),
        ScanIndexForward=False,
        Limit=limit,
    )
    return resp.get("Items", [])


async def get_savings_summary(api_key: str) -> dict[str, Any]:
    """Aggregate savings and cost from usage records."""
    records = await get_usage_stats(api_key, limit=1000)
    total_savings = sum(float(r.get("savings_usd", 0)) for r in records)
    total_cost = sum(float(r.get("actual_cost_usd", 0)) for r in records)
    total_requests = len(records)
    cache_hits = sum(1 for r in records if r.get("cache_hit"))

    by_model: dict[str, dict[str, Any]] = {}
    for r in records:
        model = r.get("model_used", "unknown")
        if model not in by_model:
            by_model[model] = {"requests": 0, "cost_usd": 0.0, "savings_usd": 0.0}
        by_model[model]["requests"] += 1
        by_model[model]["cost_usd"] += float(r.get("actual_cost_usd", 0))
        by_model[model]["savings_usd"] += float(r.get("savings_usd", 0))

    return {
        "total_requests": total_requests,
        "total_cost_usd": round(total_cost, 6),
        "total_savings_usd": round(total_savings, 6),
        "cache_hits": cache_hits,
        "by_model": by_model,
    }


# ---------------------------------------------------------------------------
# Dev helper — create tables locally (DynamoDB Local)
# ---------------------------------------------------------------------------

def create_tables_local() -> None:
    """Create DynamoDB tables against a local endpoint (for dev/testing)."""
    db = _get_dynamodb()

    for table_name, key_schema, attr_defs in [
        (
            settings.dynamodb_table_usage,
            [{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}],
            [{"AttributeName": "pk", "AttributeType": "S"}, {"AttributeName": "sk", "AttributeType": "S"}],
        ),
        (
            settings.dynamodb_table_keys,
            [{"AttributeName": "pk", "KeyType": "HASH"}],
            [{"AttributeName": "pk", "AttributeType": "S"}],
        ),
    ]:
        try:
            db.create_table(
                TableName=table_name,
                KeySchema=key_schema,
                AttributeDefinitions=attr_defs,
                BillingMode="PAY_PER_REQUEST",
            )
            print(f"[RouteIQ] Created table: {table_name}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceInUseException":
                print(f"[RouteIQ] Table already exists: {table_name}")
            else:
                raise


def make_sk(request_id: str | None = None) -> str:
    """Generate a sort key: ISO timestamp + request UUID."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    rid = request_id or uuid.uuid4().hex
    return f"{ts}#{rid}"
