"""
Per-key spend tracking and budget enforcement.

Called by the gateway before forwarding a request.  If the key has a
monthly budget and is over it, raises HTTP 429.
"""

from __future__ import annotations

from fastapi import HTTPException, status

from src.models.usage import ApiKeyRecord


def check_budget(record: ApiKeyRecord) -> None:
    """
    Raise HTTP 429 if the key has exceeded its monthly budget.
    A budget of 0.0 means unlimited.
    """
    if record.monthly_budget_usd <= 0:
        return  # unlimited

    if record.spend_this_month_usd >= record.monthly_budget_usd:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Monthly budget of ${record.monthly_budget_usd:.2f} exceeded. "
                f"Current spend: ${record.spend_this_month_usd:.2f}. "
                "Reset occurs on the 1st of each month."
            ),
            headers={"Retry-After": "86400"},
        )


def budget_remaining(record: ApiKeyRecord) -> float:
    """Return USD remaining in the budget, or -1.0 for unlimited."""
    if record.monthly_budget_usd <= 0:
        return -1.0
    return max(0.0, record.monthly_budget_usd - record.spend_this_month_usd)
