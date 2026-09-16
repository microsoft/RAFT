"""Validation and measurement for caller-owned character or token budget dicts."""

import inspect
from typing import Any


def validate_budget(
    budget: dict[str, Any], name: str, *, allow_zero: bool = False
) -> dict[str, Any]:
    if not isinstance(budget, dict):
        raise ValueError(f"{name} must be a budget dict")
    if budget.get("unit") not in ("chars", "tokens"):
        raise ValueError(f"{name}.unit must be 'chars' or 'tokens'")
    limit = budget.get("limit")
    if type(limit) is not int or limit < (0 if allow_zero else 1):
        minimum = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name}.limit must be a {minimum} integer")
    allowed = {"unit", "limit"}
    if budget["unit"] == "tokens":
        allowed.add("count_tokens")
        counter = budget.get("count_tokens")
        if (
            not callable(counter)
            or inspect.iscoroutinefunction(counter)
            or inspect.isasyncgenfunction(counter)
            or inspect.iscoroutinefunction(getattr(counter, "__call__", None))
            or inspect.isasyncgenfunction(getattr(counter, "__call__", None))
        ):
            raise ValueError(f"{name}.count_tokens must be a synchronous callable")
    if budget.keys() - allowed:
        raise ValueError(f"{name} contains unsupported keys: {sorted(budget.keys() - allowed)}")
    return dict(budget)


def measure_text(text: str, budget: dict[str, Any]) -> int:
    if budget["unit"] == "chars":
        return len(text)
    count = budget["count_tokens"](text)
    if type(count) is not int or count < 0:
        if inspect.iscoroutine(count):
            count.close()
        raise TypeError("count_tokens must return a nonnegative integer")
    return count


def budget_summary(budget: dict[str, Any]) -> dict[str, Any]:
    return {"unit": budget["unit"], "limit": budget["limit"]}
