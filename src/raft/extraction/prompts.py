from __future__ import annotations

from typing import Any

from raft._json import _to_json
from raft._text_budget import budget_summary


def _case_prompt(
    *,
    case_id: Any,
    metadata: dict[str, Any],
    current_state: Any,
    coverage: dict[str, Any],
    batch: dict[str, Any],
    target_schema: dict[str, Any],
    batch_budget: dict[str, Any],
    pass_number: int,
    validation_error: str | None,
) -> str:
    """Supply changing pass data without injecting a second instruction prompt."""
    payload = {
        "id": case_id,
        "pass_number": pass_number,
        "metadata": metadata,
        "target_output_schema": target_schema,
        "current_state": current_state,
        "coverage": coverage,
        "batch": batch,
        "batch_budget": budget_summary(batch_budget),
        "validation_error": validation_error,
    }
    return "Pass context:\n" + _to_json(payload)


def _review_prompt(
    *,
    case_id: Any,
    metadata: dict[str, Any],
    output: Any,
    target_schema: dict[str, Any],
    worker_final_revision: int,
    coverage: dict[str, Any],
) -> str:
    return "Review context:\n" + _to_json({
        "id": case_id,
        "metadata": metadata,
        "output": output,
        "target_output_schema": target_schema,
        "worker_final_revision": worker_final_revision,
        "coverage": coverage,
    })
