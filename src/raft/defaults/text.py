"""Default embedding text and agent-facing retrieval context."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .extraction import CaseExtraction

if TYPE_CHECKING:
    from raft.retrieval.types import RetrievalHit


def state_to_text(state: CaseExtraction) -> list[str]:
    """Embed each self-contained narrative directly, preserving timeline order."""
    return [entry.narrative for entry in state.timeline]


def case_to_text(state: CaseExtraction) -> str:
    """Prefer root cause and resolution; fall back to the final timeline entry."""
    text = "\n".join(
        part for part in [state.root_cause, state.resolution_steps] if part and part.strip()
    )
    if text:
        return text
    return state.timeline[-1].narrative if state.timeline else ""


def format_case(hit: RetrievalHit) -> str:
    """Serialize ID, metadata, complete extracted state, and the matched item index.

    Works with any Pydantic output model, including RootModel. Review assessments
    and execution diagnostics remain on the structured hit, not in this text.
    item_index is zero-based in the list returned by the embedding state_to_text.
    """
    payload = hit["case"].model_dump(mode="json", include={"id", "metadata", "output"})
    payload["item_index"] = hit["item_index"]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
