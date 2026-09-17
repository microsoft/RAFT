"""Three independent text views for the schema in custom_extraction.py.

state_to_text: timeline embedding/BM25 anchors.
case_to_text: optional case-graph linking text, not retrieval display.
format_case: agent-facing text, counted against the retrieval context budget.
Import these examples from the repository root after installing RAFT.
"""

import json

from examples.custom_extraction import SupportCase
from raft.retrieval.types import RetrievalHit


def state_to_text(state: SupportCase) -> list[str]:
    """Keep one self-contained narrative per entry, in unchanged timeline order.

    Returned positions become hit['item_index']. Do not filter/reorder entries
    without also changing format_case's index mapping. Avoid attaching the final
    cause or resolution to early entries: each anchor should reflect that stage.
    """
    return [entry.narrative for entry in state.timeline]


def case_to_text(state: SupportCase) -> str:
    """Link cases by exact error codes plus confirmed cause/resolution."""
    codes = ", ".join(entity.name for entity in state.entities if entity.kind == "error_code")
    conclusions = "\n".join(
        text for text in (state.root_cause, state.resolution_steps) if text is not None
    )
    if not conclusions and state.timeline:
        conclusions = state.timeline[-1].narrative
    return "\n".join(part for part in (f"Error codes: {codes}" if codes else "", conclusions) if part)


def format_case(hit: RetrievalHit) -> str:
    """Return the full trajectory and its matched index, without internal history.

    The hit also exposes metadata, scores, and entry_id; choose what your agent
    needs. Here metadata, review, execution, and handoff notes are omitted.
    """
    case = hit["case"]
    state = case.output
    if not isinstance(state, SupportCase):
        raise TypeError("This formatter requires SupportCase outputs")
    index = hit["item_index"]
    if not 0 <= index < len(state.timeline):
        raise ValueError("Matched item_index does not identify a timeline entry")
    payload = {
        "id": case.id,
        "item_index": index,
        "entities": [entity.model_dump(mode="json") for entity in state.entities],
        "timeline": [entry.narrative for entry in state.timeline],
        "root_cause": state.root_cause,
        "resolution_steps": state.resolution_steps,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
