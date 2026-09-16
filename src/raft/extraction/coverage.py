"""Coverage of source content supplied in successfully committed worker passes."""

from typing import Any

from .context import CaseContext


def _coverage_payload(context: CaseContext, position: int, offset: int) -> dict[str, Any]:
    total = len(context.artifact_char_counts)
    return {
        "basis": "committed_preloaded_batches",
        "complete": position == total,
        "total_artifacts": total,
        "covered_count": position,
        "remaining_count": total - position,
        "covered_ranges": [{"start_position": 0, "end_position_exclusive": position}]
        if position
        else [],
        "uncovered_ranges": [{"start_position": position, "end_position_exclusive": total}]
        if position < total
        else [],
        "partial_artifact": {
            "position": position,
            "committed_chars": offset,
            "total_chars": context.artifact_char_counts[position],
        }
        if offset
        else None,
    }
