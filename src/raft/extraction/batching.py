"""Ordered, lossless batches of artifact JSON, with a retryable source cursor."""

from dataclasses import dataclass
from typing import Any

from raft._text_budget import measure_text

from .context import ArtifactTooLargeError, CaseContext


@dataclass
class ArtifactBatch:
    items: list[dict[str, Any]]
    source_chars: int
    next_position: int
    next_offset: int
    is_last: bool

    def payload(self) -> dict[str, Any]:
        return {"items": self.items, "source_chars": self.source_chars, "is_last": self.is_last}


def next_batch(
    context: CaseContext, position: int, offset: int, budget: dict[str, Any]
) -> ArtifactBatch:
    """Pack whole artifacts into a batch; never split source JSON.

    The budget counts concatenated source JSON, not prompt wrappers or escaping.
    Token counts are measured on that combined string, not added per artifact.
    This internal read is independent of the per-query result budget.
    """
    if offset:
        raise ValueError("Batch cursors must start at an artifact boundary")
    items = []
    used = 0
    source = ""
    total = len(context.artifact_char_counts)
    while position < total:
        size = context.artifact_char_counts[position]
        original_position, content = context.connection.execute(
            "SELECT original_position, artifact_json FROM artifacts WHERE position = ?",
            (position,),
        ).fetchone()
        candidate_source = source + content if budget["unit"] == "tokens" else ""
        measured = (
            measure_text(candidate_source, budget) if budget["unit"] == "tokens" else used + size
        )
        if measured > budget["limit"]:
            if items:
                break
            raise ArtifactTooLargeError(position, original_position, measured, budget)
        items.append(
            {
                "position": position,
                "original_position": original_position,
                "start_char": 0,
                "end_char_exclusive": size,
                "total_chars": size,
                "artifact_json": content,
            }
        )
        used += size
        source = candidate_source
        position += 1
    return ArtifactBatch(items, used, position, 0, position == total)
