"""Ordered, lossless batches of artifact JSON, with a retryable source cursor."""

from dataclasses import dataclass
from typing import Any

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


def next_batch(context: CaseContext, position: int, offset: int, limit: int) -> ArtifactBatch:
    """Pack whole artifacts into a batch; never split source JSON.

    limit counts source JSON characters, not prompt wrappers or JSON escaping.
    This internal read is independent of the per-query result character limit.
    """
    if offset:
        raise ValueError("Batch cursors must start at an artifact boundary")
    items = []
    used = 0
    total = len(context.artifact_char_counts)
    while position < total and used < limit:
        size = context.artifact_char_counts[position]
        if items and size > limit - used:
            break
        original_position, content = context.connection.execute(
            "SELECT original_position, artifact_json FROM artifacts WHERE position = ?",
            (position,),
        ).fetchone()
        if size > limit:
            raise ArtifactTooLargeError(position, original_position, size, limit)
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
        position += 1
    return ArtifactBatch(items, used, position, 0, position == total)
