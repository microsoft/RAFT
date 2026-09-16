from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from raft.runtime import RetryDecision


@dataclass
class EmbeddingBatch:
    """One response's vectors and provider-named, additive token counters.

    Usage belongs to this response, never mutable state on a shared backend.
    Empty usage means unavailable; it does not imply a free request.
    """

    vectors: list[list[float]]
    usage: dict[str, int] = field(default_factory=dict)


class EmbeddingBackend(Protocol):
    """A provider returns one vector per input text, in exactly the same order."""

    name: str
    model: str

    async def embed(
        self, texts: list[str], *, input_type: Literal["document", "query"] = "document"
    ) -> EmbeddingBatch | list[list[float]]:
        """Return vectors with usage, or plain vectors when usage is unavailable."""
        ...

    def classify_error(self, exc: Exception) -> RetryDecision: ...
