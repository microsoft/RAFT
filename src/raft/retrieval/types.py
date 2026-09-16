"""Public types for a ranked case match and its text formatter."""

from typing import Callable, Literal, TypedDict

from raft.cases import ExtractedCase


class RetrievalHit(TypedDict):
    """A distinct case anchored at its highest-ranked embedded item."""

    id: str | int
    case: ExtractedCase
    item_index: int
    entry_id: str | int
    score: float
    cosine_similarity: float
    bm25_score: float | None
    source: Literal["direct"]


CaseFormatter = Callable[[RetrievalHit], str]
