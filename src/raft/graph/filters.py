"""Filters operate directly on canonical cases, including live Pydantic outputs."""

from typing import Callable

from raft.cases import ExtractedCase

NeighborFilter = Callable[[ExtractedCase, ExtractedCase], bool]


def validate_filter(neighbor_filter: NeighborFilter | None) -> None:
    if neighbor_filter is not None and not callable(neighbor_filter):
        raise ValueError("neighbor_filter must be a Python callable or None")
