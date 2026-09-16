"""Per-case one-hop graph expansion; independent of retrieval and storage."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from raft._json import _id_key

CaseID = str | int
Adjacency = dict[CaseID, dict[CaseID, float]]


def build_adjacency(edges: Iterable[dict[str, Any]]) -> Adjacency:
    """Prepare undirected SNN edges once for repeated expand_neighbors calls.

    Keys are original case IDs; integer 1 and string "1" remain distinct.
    Duplicate edges keep their maximum weight; self-links are ignored.
    """
    adjacency: Adjacency = {}
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if type(source) not in (str, int) or type(target) not in (str, int):
            raise ValueError("Graph case IDs must be strings or integers")
        weight = float(edge["weight"])
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Graph SNN weights must be between 0 and 1")
        if source == target:
            continue
        for left, right in ((source, target), (target, source)):
            neighbors = adjacency.setdefault(left, {})
            neighbors[right] = max(weight, neighbors.get(right, 0))
    return adjacency


def expand_neighbors(
    seed_ids: Sequence[CaseID],
    adjacency: Adjacency,
    *,
    per_case_top_k: int = 10,
    allowed_ids: Iterable[CaseID] | None = None,
) -> list[dict[str, Any]]:
    """Return each seed's own top neighbors, preserving shared relationships.

    Exclude all seed IDs and apply allowed_ids before each seed's selection.
    Each group has seed_id and neighbors (id/weight records), ordered by weight
    then stable neighbor ID. Shared neighbors appear in every selecting group
    with that seed's edge weight; there is no cross-seed merge or overall cap.
    Groups follow seed input order, ignoring duplicate seeds. Unknown seeds and
    seeds with no eligible neighbors have empty groups; zero-weight edges remain
    eligible. Empty input returns no groups.
    Uses a mapping from build_adjacency; makes no retrieval or model calls.
    """
    if type(per_case_top_k) is not int or per_case_top_k < 1:
        raise ValueError("per_case_top_k must be a positive integer")
    if isinstance(seed_ids, (str, bytes)) or any(type(id) not in (str, int) for id in seed_ids):
        raise ValueError("seed_ids must be a sequence of string or integer IDs")
    seeds = list(dict.fromkeys(seed_ids))
    seed_set = set(seeds)
    allowed = None if allowed_ids is None else set(allowed_ids)
    groups = []
    for seed in seeds:
        eligible = (
            (id, weight)
            for id, weight in adjacency.get(seed, {}).items()
            if id not in seed_set and (allowed is None or id in allowed)
        )
        selected = sorted(eligible, key=lambda item: (-item[1], _id_key(item[0])))[
            :per_case_top_k
        ]
        groups.append(
            {
                "seed_id": seed,
                "neighbors": [{"id": id, "weight": weight} for id, weight in selected],
            }
        )
    return groups
