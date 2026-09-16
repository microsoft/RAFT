"""Exact hybrid neighbors over ID-linked case embeddings; no case payload copies."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from raft._json import _id_key
from raft.cases import ExtractedCase, restore_case
from raft.embedding import BM25Index
from raft.progress import CaseProgress

from .filters import NeighborFilter, validate_filter


def link_cases(
    cases: Sequence[ExtractedCase],
    embeddings: Sequence[dict[str, Any]],
    *,
    top_k: int = 10,
    neighbor_filter: NeighborFilter | None = None,
    rrf_constant: int = 60,
    bm25_index: BM25Index | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Filter original cases, rank ID-aligned vectors/texts, then weight union edges.

    Embeddings and BM25 documents must follow case order. SNN uses Jaccard of
    directed top-k sets, excluding self; every selected edge is retained.
    """
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(rrf_constant, int) or rrf_constant < 1:
        raise ValueError("rrf_constant must be a positive integer")
    validate_filter(neighbor_filter)
    cases = [restore_case(case) for case in cases]
    keys = [_id_key(case.id) for case in cases]
    if len(set(keys)) != len(keys):
        raise ValueError("Case IDs must be unique")
    if keys != [_id_key(row["id"]) for row in embeddings]:
        raise ValueError("Case embeddings must match case IDs and order")
    index = bm25_index if bm25_index is not None else BM25Index.from_records(embeddings)
    if [(_id_key(d["id"]), d["text"]) for d in index.documents] != [
        (_id_key(r["id"]), r["text"]) for r in embeddings
    ]:
        raise ValueError("BM25 corpus must match case embeddings, texts, and order")
    if cases:
        vectors = np.asarray([row["embedding"] for row in embeddings], dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise ValueError("Case vectors must have consistent dimensions and finite values")
        scales = np.max(np.abs(vectors), axis=1, keepdims=True)
        if np.any(scales == 0):
            raise ValueError("Case vectors must be nonzero for cosine similarity")
        vectors = vectors / scales
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        if len({(r.get("provider"), r.get("model")) for r in embeddings}) != 1:
            raise ValueError("Case vectors must use the same embedding provider and model")

    neighbors, sets = [], []
    with CaseProgress(len(cases), enabled=show_progress, desc="Graph neighbors") as progress:
        for i, case in enumerate(cases):
            eligible = [
                j
                for j, candidate in enumerate(cases)
                if i != j and (neighbor_filter is None or neighbor_filter(case, candidate))
            ]
            best, ranked = [], []
            if eligible:
                dense = dict(zip(eligible, np.clip(vectors[eligible] @ vectors[i], -1, 1)))
                lexical = index.scores(embeddings[i]["text"])
                dense_order = sorted(eligible, key=lambda j: (-dense[j], keys[j]))
                lexical_order = sorted(
                    (j for j in eligible if lexical[j] > 0), key=lambda j: (-lexical[j], keys[j])
                )
                fused = {j: 1 / (rrf_constant + rank) for rank, j in enumerate(dense_order, 1)}
                for rank, j in enumerate(lexical_order, 1):
                    fused[j] += 1 / (rrf_constant + rank)
                best = sorted(eligible, key=lambda j: (-fused[j], keys[j]))[:top_k]
                ranked = [
                    {
                        "id": cases[j].id,
                        "rank": rank,
                        "rrf_score": fused[j],
                        "cosine_similarity": float(dense[j]),
                        "bm25_score": lexical[j],
                    }
                    for rank, j in enumerate(best, 1)
                ]
            neighbors.append({"id": case.id, "neighbors": ranked})
            sets.append({keys[j] for j in best})

            progress.advance()

    by_key = dict(zip(keys, cases, strict=True))
    neighbor_sets = dict(zip(keys, sets, strict=True))
    pairs = {
        tuple(sorted((source, target)))
        for source, targets in neighbor_sets.items()
        for target in targets
    }
    edges = []
    for source, target in sorted(pairs):
        left, right = neighbor_sets[source], neighbor_sets[target]
        shared, union = len(left & right), len(left | right)
        edges.append(
            {
                "source": by_key[source].id,
                "target": by_key[target].id,
                "weight": shared / union if union else 0.0,
                "shared_neighbors": shared,
                "neighbor_union": union,
                "mutual": target in left and source in right,
            }
        )
    return {
        "neighbors": neighbors,
        "edges": edges,
        "summary": {
            "nodes": len(cases),
            "directed_links": sum(map(len, sets)),
            "edges": len(edges),
        },
        "settings": {
            "top_k": top_k,
            "rrf_constant": rrf_constant,
            "neighbor_filter": "python_callable" if neighbor_filter else None,
            "edge_rule": "union",
            "weight": "snn_jaccard_directed_neighbors",
        },
    }
