"""Exact entry ranking and promotion to distinct parent cases."""

from __future__ import annotations

import inspect
from typing import Sequence

import numpy as np

from raft._json import _id_key

from .types import CaseFormatter, RetrievalHit


def normalize(vectors) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or not vectors.shape[1] or not np.isfinite(vectors).all():
        raise ValueError("Vectors must have consistent dimensions and finite values")
    scales = np.max(np.abs(vectors), axis=1, keepdims=True)
    if np.any(scales == 0):
        raise ValueError("Vectors must be nonzero for cosine similarity")
    vectors = vectors / scales
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def rank_cases(
    rows, vectors, query_vector, eligible, cases, lexical, rrf_constant
) -> list[RetrievalHit]:
    """Rank every eligible entry before deduplication; no entry-level top-k cutoff."""
    dense = dict(zip(eligible, np.clip(vectors[eligible] @ query_vector, -1, 1), strict=True))

    def tie(i):
        return _id_key(rows[i]["id"])

    dense_order = sorted(eligible, key=lambda i: (-dense[i], tie(i)))
    if lexical is None:
        scores = dense
    else:
        scores = {i: 1 / (rrf_constant + rank) for rank, i in enumerate(dense_order, 1)}
        lexical_order = sorted(
            (i for i in eligible if lexical[i] > 0), key=lambda i: (-lexical[i], tie(i))
        )
        for rank, i in enumerate(lexical_order, 1):
            scores[i] += 1 / (rrf_constant + rank)
    ranked = sorted(eligible, key=lambda i: (-scores[i], tie(i)))
    seen = set()
    hits: list[RetrievalHit] = []
    for i in ranked:
        row = rows[i]
        key = _id_key(row["case_id"])
        if key in seen:
            continue
        seen.add(key)
        hits.append(
            {
                "id": row["case_id"],
                "case": cases[key],
                "item_index": row["item_index"],
                "entry_id": row["id"],
                "score": float(scores[i]),
                "cosine_similarity": float(dense[i]),
                "bm25_score": None if lexical is None else float(lexical[i]),
                "source": "direct",
            }
        )
    return hits


def cap_cases(
    hits: Sequence[RetrievalHit], max_chars: int | None, format_case: CaseFormatter
) -> tuple[list[RetrievalHit], str, bool]:
    """Format a ranked prefix once, counting the exact text plus blank-line separators."""
    selected: list[RetrievalHit] = []
    parts: list[str] = []
    used = 0
    for hit in hits:
        text = format_case(hit)
        if not isinstance(text, str):
            if inspect.iscoroutine(text):
                text.close()
            raise TypeError("format_case must return a string synchronously")
        size = len(text) + (2 if parts else 0)
        if max_chars is not None and used + size > max_chars:
            break
        selected.append(hit)
        parts.append(text)
        used += size
    return selected, "\n\n".join(parts), len(selected) < len(hits)
