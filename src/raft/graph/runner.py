"""Case-level text/vector indexes and graph, referencing canonical case records."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Sequence, TypeVar

from pydantic import BaseModel

from raft._json import _id_key
from raft.cases import ExtractedCase, restore_case
from raft.embedding import BM25Index, EmbeddingBackend
from raft.embedding._batching import embed_text_batches, validate_batch_size
from raft.progress import CaseProgress
from raft.runtime import RetryDecision, _failed_case, validate_limits

from .filters import NeighborFilter
from .neighbors import link_cases

StateT = TypeVar("StateT", bound=BaseModel)


async def build_case_graph(
    *,
    cases: Sequence[ExtractedCase[StateT] | dict[str, Any]],
    backend: EmbeddingBackend,
    case_to_text: Callable[[StateT], str],
    output_type: type[StateT] | None = None,
    top_k: int = 10,
    neighbor_filter: NeighborFilter | None = None,
    rrf_constant: int = 60,
    batch_size: int = 64,
    concurrency: int = 4,
    timeout: float = 120.0,
    retries: int = 1,
    rpm: int = 60,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Reuse case/model objects; keep derived embeddings separate and ID-linked.

    Prepare one summary per case, then batch summaries across cases. Concurrency,
    timeout, retries, and RPM apply to embedding requests. Graph work runs in a
    thread, outside the embedding timeout. Only successfully embedded cases become
    nodes. embedding_usage and embedding_requests account for actual requests;
    shared-response usage is not attributed to individual cases.
    Returns in-memory results only; use raft.storage.save_graph to save a snapshot.
    """
    link_cases([], [], top_k=top_k, rrf_constant=rrf_constant, neighbor_filter=neighbor_filter)
    validate_limits(concurrency, timeout, retries, rpm)
    validate_batch_size(batch_size)
    if output_type is not None and (
        not isinstance(output_type, type) or not issubclass(output_type, BaseModel)
    ):
        raise ValueError("output_type must be a Pydantic model class")
    prepared = []
    failures = {}
    completed = {}
    seen = set()

    def failure(position, case_id, case, exc, decision, failure_type, *,
                attempts=0, requests=0, elapsed=0.0):
        record = _failed_case(
            case_id=case_id,
            case=case,
            category=decision.category,
            error=exc,
            retryable=decision.retryable,
            attempts=attempts,
            elapsed=elapsed,
            failure_type=failure_type,
        )
        record.update(requests=requests)
        failures[position] = record

    with CaseProgress(len(cases), enabled=show_progress, desc="Graph embeddings") as progress:
        for position, case in enumerate(cases):
            started = time.monotonic()
            case_id = (
                case.id if isinstance(case, ExtractedCase)
                else case.get("id") if isinstance(case, dict) else None
            )
            key = _id_key(case_id)
            duplicate = key in seen
            seen.add(key)
            try:
                if case_id is None or duplicate:
                    raise ValueError("Each case must have a non-null, unique id")
                case = restore_case(case, output_type)
            except (TypeError, ValueError) as exc:
                failure(
                    position, case_id, case, exc, RetryDecision(False, "invalid_case"),
                    "invalid_case", elapsed=time.monotonic() - started,
                )
                progress.advance("failed")
                continue
            try:
                text = case_to_text(case.output)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("case_to_text must return one nonempty string")
            except Exception as exc:
                failure(
                    position, case_id, case, exc, RetryDecision(False, "case_to_text_error"),
                    "terminal_error", elapsed=time.monotonic() - started,
                )
                progress.advance("failed")
                continue
            prepared.append((position, case, text))

        def complete(index, item):
            position, case, text = prepared[index]
            if item["error"] is not None:
                decision = item["decision"]
                failure(
                    position, case.id, case, item["error"], decision,
                    "retry_exhausted" if decision.retryable else "terminal_error",
                    attempts=item["attempts"], requests=item["requests"],
                    elapsed=item["elapsed_seconds"],
                )
                progress.advance("failed")
                return
            vector = item["embedding"]
            completed[position] = (case, {
                "id": case.id,
                "text": text,
                "embedding": vector,
                "provider": backend.name,
                "model": backend.model,
                "dimensions": len(vector),
            })
            progress.advance()

        embedded = await embed_text_batches(
            [text for _, _, text in prepared],
            backend=backend,
            input_type="document",
            batch_size=batch_size,
            concurrency=concurrency,
            timeout=timeout,
            retries=retries,
            rpm=rpm,
            on_item=complete,
        )
    nodes = [completed[position][0] for position in sorted(completed)]
    embeddings = [completed[position][1] for position in sorted(completed)]
    failed_cases = [failures[position] for position in sorted(failures)]

    def construct():
        index = BM25Index.from_records(embeddings)
        linked = link_cases(
            nodes,
            embeddings,
            top_k=top_k,
            neighbor_filter=neighbor_filter,
            rrf_constant=rrf_constant,
            bm25_index=index,
            show_progress=show_progress,
        )
        result = {
            "nodes": nodes,
            "embeddings": embeddings,
            **linked,
            "failed_cases": failed_cases,
            "summary": {
                "total": len(cases),
                **linked["summary"],
                "failed": len(failed_cases),
            },
            "embedding_summary": {
                "total": len(cases),
                "embedded": len(nodes),
                "skipped": 0,
                "failed": len(failed_cases),
                "items": len(embeddings),
            },
            "embedding_usage": embedded["usage"],
            "embedding_requests": embedded["requests"],
        }
        return result

    return await asyncio.to_thread(construct)
