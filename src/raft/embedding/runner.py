from __future__ import annotations

import asyncio
import hashlib
import math
import time
from typing import Any, Callable, Sequence, TypeVar

from pydantic import BaseModel

from raft._json import _id_key, _to_json
from raft.cases import ExtractedCase, restore_case
from raft.runtime import (
    RetryDecision,
    _failed_case,
    _retry_delay,
    _RollingRateLimiter,
    map_concurrent,
    validate_limits,
)

from ._batching import validate_batch_size
from .backend import EmbeddingBackend, EmbeddingBatch

StateT = TypeVar("StateT", bound=BaseModel)


async def embed_cases(
    *,
    cases: Sequence[ExtractedCase[StateT] | dict[str, Any]],
    backend: EmbeddingBackend,
    state_to_text: Callable[[StateT], list[str]],
    should_embed: Callable[[StateT], bool] | None = None,
    output_type: type[StateT] | None = None,
    batch_size: int = 64,
    concurrency: int = 4,
    timeout: float = 120.0,
    retries: int = 1,
    rpm: int = 60,
    show_progress: bool = False,
    progress_desc: str = "Embedding",
) -> dict[str, Any]:
    """Convert each final Pydantic state into an ordered list of texts and embed it.

    Accepts canonical run_cases()['extracted_cases'] records. Supply output_type
    only when restoring serialized outputs; live models and records are reused.
    state_to_text receives the entire
    model once per case and returns one string per desired embedding. Returned
    list positions become item_index in returned records. An empty list yields no
    vectors. The callback is synchronous and should only prepare text locally.

    should_embed optionally receives the live output model once per valid case,
    before text conversion or API calls: True embeds, False skips. Skipped cases
    are returned as canonical ExtractedCase records, not failures. Exceptions or
    non-boolean results are terminal should_embed_error failures. Without a
    predicate every valid case is eligible, regardless of its output fields.

    The worker limit is per case, RPM per embedding request. timeout covers a
    whole case attempt, and retries resume at the first uncompleted batch.
    Only complete cases are published; failures retain no partial vectors.
    embedding_usage and embedding_requests retain observed usage/calls once across
    the operation, including responses before a later failure. Per-case records
    contain no embedding token usage. Missing response usage is not estimated.
    batch_size is a text-count limit; each text and request must also fit
    the provider's token limits.
    This function does not persist results or construct a BM25 index. Callers
    own storage and corpus-level indexing of the returned texts and vectors.
    """
    validate_limits(concurrency, timeout, retries, rpm)
    validate_batch_size(batch_size)
    if output_type is not None and (
        not isinstance(output_type, type) or not issubclass(output_type, BaseModel)
    ):
        raise ValueError("output_type must be a Pydantic model class")
    limiter = _RollingRateLimiter(rpm)
    embedding_usage: dict[str, int] = {}
    seen: set[str] = set()
    work = []
    for case in cases:
        case_id = (
            case.id
            if isinstance(case, ExtractedCase)
            else (case.get("id") if isinstance(case, dict) else None)
        )
        key = _id_key(case_id)
        work.append((case, key in seen))
        seen.add(key)

    async def process(entry: tuple[Any, bool]) -> tuple[str, dict[str, Any] | ExtractedCase]:
        case, duplicate = entry
        started = time.monotonic()
        case_id = (
            case.id
            if isinstance(case, ExtractedCase)
            else (case.get("id") if isinstance(case, dict) else None)
        )
        attempts = 0
        requests = 0
        vectors: list[list[float]] = []

        def failure(exc: Exception, decision: RetryDecision, failure_type: str):
            item = _failed_case(
                case_id=case_id,
                case=case,
                category=decision.category,
                error=exc,
                retryable=decision.retryable,
                attempts=attempts,
                elapsed=time.monotonic() - started,
                failure_type=failure_type,
            )
            item.update(requests=requests)
            return "failed", item

        try:
            if case_id is None or duplicate:
                raise ValueError("Each case must have a non-null, unique id")
            case = restore_case(case, output_type)
            state = case.output
        except (TypeError, ValueError) as exc:
            return failure(exc, RetryDecision(False, "invalid_case"), "invalid_case")

        if should_embed is not None:
            try:
                eligible = should_embed(state)
                if not isinstance(eligible, bool):
                    raise ValueError("should_embed must return a bool")
                if not eligible:
                    return "skipped", case
            except Exception as exc:
                return failure(exc, RetryDecision(False, "should_embed_error"), "terminal_error")

        try:
            texts = state_to_text(state)
            if not isinstance(texts, list) or any(
                not isinstance(text, str) or not text.strip() for text in texts
            ):
                raise ValueError("state_to_text must return a list of nonempty strings (or [])")
        except Exception as exc:
            return failure(exc, RetryDecision(False, "state_to_text_error"), "terminal_error")

        for attempt in range(retries + 1):
            attempts = attempt + 1
            try:
                async with asyncio.timeout(timeout):
                    while len(vectors) < len(texts):
                        batch = texts[len(vectors) : len(vectors) + batch_size]
                        await limiter.acquire()
                        requests += 1
                        embedded = await backend.embed(batch, input_type="document")
                        if isinstance(embedded, EmbeddingBatch):
                            for key, value in embedded.usage.items():
                                embedding_usage[key] = embedding_usage.get(key, 0) + value
                            embedded = embedded.vectors
                        if len(embedded) != len(batch):
                            raise ValueError("Provider returned the wrong number of embeddings")
                        dimension = len(vectors[0]) if vectors else len(embedded[0])
                        if not dimension or any(
                            len(vector) != dimension or not all(math.isfinite(x) for x in vector)
                            for vector in embedded
                        ):
                            raise ValueError(
                                "Embedding vectors must have consistent dimensions and finite values"
                            )
                        vectors.extend(embedded)
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                decision = backend.classify_error(exc)
                if not decision.retryable or attempt == retries:
                    return failure(
                        exc, decision, "retry_exhausted" if decision.retryable else "terminal_error"
                    )
                await asyncio.sleep(_retry_delay(decision, attempts))

        records = []
        for index, (text, vector) in enumerate(zip(texts, vectors, strict=True)):
            identity = _to_json([case_id, index])
            records.append(
                {
                    "id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                    "case_id": case_id,
                    "item_index": index,
                    "text": text,
                    "embedding": vector,
                    "provider": backend.name,
                    "model": backend.model,
                    "dimensions": len(vector),
                }
            )
        return "embedded", {
            "id": case_id,
            "case": case,
            "embeddings": records,
            "attempts": attempts,
            "requests": requests,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    outcomes = await map_concurrent(
        work, process, concurrency, show_progress=show_progress, progress_desc=progress_desc,
        progress_status=lambda result: "succeeded" if result[0] == "embedded" else result[0],
    )
    embedded_cases = [item for status, item in outcomes if status == "embedded"]
    skipped_cases = [item for status, item in outcomes if status == "skipped"]
    failed_cases = [item for status, item in outcomes if status == "failed"]
    result = {
        "embedded_cases": embedded_cases,
        "skipped_cases": skipped_cases,
        "failed_cases": failed_cases,
        "embedding_usage": embedding_usage,
        "embedding_requests": sum(item["requests"] for item in [*embedded_cases, *failed_cases]),
        "summary": {
            "total": len(cases),
            "embedded": len(embedded_cases),
            "skipped": len(skipped_cases),
            "failed": len(failed_cases),
            "items": sum(len(item["embeddings"]) for item in embedded_cases),
        },
    }
    return result
