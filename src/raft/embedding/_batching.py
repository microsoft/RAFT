"""Bounded multi-input requests with response-level usage and indexed outcomes."""

from __future__ import annotations

import asyncio
import math
import time
from contextlib import nullcontext
from typing import Any, Callable, Literal, Sequence

from raft.progress import CaseProgress
from raft.runtime import _retry_delay, _RollingRateLimiter, map_concurrent, validate_limits

from .backend import EmbeddingBackend, EmbeddingBatch


def validate_batch_size(batch_size: int) -> None:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")


async def embed_text_batches(
    texts: Sequence[str],
    *,
    backend: EmbeddingBackend,
    input_type: Literal["document", "query"],
    batch_size: int,
    concurrency: int,
    timeout: float,
    retries: int,
    rpm: int,
    on_item: Callable[[int, dict[str, Any]], None] | None = None,
    progress: CaseProgress | None = None,
) -> dict[str, Any]:
    """Embed ready inputs without waiting for other calls to fill a batch.

    Concurrency bounds active batch workers; timeout bounds each request attempt,
    including its RPM wait. Retry transient failures without repeating successful
    batches. Multi-input HTTP 400/413/422 failures are split to isolate invalid
    inputs or reduce a request that exceeds the provider's limits. Individual
    overlength texts are never truncated. Other failures affect their batch.

    Aggregate usage counts each observed response once, including malformed
    responses. Token usage is reported only at operation level.
    Per-item attempts/requests count participation, not disjoint HTTP requests.
    An optional caller-owned progress counts retries/throttles once per batch,
    not once per item; it does not advance terminal item counts or own the bar.
    """
    validate_limits(concurrency, timeout, retries, rpm)
    validate_batch_size(batch_size)
    limiter = _RollingRateLimiter(rpm)
    usage: dict[str, int] = {}
    requests = 0
    items = [
        {
            "embedding": None,
            "error": None,
            "decision": None,
            "attempts": 0,
            "requests": 0,
            "elapsed_seconds": 0.0,
        }
        for _ in texts
    ]

    async def process(indices: list[int], started: float | None = None) -> None:
        nonlocal requests
        if started is None:
            started = time.monotonic()
        for attempt in range(retries + 1):
            for i in indices:
                items[i]["attempts"] += 1
            try:
                async with asyncio.timeout(timeout):
                    await limiter.acquire()
                    requests += 1
                    for i in indices:
                        items[i]["requests"] += 1
                    response = await backend.embed(
                        [texts[i] for i in indices], input_type=input_type
                    )
                    if isinstance(response, EmbeddingBatch):
                        for key, value in response.usage.items():
                            usage[key] = usage.get(key, 0) + value
                        vectors = response.vectors
                    else:
                        vectors = response
                    if len(vectors) != len(indices):
                        raise ValueError("Provider returned the wrong number of embeddings")
                    dimensions = len(vectors[0])
                    if not dimensions or any(
                        len(vector) != dimensions
                        or not all(math.isfinite(value) for value in vector)
                        for vector in vectors
                    ):
                        raise ValueError(
                            "Embedding vectors must have consistent dimensions and finite values"
                        )
                for i, vector in zip(indices, vectors, strict=True):
                    items[i]["embedding"] = vector
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                decision = backend.classify_error(exc)
                if progress is not None:
                    progress.observe_error(decision.category)
                if decision.retryable and attempt < retries:
                    with progress.retry_wait() if progress is not None else nullcontext():
                        await asyncio.sleep(_retry_delay(decision, attempt + 1))
                    continue
                if (
                    not decision.retryable
                    and decision.category in {"http_400", "http_413", "http_422"}
                    and len(indices) > 1
                ):
                    middle = len(indices) // 2
                    await process(indices[:middle], started)
                    await process(indices[middle:], started)
                    return
                for i in indices:
                    items[i].update(error=exc, decision=decision)
                break
        for i in indices:
            items[i]["elapsed_seconds"] = round(time.monotonic() - started, 3)
            if on_item is not None:
                on_item(i, items[i])

    batches = [list(range(i, min(i + batch_size, len(texts)))) for i in range(0, len(texts), batch_size)]
    await map_concurrent(batches, process, concurrency)
    return {"items": items, "usage": usage, "requests": requests}
