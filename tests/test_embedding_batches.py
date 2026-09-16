import asyncio
import inspect
from collections import Counter
from types import SimpleNamespace

import pytest

from raft.embedding._batching import embed_text_batches
from raft.embedding.backend import EmbeddingBatch
from raft.runtime import RetryDecision


class HTTPFailure(Exception):
    def __init__(self, status):
        self.status = status
        super().__init__(f"HTTP {status}")


def success(texts, tokens=None):
    return EmbeddingBatch(
        [[float(ord(text[0])), float(len(text))] for text in texts],
        {"total_tokens": len(texts) if tokens is None else tokens},
    )


class Backend:
    name = model = "fake"

    def __init__(self, respond=success, *, retry_validation=False):
        self.respond = respond
        self.retry_validation = retry_validation
        self.calls = []
        self.input_types = []
        self.errors = []

    async def embed(self, texts, *, input_type="document"):
        self.calls.append(list(texts))
        self.input_types.append(input_type)
        response = self.respond(texts)
        return await response if inspect.isawaitable(response) else response

    def classify_error(self, exc):
        self.errors.append(exc)
        if isinstance(exc, HTTPFailure):
            return RetryDecision(exc.status >= 500, f"http_{exc.status}")
        if isinstance(exc, TimeoutError):
            return RetryDecision(True, "timeout")
        return RetryDecision(self.retry_validation, "invalid_vectors")


async def run(texts, backend, **overrides):
    options = dict(
        backend=backend, input_type="document", batch_size=2,
        concurrency=1, timeout=1, retries=0, rpm=1000,
    )
    options.update(overrides)
    return await embed_text_batches(texts, **options)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "2", None])
async def test_invalid_batch_size_fails_before_provider_calls(batch_size):
    backend = Backend()
    with pytest.raises(ValueError, match="batch_size.*positive integer"):
        await run(["a"], backend, batch_size=batch_size)
    assert backend.calls == []


async def test_empty_input_makes_no_requests_or_callbacks():
    backend = Backend()
    completed = []
    result = await run([], backend, on_item=lambda *args: completed.append(args))
    assert result == {"items": [], "usage": {}, "requests": 0}
    assert backend.calls == completed == []


async def test_boundaries_remainder_and_input_order_survive_out_of_order_completion():
    release_first = asyncio.Event()

    async def respond(texts):
        if texts == ["a", "b"]:
            await release_first.wait()
        if texts == ["e"]:
            release_first.set()
        return success(texts)

    backend = Backend(respond)
    completed = []
    result = await run(
        list("abcde"), backend, concurrency=2, input_type="query",
        on_item=lambda index, item: completed.append(index),
    )
    assert backend.calls == [["a", "b"], ["c", "d"], ["e"]]
    assert backend.input_types == ["query"] * 3
    assert completed == [2, 3, 4, 0, 1]
    assert [item["embedding"] for item in result["items"]] == success(list("abcde")).vectors
    assert result["requests"] == 3
    assert result["usage"] == {"total_tokens": 5}
    assert all("usage" not in item for item in result["items"])
    assert all(item["attempts"] == item["requests"] == 1 for item in result["items"])


async def test_concurrency_limits_active_requests_and_reuses_available_workers():
    started = asyncio.Event()
    release = asyncio.Event()
    active = peak = 0

    async def respond(texts):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            started.set()
        try:
            await release.wait()
            return success(texts)
        finally:
            active -= 1

    backend = Backend(respond)
    task = asyncio.create_task(run(list("abcdefg"), backend, concurrency=2))
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert len(backend.calls) == active == 2
        release.set()
        result = await asyncio.wait_for(task, 1)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert peak == 2 and active == 0
    assert result["requests"] == len(backend.calls) == 4
    assert all(item["error"] is None for item in result["items"])


async def test_rpm_counts_actual_requests_including_retries_and_splits(monkeypatch):
    # Exercise the real rolling limiter with a logical clock, without a minute-long wait.
    now = 0.0
    sleeps = []
    original_sleep = asyncio.sleep

    async def sleep(seconds):
        nonlocal now
        sleeps.append(seconds)
        now += seconds
        await original_sleep(0)

    runtime_asyncio = SimpleNamespace(**vars(asyncio))
    runtime_asyncio.sleep = sleep
    monkeypatch.setattr("raft.runtime.time", SimpleNamespace(monotonic=lambda: now))
    monkeypatch.setattr("raft.runtime.asyncio", runtime_asyncio)
    monkeypatch.setattr("raft.embedding._batching._retry_delay", lambda *args: 0)
    starts = []
    counts = Counter()

    def respond(texts):
        starts.append(now)
        counts[tuple(texts)] += 1
        if texts == ["c", "bad"]:
            raise HTTPFailure(400)
        if texts == ["c"] and counts[tuple(texts)] == 1:
            raise TimeoutError("retry the singleton")
        if texts == ["bad"]:
            raise HTTPFailure(422)
        return success(texts)

    backend = Backend(respond)
    result = await run(["a", "b", "c", "bad"], backend, retries=1, rpm=2)
    assert backend.calls == [["a", "b"], ["c", "bad"], ["c"], ["c"], ["bad"]]
    assert starts == [0.0, 0.0, 60.0, 60.0, 120.0]
    assert sleeps == [60.0, 60.0]
    assert result["requests"] == 5
    assert [item["requests"] for item in result["items"]] == [1, 1, 3, 2]
    assert result["usage"] == {"total_tokens": 3}
    assert all("usage" not in item for item in result["items"])


@pytest.mark.parametrize("malformed", [[[1.0]], [[float("nan")], [1.0]]])
async def test_retry_preserves_completed_batches_and_counts_each_response_usage_once(
    monkeypatch, malformed,
):
    monkeypatch.setattr("raft.embedding._batching._retry_delay", lambda *args: 0)
    counts = Counter()

    def respond(texts):
        counts[tuple(texts)] += 1
        if texts == ["a", "b"]:
            return success(texts, tokens=10)
        attempt = counts[tuple(texts)]
        if attempt == 1:
            raise TimeoutError("no response")
        if attempt == 2:
            return EmbeddingBatch(malformed, {"total_tokens": 7})
        return success(texts, tokens=11)

    backend = Backend(respond, retry_validation=True)
    completed = []
    result = await run(
        list("abcd"), backend, retries=2,
        on_item=lambda index, item: completed.append(index),
    )
    assert backend.calls == [["a", "b"], ["c", "d"], ["c", "d"], ["c", "d"]]
    assert result["requests"] == 4
    assert result["usage"] == {"total_tokens": 28}
    assert completed == [0, 1, 2, 3]
    assert [item["attempts"] for item in result["items"]] == [1, 1, 3, 3]
    assert all(item["error"] is None and "usage" not in item for item in result["items"])


@pytest.mark.parametrize("malformed", [[], [[]], [[float("inf")]], [[1.0], [1.0, 2.0]]])
async def test_malformed_singleton_retains_usage_and_does_not_block_later_batches(malformed):
    def respond(texts):
        if texts == ["a"]:
            return EmbeddingBatch(malformed, {"prompt_tokens": 9, "total_tokens": 9})
        return EmbeddingBatch([[2.0]], {"prompt_tokens": 3, "total_tokens": 3})

    backend = Backend(respond)
    result = await run(["a", "b"], backend, batch_size=1)
    first, second = result["items"]
    assert isinstance(first["error"], ValueError)
    assert first["embedding"] is None
    assert "usage" not in first
    assert second["error"] is None and second["embedding"] == [2.0]
    assert "usage" not in second
    assert result["usage"] == {"prompt_tokens": 12, "total_tokens": 12}
    assert result["requests"] == 2


@pytest.mark.parametrize("status", [400, 413, 422])
async def test_input_failures_split_and_isolate_bad_text_without_truncation(status):
    bad = "bad-" + "x" * 10_000
    error = HTTPFailure(status)

    def respond(texts):
        if bad in texts:
            raise error
        return success(texts)

    backend = Backend(respond)
    completed = []
    result = await run(
        ["a", bad, "c", "d"], backend, batch_size=4,
        on_item=lambda index, item: completed.append(index),
    )
    assert backend.calls == [["a", bad, "c", "d"], ["a", bad], ["a"], [bad], ["c", "d"]]
    assert completed == [0, 1, 2, 3]
    assert result["items"][1]["error"] is error
    assert result["items"][1]["embedding"] is None
    assert all(result["items"][i]["error"] is None for i in (0, 2, 3))
    assert [item["attempts"] for item in result["items"]] == [3, 3, 2, 2]
    assert result["requests"] == 5
    assert result["usage"] == {"total_tokens": 3}


async def test_oversized_request_splits_until_all_inputs_succeed():
    def respond(texts):
        if len(texts) > 2:
            raise HTTPFailure(413)
        return success(texts)

    backend = Backend(respond)
    result = await run(list("abcde"), backend, batch_size=5)
    assert backend.calls == [["a", "b", "c", "d", "e"], ["a", "b"], ["c", "d", "e"], ["c"], ["d", "e"]]
    assert all(item["error"] is None for item in result["items"])
    assert [item["embedding"] for item in result["items"]] == success(list("abcde")).vectors
    assert result["requests"] == 5
    assert result["usage"] == {"total_tokens": 5}


@pytest.mark.parametrize("status, attempts", [(401, 1), (403, 1), (500, 2), (503, 2)])
async def test_auth_and_server_failures_are_not_split(monkeypatch, status, attempts):
    monkeypatch.setattr("raft.embedding._batching._retry_delay", lambda *args: 0)
    error = HTTPFailure(status)

    def respond(texts):
        if texts == ["a", "b"]:
            raise error
        return success(texts)

    backend = Backend(respond)
    result = await run(list("abcd"), backend, retries=1)
    assert backend.calls == [["a", "b"]] * attempts + [["c", "d"]]
    assert all(item["error"] is error for item in result["items"][:2])
    assert all(item["attempts"] == attempts for item in result["items"][:2])
    assert all(item["error"] is None for item in result["items"][2:])
    assert result["requests"] == attempts + 1
    assert result["usage"] == {"total_tokens": 2}


@pytest.mark.parametrize("source", ["caller", "backend"])
async def test_cancellation_propagates_and_cleans_up_active_workers(source):
    started = asyncio.Event()
    blocked = asyncio.Event()
    entered = cancelled = 0

    async def respond(texts):
        nonlocal entered, cancelled
        entered += 1
        if entered == 2:
            started.set()
        try:
            if source == "backend" and texts == ["a", "b"]:
                await started.wait()
                raise asyncio.CancelledError()
            await blocked.wait()
        except asyncio.CancelledError:
            cancelled += 1
            raise

    backend = Backend(respond)
    completed = []
    task = asyncio.create_task(run(
        list("abcdef"), backend, concurrency=2,
        on_item=lambda *args: completed.append(args),
    ))
    try:
        await asyncio.wait_for(started.wait(), 1)
        if source == "caller":
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert entered == cancelled == 2
    assert backend.calls == [["a", "b"], ["c", "d"]]
    assert backend.errors == completed == []


async def test_timeout_waiting_for_rpm_does_not_count_an_unsent_request(monkeypatch):
    class BlockedLimiter:
        async def acquire(self):
            await asyncio.Event().wait()

    monkeypatch.setattr("raft.embedding._batching._RollingRateLimiter", lambda rpm: BlockedLimiter())
    backend = Backend()
    result = await run(["a", "b"], backend, timeout=0.01)
    assert backend.calls == []
    assert result["requests"] == 0 and result["usage"] == {}
    assert all(item["requests"] == 0 and item["attempts"] == 1 for item in result["items"])
    assert all(isinstance(item["error"], TimeoutError) for item in result["items"])
