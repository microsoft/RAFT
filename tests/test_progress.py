"""Progress counts terminal outcomes, including errors, without changing scheduling."""

import asyncio
import io
import logging

import httpx2
import openai
import pytest
from agent_helpers import run_cases
from test_local_pipeline import Embeddings, pipeline, raw
from test_stages import Worker, cases, config

from raft import build_case_graph, embed_cases
from raft._openai_errors import classify_error
from raft.embedding._batching import embed_text_batches
from raft.progress import CaseProgress
from raft.runtime import map_concurrent


@pytest.fixture
def bars(monkeypatch):
    import tqdm.auto

    created = []

    class Bar:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.n = 0
            self.closed = False
            self.counts = {}
            self.history = []
            created.append(self)

        def set_postfix(self, counts, **kwargs):
            self.counts = dict(counts)
            self.history.append((self.n, dict(counts)))

        def update(self, n):
            self.n += n

        def close(self):
            self.closed = True

    monkeypatch.setattr(tqdm.auto, "tqdm", Bar)
    return created


RETRY_ZERO = {"retries": 0, "rate_limited": 0, "waiting_retry": 0}


def rate_limit(code="rate_limit_exceeded"):
    return openai.RateLimitError(
        "Rate limited",
        response=httpx2.Response(
            429, headers={"retry-after": "0", "x-request-id": "request-test"},
            request=httpx2.Request("POST", "https://provider.example"),
        ),
        body={"code": code},
    )


@pytest.mark.asyncio
async def test_extraction_and_embedding_count_final_outcomes(bars):
    settings = config(Worker(), Embeddings())
    extraction = await run_cases(
        cases=cases("ok", "filtered", "scan-fails"),
        should_keep=lambda case: case.output.extractable,
        show_progress=True, **settings["extraction"],
    )
    assert bars[0].counts == {**RETRY_ZERO, "succeeded": 1, "failed": 1, "filtered": 1}
    assert bars[0].n == bars[0].options["total"] == 3
    assert bars[0].closed
    records = [
        *extraction["extracted_cases"], *extraction["filtered_cases"],
        {"id": "invalid", "output": {}},
    ]
    await embed_cases(cases=records, show_progress=True, **settings["embedding"])
    assert bars[1].counts == {**RETRY_ZERO, "succeeded": 1, "failed": 1, "skipped": 1}
    assert bars[1].n == 3 and bars[1].closed


@pytest.mark.asyncio
async def test_completion_order_and_cancellation_close_progress(bars):
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def operation(item):
        if item == 0:
            first_started.set()
            await release_first.wait()
        else:
            await first_started.wait()
        return item

    task = asyncio.create_task(map_concurrent([0, 1], operation, 2, show_progress=True))
    await first_started.wait()
    await asyncio.sleep(0)
    assert bars[0].n == 1
    release_first.set()
    assert await task == [0, 1]
    assert bars[0].n == 2 and bars[0].closed

    first_started.clear()
    release_first.clear()
    task = asyncio.create_task(map_concurrent([0], operation, 1, show_progress=True))
    await first_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert bars[1].closed and bars[1].n == 0


@pytest.mark.asyncio
async def test_disabled_is_silent_and_error_closes_bar(bars):
    async def operation(item):
        return item

    assert await map_concurrent([1], operation, 1) == [1]
    assert bars == []

    async def fail(item):
        raise ValueError("failure")

    with pytest.raises(ValueError, match="failure"):
        await map_concurrent([1], fail, 1, show_progress=True)
    assert bars[0].closed and bars[0].n == 0


@pytest.mark.asyncio
async def test_pipeline_propagates_progress_to_all_runners(bars, tmp_path):
    p = pipeline(
        tmp_path, show_progress=True,
        graph={"backend": Embeddings(), "case_to_text": lambda s: " ".join(s.timeline), "rpm": 1000},
    )
    assert all(c["show_progress"] for c in (p.extraction, p.embedding, p.graph))
    await p.index([raw("a"), raw("b")])
    assert {b.options["desc"] for b in bars} == {
        "Extraction", "Embedding", "Graph embeddings", "Graph neighbors",
    }
    assert all(b.closed and b.n == b.options["total"] for b in bars)
    await p.retrieve(["test"])
    assert bars[-1].options["desc"] == "Retrieval"
    assert bars[-1].counts == {**RETRY_ZERO, "succeeded": 1, "failed": 0}


def test_terminal_progress_renders_counters(monkeypatch):
    import tqdm.auto
    from tqdm import tqdm as terminal_tqdm

    stream = io.StringIO()
    monkeypatch.setattr(tqdm.auto, "tqdm", lambda **kwargs: terminal_tqdm(file=stream, **kwargs))
    with CaseProgress(2, enabled=True, desc="Extraction") as progress:
        progress.observe_error("rate_limit")
        with progress.retry_wait():
            pass
        progress.advance()
        progress.advance("failed")
    assert "succeeded=1" in stream.getvalue() and "failed=1" in stream.getvalue()
    assert "100%" in stream.getvalue()
    assert "retries=1" in stream.getvalue()
    assert "rate_limited=1" in stream.getvalue()
    assert progress.counts["waiting_retry"] == 0


def test_retry_counters_refresh_without_advancing_and_balance_exception(bars):
    with CaseProgress(2, enabled=True, desc="test") as progress:
        progress.observe_error("rate_limit")
        assert bars[0].n == 0 and bars[0].counts["rate_limited"] == 1
        with pytest.raises(ValueError):
            with progress.retry_wait():
                assert bars[0].counts["waiting_retry"] == bars[0].counts["retries"] == 1
                with progress.retry_wait():
                    assert bars[0].counts["waiting_retry"] == 2
                    raise ValueError("cancel both waits")
        assert bars[0].counts["retries"] == 2
        assert bars[0].counts["waiting_retry"] == 0 and bars[0].n == 0
    assert bars[0].closed


@pytest.mark.asyncio
async def test_borrowed_progress_is_not_closed_or_replaced_by_map(bars):
    async def operation(item):
        return item

    with CaseProgress(2, enabled=True, desc="owned") as progress:
        assert await map_concurrent([1, 2], operation, 2, progress=progress) == [1, 2]
        assert len(bars) == 1 and not bars[0].closed
        assert bars[0].n == 2
    assert bars[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["worker", "review"])
async def test_extraction_recovered_throttle_counts_once_not_as_failure(bars, stage):
    class ThrottledWorker(Worker):
        errored = False
        worker_invocations = 0

        async def run(self, *args, **kwargs):
            self.worker_invocations += 1
            if stage == "worker" and not self.errored:
                self.errored = True
                raise rate_limit()
            return await super().run(*args, **kwargs)

        async def review(self, *args, **kwargs):
            if stage == "review" and not self.errored:
                self.errored = True
                raise rate_limit()
            return await super().review(*args, **kwargs)

        def classify_error(self, exc):
            return classify_error(exc)

    worker = ThrottledWorker()
    settings = config(worker, Embeddings())["extraction"]
    settings["retries"] = 2
    result = await run_cases(cases=cases("ok"), show_progress=True, **settings)
    assert not result["failed_cases"] and len(result["extracted_cases"]) == 1
    assert worker.worker_invocations == (2 if stage == "worker" else 1)
    assert bars[0].counts == {
        "succeeded": 1, "failed": 0, "retries": 1, "rate_limited": 1, "waiting_retry": 0,
    }
    assert any(n == 0 and c["waiting_retry"] == 1 for n, c in bars[0].history)
    assert result["extracted_cases"][0].execution["attempts"] == 2
    assert bars[0].n == 1 and bars[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("code,expected_retries,expected_throttles", [
    ("rate_limit_exceeded", 2, 3),
    ("insufficient_quota", 0, 0),
])
async def test_exhausted_throttles_and_terminal_quota_are_distinct(
    bars, code, expected_retries, expected_throttles
):
    class FailingWorker(Worker):
        async def run(self, *args, **kwargs):
            raise rate_limit(code)

        def classify_error(self, exc):
            return classify_error(exc)

    settings = config(FailingWorker(), Embeddings())["extraction"]
    settings["retries"] = 2
    result = await run_cases(cases=cases("bad"), show_progress=True, **settings)
    assert len(result["failed_cases"]) == 1
    failure = result["failed_cases"][0]
    assert failure["stage"] == "worker"
    assert failure["execution"]["attempts"] == expected_retries + 1
    assert failure["error_details"] == {
        "status_code": 429, "code": code, "request_id": "request-test", "retry_after": 0,
    }
    assert bars[0].counts == {
        "succeeded": 0, "failed": 1, "retries": expected_retries,
        "rate_limited": expected_throttles, "waiting_retry": 0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status,exception_type", [
    (400, openai.BadRequestError), (401, openai.AuthenticationError),
    (403, openai.PermissionDeniedError), (404, openai.NotFoundError),
])
async def test_unavailable_model_or_auth_errors_are_terminal_per_case(
    bars, status, exception_type
):
    class MissingModel(Worker):
        def __init__(self):
            super().__init__()
            self.invocations = 0

        async def run(self, *args, **kwargs):
            self.invocations += 1
            raise exception_type(
                "Model not available",
                response=httpx2.Response(
                    status, headers={"x-request-id": "model-test"},
                    request=httpx2.Request("POST", "https://provider.example"),
                ),
                body={"error": {"code": "model_not_found", "private_input": "do not copy"}},
            )

        def classify_error(self, exc):
            return classify_error(exc)

    worker = MissingModel()
    settings = config(worker, Embeddings())["extraction"]
    settings["retries"] = 3
    result = await run_cases(cases=cases("a", "b"), show_progress=True, **settings)
    assert worker.invocations == 2
    assert len(result["failed_cases"]) == 2
    assert bars[0].counts == {**RETRY_ZERO, "succeeded": 0, "failed": 2}
    for failure in result["failed_cases"]:
        assert failure["retryable"] is False
        assert failure["error_details"] == {
            "status_code": status, "code": "model_not_found", "request_id": "model-test",
        }


@pytest.mark.asyncio
async def test_embedding_retry_keeps_completed_batches(bars):
    from test_retrieval import State

    from raft import ExtractedCase

    class Throttled(Embeddings):
        errored = False

        async def embed(self, texts, *, input_type="document"):
            if texts == ["second"] and not self.errored:
                self.calls.append(texts)
                self.errored = True
                raise rate_limit()
            return await super().embed(texts, input_type=input_type)

        def classify_error(self, exc):
            return classify_error(exc)

    backend = Throttled()
    result = await embed_cases(
        cases=[ExtractedCase(id="a", metadata={}, output=State(timeline=["first", "second"]))],
        backend=backend, state_to_text=lambda state: state.timeline,
        batch_size=1, retries=1, show_progress=True, rpm=1000,
    )
    assert backend.calls == [["first"], ["second"], ["second"]]
    assert not result["failed_cases"]
    assert bars[0].counts == {
        "succeeded": 1, "failed": 0, "retries": 1, "rate_limited": 1, "waiting_retry": 0,
    }
    assert bars[0].n == 1


class ThrottledBatch(Embeddings):
    def __init__(self):
        super().__init__()
        self.errored = False

    async def embed(self, texts, *, input_type="document"):
        if not self.errored:
            self.errored = True
            self.calls.append(texts)
            raise rate_limit()
        return await super().embed(texts, input_type=input_type)

    def classify_error(self, exc):
        return classify_error(exc)


@pytest.mark.asyncio
async def test_graph_counts_one_retry_for_shared_batch_not_each_case(bars):
    from test_retrieval import fixture

    case_list, _ = fixture()
    result = await build_case_graph(
        cases=case_list, backend=ThrottledBatch(),
        case_to_text=lambda state: state.timeline[0],
        retries=1, rpm=1000, show_progress=True,
    )
    assert not result["failed_cases"]
    assert len(bars) == 2  # Embeddings + neighbors, not an extra hidden retry bar.
    assert bars[0].counts == {
        "succeeded": 3, "failed": 0, "retries": 1, "rate_limited": 1, "waiting_retry": 0,
    }
    assert bars[0].n == 3 and bars[0].closed
    assert bars[1].counts["retries"] == 0


@pytest.mark.asyncio
async def test_retrieval_shared_embeddings_do_not_multiply_retry_counts(bars):
    from test_retrieval import fixture

    from raft import LocalRetriever

    case_list, rows = fixture()
    backend = ThrottledBatch()
    backend.name, backend.model = "fake", "test"
    result = await LocalRetriever(cases=case_list, embeddings=rows).retrieve(
        ["one", "two", "three"], backend=backend, retries=1, rpm=1000, show_progress=True,
    )
    assert all(r["error"] is None for r in result["results"])
    assert result["embedding_requests"] == 2
    assert len(bars) == 1 and bars[0].n == 3
    assert bars[0].counts == {
        "succeeded": 3, "failed": 0, "retries": 1, "rate_limited": 1, "waiting_retry": 0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["filter", "rank"])
async def test_retrieval_local_retry_counts_without_reembedding(bars, monkeypatch, phase):
    from test_retrieval import fixture

    from raft import LocalRetriever

    case_list, rows = fixture()
    retriever = LocalRetriever(cases=case_list, embeddings=rows)
    backend = ThrottledBatch()
    backend.errored = True
    backend.name, backend.model = "fake", "test"
    attribute = "_eligible" if phase == "filter" else "_rank"
    original = getattr(retriever, attribute)
    calls = 0

    def flaky(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("local phase timed out")
        return original(*args)

    monkeypatch.setattr(retriever, attribute, flaky)
    monkeypatch.setattr("raft.retrieval.local._retry_delay", lambda *args: 0)
    result = await retriever.retrieve(
        ["one"], backend=backend, retries=1, rpm=1000, show_progress=True,
    )
    assert result["results"][0]["error"] is None
    assert len(backend.calls) == 1
    assert bars[0].counts == {
        "succeeded": 1, "failed": 0, "retries": 1, "rate_limited": 0, "waiting_retry": 0,
    }


@pytest.mark.asyncio
async def test_shared_embedding_split_is_not_a_retry_or_throttle(bars):
    from raft.runtime import RetryDecision

    class Split(Embeddings):
        async def embed(self, texts, *, input_type="document"):
            if len(texts) > 1:
                raise ValueError("split this batch")
            return await super().embed(texts, input_type=input_type)

        def classify_error(self, exc):
            return RetryDecision(False, "http_400")

    with CaseProgress(2, enabled=True, desc="batch") as progress:
        result = await embed_text_batches(
            ["a", "b"], backend=Split(), input_type="document",
            batch_size=2, concurrency=2, timeout=10, retries=2, rpm=1000,
            progress=progress, on_item=lambda i, item: progress.advance(),
        )
    assert result["requests"] == 3
    assert bars[0].counts == {**RETRY_ZERO, "succeeded": 2, "failed": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["extraction", "embedding", "graph", "retrieval"])
async def test_cancelled_backoff_clears_waiting_and_does_not_advance(bars, monkeypatch, stage):
    from test_retrieval import fixture

    from raft import LocalRetriever

    started = asyncio.Event()

    async def sleep(delay):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncio, "sleep", sleep)
    backend = ThrottledBatch()
    case_list, rows = fixture()

    class ThrottledWorker(Worker):
        async def run(self, *args, **kwargs):
            raise rate_limit()

        def classify_error(self, exc):
            return classify_error(exc)

    if stage == "extraction":
        settings = config(ThrottledWorker(), backend)["extraction"]
        settings["retries"] = 1
        operation = run_cases(cases=cases("one"), show_progress=True, **settings)
    elif stage == "embedding":
        operation = embed_cases(
            cases=case_list[:1], backend=backend, state_to_text=lambda state: state.timeline,
            show_progress=True,
        )
    elif stage == "graph":
        operation = build_case_graph(
            cases=case_list, backend=backend, case_to_text=lambda state: state.timeline[0],
            show_progress=True,
        )
    else:
        backend.name, backend.model = "fake", "test"
        operation = LocalRetriever(cases=case_list, embeddings=rows).retrieve(
            ["one"], backend=backend, show_progress=True,
        )
    task = asyncio.create_task(operation)
    await asyncio.wait_for(started.wait(), timeout=10)
    assert bars[0].n == 0 and bars[0].counts["waiting_retry"] == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert bars[0].counts["waiting_retry"] == 0
    assert bars[0].n == 0 and bars[0].closed


@pytest.mark.asyncio
async def test_disabled_retry_progress_does_not_change_results_or_print(bars, capsys):
    from test_retrieval import fixture

    case_list, _ = fixture()
    result = await embed_cases(
        cases=case_list[:1], backend=ThrottledBatch(),
        state_to_text=lambda state: state.timeline, show_progress=False,
    )
    assert len(result["embedded_cases"]) == 1
    assert not result["failed_cases"]
    assert bars == []
    assert capsys.readouterr().err == ""


@pytest.mark.asyncio
async def test_pipeline_accepts_quiet_flag_and_returns_failure_details(bars, caplog, tmp_path):
    class MissingModel(Worker):
        async def run(self, *args, **kwargs):
            logging.getLogger("openai.agents").error("%s", "Error getting response")
            raise openai.NotFoundError(
                "Unknown deployment",
                response=httpx2.Response(
                    404, headers={"x-request-id": "missing-model"},
                    request=httpx2.Request("POST", "https://provider.example"),
                ),
                body={"code": "DeploymentNotFound"},
            )

        def classify_error(self, exc):
            return classify_error(exc)

    settings = config(MissingModel(), Embeddings())
    settings["extraction"]["suppress_response_errors"] = True
    from agent_helpers import LocalPipeline

    caplog.set_level(logging.ERROR, logger="openai.agents")
    result = await LocalPipeline(tmp_path, **settings).index(cases("one"))
    failure = result["extraction"]["failed_cases"][0]
    assert failure["error_type"] == "NotFoundError"
    assert failure["error_details"]["status_code"] == 404
    assert failure["error_details"]["request_id"] == "missing-model"
    assert "Error getting response" not in caplog.text


def test_pipeline_stage_override_does_not_mutate_caller_settings(tmp_path):
    from agent_helpers import LocalPipeline

    settings = config(Worker(), Embeddings())
    settings["extraction"]["show_progress"] = False
    p = LocalPipeline(tmp_path, show_progress=True, **settings)
    assert p.extraction["show_progress"] is False
    assert p.embedding["show_progress"] is True
    assert "show_progress" not in settings["embedding"]


@pytest.mark.parametrize("omit", [False, True])
def test_pipeline_requires_reviewer_at_construction(tmp_path, omit):
    from agent_helpers import LocalPipeline

    settings = config(Worker(), Embeddings())
    if omit:
        settings["extraction"].pop("reviewer_agent")
    else:
        settings["extraction"]["reviewer_agent"] = None
    with pytest.raises((TypeError, ValueError), match="reviewer_agent"):
        LocalPipeline(tmp_path, **settings)
