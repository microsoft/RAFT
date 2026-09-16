import asyncio
import json
import time
from types import SimpleNamespace

import httpx2
import openai
import pytest
from agent_helpers import REVIEWER, LocalPipeline, WorkerTestRunner
from agents import Agent
from pydantic import BaseModel

from raft.embedding import BM25Index
from raft.extraction._agent import classify_error
from raft.extraction.state import apply_edit
from raft.runtime import (
    RetryDecision,
    _error_details,
    _failed_case,
    _retry_delay,
    _RollingRateLimiter,
    map_concurrent,
)
from raft.tools import edit_state, query_case_sql


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [asyncio.CancelledError, ValueError])
async def test_worker_failure_cancels_and_joins_other_workers(error):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def operation(item):
        if item == "fail":
            await started.wait()
            raise error()
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            stopped.set()

    with pytest.raises(error):
        await asyncio.wait_for(map_concurrent(["fail", "wait"], operation, 2), timeout=1)
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_rolling_rpm_window():
    limiter = _RollingRateLimiter(2, window_seconds=0.03)
    start = time.monotonic()
    await limiter.acquire()
    await limiter.acquire()
    await limiter.acquire()
    assert time.monotonic() - start >= 0.025


def test_quota_vs_transient_rate_limit_and_retry_after():
    response = httpx2.Response(
        429, headers={"retry-after": "90"}, request=httpx2.Request("POST", "https://api.openai.com")
    )
    exc = openai.RateLimitError(
        "rate pressure", response=response, body={"code": "rate_limit_exceeded"}
    )
    assert classify_error(exc).retryable
    assert _retry_delay(classify_error(exc), 1) == 90
    exc = openai.RateLimitError("quota", response=response, body={"code": "insufficient_quota"})
    assert not classify_error(exc).retryable
    assert not classify_error(ValueError("bad input")).retryable
    assert not classify_error(RuntimeError("bug")).retryable


def test_error_details_are_allowlisted_and_do_not_copy_credentials_or_response_body():
    response = httpx2.Response(
        429,
        headers={
            "retry-after": "12", "x-request-id": "req-test",
            "authorization": "Bearer not-for-diagnostics", "set-cookie": "private-cookie",
        },
        request=httpx2.Request("POST", "https://provider.example"),
    )
    exc = openai.RateLimitError(
        "throttled", response=response,
        body={"error": {"code": "rate_limit_exceeded", "private": "model-input"}},
    )
    expected = {
        "status_code": 429, "code": "rate_limit_exceeded", "request_id": "req-test",
        "retry_after": 12.0,
    }
    assert _error_details(exc) == expected
    failure = _failed_case(
        case_id="a", case={}, category="rate_limit", error=exc,
        retryable=True, attempts=3, elapsed=1.25, failure_type="retry_exhausted",
    )
    assert failure["error_details"] == expected
    assert "headers" not in failure and "body" not in failure


def test_nonprovider_errors_keep_the_existing_failure_shape():
    exc = ValueError("invalid input")
    assert _error_details(exc) == {}
    failure = _failed_case(
        case_id="a", case={}, category="invalid_case", error=exc,
        retryable=False, attempts=0, elapsed=0,
    )
    assert "error_details" not in failure
    assert failure["error_message"] == "invalid input"


@pytest.mark.asyncio
@pytest.mark.parametrize("with_bm25", [True, False])
@pytest.mark.parametrize("with_graph", [True, False])
async def test_pipeline_passes_only_kept_outputs_and_saves_results(
    monkeypatch, tmp_path, with_bm25, with_graph
):
    class Output(BaseModel):
        extractable: bool
        timeline: list[str]

    async def run(agent, prompt, *, context, max_turns, hooks, run_config):
        apply_edit(
            context=context,
            patch_json=json.dumps(
                [
                    {
                        "op": "add",
                        "path": "",
                        "value": {
                            "extractable": context.case_id == "keep",
                            "timeline": ["symptom", "resolved"],
                        },
                    }
                ]
            ),
            finish_pass=True,
        )
        return SimpleNamespace(
            new_items=[],
            final_output="done",
        )

    class Backend:
        name = "fake"
        model = "embedding"

        async def embed(self, texts, *, input_type="document"):
            return [[1.0] for _ in texts]

        def classify_error(self, exc):
            return RetryDecision(False, "error")

    monkeypatch.setattr("raft.extraction._agent.Runner.run", run)
    pipeline = LocalPipeline(
        extraction=dict(
            output_type=Output,
            reviewer_agent=REVIEWER,
            _agent_runner=WorkerTestRunner(),
            worker_agent=Agent(name="scan", tools=[query_case_sql, edit_state]),
            id_field="ticket",
            metadata_field="metadata",
            artifacts_field="artifacts",
        ),
        embedding=dict(
            should_embed=lambda out: out.extractable,
            backend=Backend(),
            state_to_text=lambda state: state.timeline,
        ),
        output_dir=tmp_path,
        bm25=with_bm25,
        graph=dict(backend=Backend(), case_to_text=lambda state: state.timeline[-1])
        if with_graph
        else None,
    )
    result = await pipeline.index(
        [{"ticket": t, "metadata": {}, "artifacts": []} for t in ["keep", "skip"]]
    )
    assert result["extraction"]["summary"]["extracted"] == 2
    assert result["embedding"]["summary"]["total"] == 2
    assert result["embedding"]["summary"]["skipped"] == 1
    catalog = pipeline.store.read()
    assert len(catalog["cases"]) == 2
    assert len(pipeline._retriever.rows) == 2
    cache = pipeline.store.cache_dir(catalog)
    if with_bm25:
        index = BM25Index.load(cache / "bm25")
        assert len(index.documents) == 2
        assert index.search("resolved")[0]["case_id"] == "keep"
    else:
        assert pipeline._retriever.bm25 is None
        assert not (cache / "bm25").exists()
    if with_graph:
        assert result["graph"]["summary"]["total"] == 1
        assert result["graph"]["nodes"][0].output.extractable
        assert result["graph"]["nodes"][0] is result["extraction"]["extracted_cases"][0]
        assert result["graph"]["embeddings"][0]["text"] == "resolved"
        assert (cache / "graph.json").exists()
    else:
        assert result["graph"] is None and not (cache / "graph.json").exists()
