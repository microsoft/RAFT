"""Progress counts terminal outcomes, including errors, without changing scheduling."""

import asyncio
import io

import pytest
from agent_helpers import run_cases
from test_local_pipeline import Embeddings, pipeline, raw
from test_stages import Worker, cases, config

from raft import embed_cases
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
            created.append(self)

        def set_postfix(self, counts, **kwargs):
            self.counts = dict(counts)

        def update(self, n):
            self.n += n

        def close(self):
            self.closed = True

    monkeypatch.setattr(tqdm.auto, "tqdm", Bar)
    return created


@pytest.mark.asyncio
async def test_extraction_and_embedding_count_final_outcomes(bars):
    settings = config(Worker(), Embeddings())
    extraction = await run_cases(
        cases=cases("ok", "filtered", "scan-fails"),
        should_keep=lambda case: case.output.extractable,
        show_progress=True, **settings["extraction"],
    )
    assert bars[0].counts == {"succeeded": 1, "failed": 1, "filtered": 1}
    assert bars[0].n == bars[0].options["total"] == 3
    assert bars[0].closed
    records = [
        *extraction["extracted_cases"], *extraction["filtered_cases"],
        {"id": "invalid", "output": {}},
    ]
    await embed_cases(cases=records, show_progress=True, **settings["embedding"])
    assert bars[1].counts == {"succeeded": 1, "failed": 1, "skipped": 1}
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
    assert bars[-1].counts == {"succeeded": 1, "failed": 0}


def test_terminal_progress_renders_counters(monkeypatch):
    import tqdm.auto
    from tqdm import tqdm as terminal_tqdm

    stream = io.StringIO()
    monkeypatch.setattr(tqdm.auto, "tqdm", lambda **kwargs: terminal_tqdm(file=stream, **kwargs))
    with CaseProgress(2, enabled=True, desc="Extraction") as progress:
        progress.advance()
        progress.advance("failed")
    assert "succeeded=1" in stream.getvalue() and "failed=1" in stream.getvalue()
    assert "100%" in stream.getvalue()


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
