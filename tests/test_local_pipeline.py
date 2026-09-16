import asyncio
import json

import pytest
from agent_helpers import FakeReview, LocalPipeline
from pydantic import BaseModel

from raft import ExtractedCase
from raft.embedding import BM25Index, EmbeddingBatch
from raft.extraction.state import apply_edit
from raft.runtime import RetryDecision
from raft.storage import save_json, save_jsonl


class State(BaseModel):
    timeline: list[str]
    extractable: bool = True


class Worker(FakeReview):
    def __init__(self):
        self.calls = []
        self.fail = False

    def prepare(self, agent):
        return agent

    async def run(self, agent, prompt, *, context, max_turns, telemetry):
        self.calls.append(context.case_id)
        if self.fail:
            raise ValueError("extraction failed")
        telemetry.usage = {"test-model": {"input_tokens": 5}}
        assert apply_edit(
            context=context,
            patch_json=json.dumps(
                [
                    {
                        "op": "add",
                        "path": "",
                        "value": {
                            "timeline": context.metadata["texts"],
                            "extractable": context.metadata["keep"],
                        },
                    }
                ]
            ),
            finish_pass=True,
        )["ok"]

    def aggregate_usage(self, usages):
        return {"test-model": {"input_tokens": sum(u["test-model"].get("input_tokens", 0) for u in usages)}}

    def classify_error(self, exc):
        return RetryDecision(False, "error")


class Embeddings:
    name = "fake"
    model = "state"

    def __init__(self):
        self.calls = []
        self.fail = False

    async def embed(self, texts, *, input_type="document"):
        self.calls.append(list(texts))
        if self.fail:
            raise ValueError("embedding failed")
        return EmbeddingBatch(
            [[float(len(text)), 1.0] for text in texts], {"prompt_tokens": len(texts) * 3}
        )

    def classify_error(self, exc):
        return RetryDecision(False, "error")


def raw(id, texts=None, keep=True):
    return {"ticket": id, "metadata": {"texts": texts or [str(id)], "keep": keep}, "items": []}


def pipeline(path, worker=None, backend=None, **options):
    return LocalPipeline(
        path,
        extraction={
            "_agent_runner": worker or Worker(),
            "worker_agent": object(),
            "reviewer_agent": object(),
            "output_type": State,
            "id_field": "ticket",
            "metadata_field": "metadata",
            "artifacts_field": "items",
            "retries": 0,
            "rpm": 1000,
        },
        embedding={
            "backend": backend or Embeddings(),
            "state_to_text": lambda s: s.timeline,
            "should_embed": lambda s: s.extractable,
            "retries": 0,
            "rpm": 1000,
        },
        **options,
    )


@pytest.mark.asyncio
async def test_new_index_unchanged_incremental_reopen_and_usage(tmp_path):
    worker, backend = Worker(), Embeddings()
    path = tmp_path / "nested/index"
    p = pipeline(path, worker, backend)
    result = await p.index([raw("a", ["opening", "resolved"]), raw("b")])
    assert path.exists() and result["summary"]["stored_embedded"] == 2
    assert result["indexed_cases"][0]["case"].metadata["keep"]
    assert "usage" not in result["indexed_cases"][0]
    assert result["embedding"]["embedding_usage"] == {"prompt_tokens": 9}
    again = await p.index([raw("a", ["opening", "resolved"]), raw("b")])
    assert again["skipped_ids"] == ["a", "b"]
    assert again["embedding"]["embedding_usage"] == {}
    assert again["embedding"]["embedding_requests"] == 0
    assert len(worker.calls) == len(backend.calls) == 2
    update = await p.index([raw("c")])
    assert update["embedding"]["embedding_usage"] == {"prompt_tokens": 3}
    assert len(worker.calls) == len(backend.calls) == 3
    reopened = pipeline(path)
    results = await reopened.retrieve(["opening"], top_k=3)
    assert {h["id"] for h in results["results"][0]["candidates"]} == {"a", "b", "c"}
    assert all(isinstance(h["case"].output, State) for h in results["results"][0]["candidates"])
    assert results["embedding_usage"] == {"prompt_tokens": 3}
    assert results["embedding_requests"] == 1
    catalog = json.loads((path / "catalog.json").read_text())
    assert "usage" not in catalog["cases"]['"a"']["embedding"]
    assert catalog["cases"]['"a"']["case"]["execution"]["usage"]["test-model"]["input_tokens"] == 5


@pytest.mark.asyncio
async def test_pipeline_forwards_formatter_and_budgets_without_changing_saved_history(tmp_path):
    p = pipeline(tmp_path)
    await p.index([raw("a", ["opening", "resolved"]), raw("b")])
    catalog_before = (tmp_path / "catalog.json").read_bytes()
    result = (await p.retrieve(["opening"], top_k=1))["results"][0]
    hit = result["candidates"][0]
    assert json.loads(result["formatted_context"]) == {
        "id": hit["id"], "metadata": hit["case"].metadata,
        "output": hit["case"].output.model_dump(), "item_index": hit["item_index"],
    }
    assert "usage" in hit["case"].execution
    received = []

    def formatter(candidate):
        received.append(candidate)
        return f"{candidate['id']}:{candidate['item_index']}"

    reopened = pipeline(tmp_path)
    result = (await reopened.retrieve(
        ["opening"], top_k=1, max_chars=3, format_case=formatter,
    ))["results"][0]
    assert len(received) == 1
    assert result["formatted_context"] == f"{received[0]['id']}:{received[0]['item_index']}"
    assert result["used_chars"] == 3
    assert not result["truncated"] and result["error"] is None
    assert (tmp_path / "catalog.json").read_bytes() == catalog_before


@pytest.mark.asyncio
async def test_update_replaces_old_entries_and_skip_removes_vectors(tmp_path):
    p = pipeline(tmp_path)
    await p.index([raw("a", ["first", "obsolete"]), raw("b")])
    await p.index([raw("a", ["replacement"])], rewrite=True)
    assert [r["text"] for r in p._retriever.rows if r["case_id"] == "a"] == ["replacement"]
    result = await p.index([raw("a", ["RFI"], keep=False)], rewrite=True)
    assert result["summary"]["stored_skipped"] == 1
    assert {r["case_id"] for r in p._retriever.rows} == {"b"}
    assert result["embedding"]["skipped_cases"][0].output.timeline == ["RFI"]


@pytest.mark.asyncio
async def test_embedding_failure_resumes_after_reopen_without_reextracting(tmp_path):
    worker, backend = Worker(), Embeddings()
    backend.fail = True
    result = await pipeline(tmp_path, worker, backend).index([raw("a")])
    assert result["summary"]["stored_failed"] == 1
    assert len(worker.calls) == 1
    backend.fail = False
    result = await pipeline(tmp_path, worker, backend).index()
    assert len(worker.calls) == 1 and result["summary"]["stored_embedded"] == 1
    assert result["extraction"]["summary"]["total"] == 0


@pytest.mark.asyncio
async def test_failed_replacement_extraction_keeps_old_case(tmp_path):
    worker = Worker()
    p = pipeline(tmp_path, worker)
    await p.index([raw("a", ["old"])])
    worker.fail = True
    result = await p.index([raw("a", ["new"])], rewrite=True)
    assert result["extraction"]["summary"]["failed"] == 1
    hit = (await p.retrieve(["query"]))["results"][0]["candidates"][0]
    assert hit["case"].output.timeline == ["old"]


@pytest.mark.asyncio
async def test_changed_content_is_skipped_unless_rewrite(tmp_path):
    worker, backend = Worker(), Embeddings()
    p = pipeline(tmp_path, worker, backend)
    await p.index([raw("a"), raw("b")])
    result = await p.index([raw("a", ["changed"])])
    assert result["skipped_ids"] == ["a"]
    assert len(worker.calls) == len(backend.calls) == 2
    assert [r["text"] for r in p._retriever.rows] == ["a", "b"]
    await p.index([raw("a", ["changed"])], rewrite=True)
    assert worker.calls == ["a", "b", "a"]
    assert len(backend.calls) == 3
    assert [r["text"] for r in p._retriever.rows] == ["changed", "b"]


@pytest.mark.asyncio
async def test_different_embedding_space_rejected(tmp_path):
    await pipeline(tmp_path).index([raw("a")])
    backend = Embeddings()
    backend.model = "different"
    with pytest.raises(ValueError, match="same embedding model"):
        await pipeline(tmp_path, backend=backend).retrieve(["query"])
    assert not backend.calls


@pytest.mark.asyncio
async def test_bm25_rebuilds_full_corpus_only_after_updates(tmp_path, monkeypatch):
    builds = []
    original = BM25Index.from_records

    def build(records):
        rows = list(records)
        builds.append([r["text"] for r in rows])
        return original(rows)

    monkeypatch.setattr(BM25Index, "from_records", build)
    backend = Embeddings()
    p = pipeline(tmp_path, backend=backend)
    await p.index([raw("a", ["first", "obsolete"]), raw("b")])
    assert builds == [["first", "obsolete", "b"]]
    await p.index([raw("a", ["ignored"])])
    await pipeline(tmp_path, backend=backend).index()
    assert len(builds) == 1 and len(backend.calls) == 2
    await p.index([raw("c")])
    assert builds[-1] == ["first", "obsolete", "b", "c"]
    await p.index([raw("a", ["replacement"])], rewrite=True)
    assert builds[-1] == ["replacement", "b", "c"]
    assert len(builds) == 3 and len(backend.calls) == 4
    assert not p._retriever.bm25.search("obsolete")
    assert p._retriever.bm25.scores("replacement b") == original(p._retriever.rows).scores(
        "replacement b"
    )


@pytest.mark.asyncio
async def test_incomplete_bm25_cache_rebuilt_without_embedding(tmp_path):
    backend = Embeddings()
    p = pipeline(tmp_path, backend=backend)
    await p.index([raw("a")])
    cache = p.store.cache_dir(p.store.read()) / "bm25"
    save_json(cache / "raft.json", {"version": 1, "complete": False})
    reopened = pipeline(tmp_path, backend=backend)
    await reopened.index()
    assert len(backend.calls) == 1
    assert BM25Index.load(cache).documents == reopened._retriever.bm25.documents


@pytest.mark.asyncio
async def test_legacy_extraction_only_resume_and_existing_vectors_import(tmp_path):
    worker, backend = Worker(), Embeddings()
    case = ExtractedCase(id="a", metadata={}, output=State(timeline=["text"]))
    save_json(tmp_path / "extraction.json", {"extracted_cases": [case]})
    result = await pipeline(tmp_path, worker, backend).index()
    assert not worker.calls and result["summary"]["stored_embedded"] == 1
    other = tmp_path / "other"
    save_json(other / "extraction.json", {"extracted_cases": [case]})
    save_jsonl(other / "embeddings.jsonl", result["indexed_cases"][0]["embeddings"])
    count = len(backend.calls)
    await pipeline(other, worker, backend).index()
    assert len(backend.calls) == count and not worker.calls


@pytest.mark.asyncio
async def test_graph_refresh_and_reuse_and_char_filter_retrieval(tmp_path):
    graph_backend = Embeddings()
    graph_backend.model = "graph"
    graph = {
        "backend": graph_backend,
        "case_to_text": lambda s: s.timeline[-1],
        "top_k": 1,
        "rpm": 1000,
    }
    p = pipeline(tmp_path, graph=graph)
    await p.index([raw("a"), raw("b")])
    assert len(graph_backend.calls) == 1
    await pipeline(tmp_path, graph=graph).index()
    assert len(graph_backend.calls) == 1
    await p.index([raw("c")])
    assert graph_backend.calls == [["a", "b"], ["a", "b", "c"]]
    neighbors = await p.graph_expansion(["a"], per_case_top_k=1)
    assert len(neighbors) == 1 and neighbors[0]["seed_id"] == "a"
    assert len(neighbors[0]["neighbors"]) == 1
    assert len(graph_backend.calls) == 2  # Full graph rebuild on corpus change.
    result = (await p.retrieve(["query"], top_k=2))["results"][0]
    assert len(result["candidates"]) == 2
    one = (await p.retrieve(["query"], case_filter=lambda q, c: c.id == "b"))["results"][0]
    assert [h["id"] for h in one["candidates"]] == ["b"]
    assert (await p.retrieve(["query"], max_chars=0))["results"][0]["candidates"] == []


@pytest.mark.asyncio
async def test_graph_expansion_reopens_saved_graph_without_models_or_bm25(tmp_path, monkeypatch):
    worker, backend, graph_backend = Worker(), Embeddings(), Embeddings()
    graph_backend.model = "graph"
    graph = {
        "backend": graph_backend,
        "case_to_text": lambda s: s.timeline[-1],
        "top_k": 2,
        "rpm": 1000,
    }
    p = pipeline(tmp_path, worker, backend, graph=graph, bm25=False)
    await p.index([raw(id) for id in ("a", "b", "x", "y")])
    cache = p.store.cache_dir(p.store.read()) / "graph.json"
    saved = json.loads(cache.read_text())
    # A fixed saved topology isolates expansion from graph-construction ranking.
    saved["result"]["edges"] = [
        {"source": "a", "target": "b", "weight": 1},
        {"source": "a", "target": "x", "weight": 0.9},
        {"source": "a", "target": "y", "weight": 0.6},
        {"source": "b", "target": "y", "weight": 0.8},
        {"source": "b", "target": "x", "weight": 0.7},
    ]
    save_json(cache, saved)
    counts = (len(worker.calls), len(backend.calls), len(graph_backend.calls))
    worker.fail = backend.fail = graph_backend.fail = True

    def unexpected_build(*args, **kwargs):
        raise AssertionError("Graph lookup must not build derived indexes")

    monkeypatch.setattr("raft.pipeline.build_case_graph", unexpected_build)
    monkeypatch.setattr(BM25Index, "from_records", unexpected_build)
    # No graph configuration is needed to use the saved current-revision graph.
    reopened = pipeline(tmp_path, worker, backend)
    adjacency = await reopened.graph_adjacency()
    assert adjacency == {
        "a": {"b": 1.0, "x": 0.9, "y": 0.6},
        "b": {"a": 1.0, "y": 0.8, "x": 0.7},
        "x": {"a": 0.9, "b": 0.7},
        "y": {"a": 0.6, "b": 0.8},
    }
    original_cache = cache.read_bytes()
    adjacency["a"]["x"] = 0.01
    adjacency["b"].clear()
    adjacency.pop("y")
    fresh = await reopened.graph_adjacency()
    assert fresh["a"]["x"] == 0.9 and fresh["b"]["y"] == 0.8
    assert "y" in fresh
    assert cache.read_bytes() == original_cache
    assert await reopened.graph_expansion(["a", "b", "a"], per_case_top_k=1) == [
        {"seed_id": "a", "neighbors": [{"id": "x", "weight": 0.9}]},
        {"seed_id": "b", "neighbors": [{"id": "y", "weight": 0.8}]},
    ]
    assert await reopened.graph_expansion(["a", "b"], per_case_top_k=2) == [
        {
            "seed_id": "a",
            "neighbors": [{"id": "x", "weight": 0.9}, {"id": "y", "weight": 0.6}],
        },
        {
            "seed_id": "b",
            "neighbors": [{"id": "y", "weight": 0.8}, {"id": "x", "weight": 0.7}],
        },
    ]
    assert await reopened.graph_expansion(
        ["a", "b"], per_case_top_k=1, allowed_ids={"y"}
    ) == [
        {"seed_id": "a", "neighbors": [{"id": "y", "weight": 0.6}]},
        {"seed_id": "b", "neighbors": [{"id": "y", "weight": 0.8}]},
    ]
    assert await reopened.graph_expansion([]) == []
    assert await reopened.graph_expansion(["unknown"]) == [
        {"seed_id": "unknown", "neighbors": []}
    ]
    assert await reopened.graph_expansion(["a"], allowed_ids=[]) == [
        {"seed_id": "a", "neighbors": []}
    ]
    assert (len(worker.calls), len(backend.calls), len(graph_backend.calls)) == counts
    assert not list(tmp_path.glob("indexes/*/bm25"))


@pytest.mark.asyncio
@pytest.mark.parametrize("availability", ["missing", "incomplete", "incompatible", "stale"])
async def test_graph_expansion_requires_complete_compatible_current_graph(tmp_path, availability):
    worker, backend, graph_backend = Worker(), Embeddings(), Embeddings()
    graph_backend.model = "graph"
    graph = {
        "backend": graph_backend,
        "case_to_text": lambda s: s.timeline[-1],
        "top_k": 1,
        "rpm": 1000,
    }
    p = pipeline(tmp_path, worker, backend, graph=graph, bm25=False)
    await p.index([raw("a"), raw("b")])
    catalog = p.store.read()
    cache = p.store.cache_dir(catalog) / "graph.json"
    if availability == "missing":
        cache.unlink()
    elif availability == "incomplete":
        saved = json.loads(cache.read_text())
        saved["result"]["failed_cases"] = [{"id": "a", "error": "graph embedding failed"}]
        save_json(cache, saved)
    elif availability == "incompatible":
        graph = {**graph, "top_k": 2}
    else:
        p.store.save(catalog)
    counts = (len(worker.calls), len(backend.calls), len(graph_backend.calls))
    reopened = pipeline(tmp_path, worker, backend, graph=graph)
    with pytest.raises(ValueError, match=r"index\("):
        await reopened.graph_expansion(["a"])
    assert (len(worker.calls), len(backend.calls), len(graph_backend.calls)) == counts
    assert not list(tmp_path.glob("indexes/*/bm25"))


@pytest.mark.asyncio
async def test_pipeline_retrieve_rejects_graph_slots_without_embedding(tmp_path):
    backend = Embeddings()
    p = pipeline(tmp_path, backend=backend)
    await p.index([raw("a")])
    calls = len(backend.calls)
    with pytest.raises(TypeError, match="graph_slots"):
        await p.retrieve(["query"], graph_slots=1)
    assert len(backend.calls) == calls


@pytest.mark.asyncio
async def test_pipeline_query_batching_forwards_configuration_and_aggregate_usage(tmp_path):
    backend = Embeddings()
    p = pipeline(tmp_path, backend=backend)
    await p.index([raw("a"), raw("b")])
    backend.calls.clear()
    queries = ["first", "second", "first", "fourth", "last"]
    response = await p.retrieve(queries, batch_size=2, concurrency=1, rpm=1000)
    assert backend.calls == [queries[:2], queries[2:4], queries[4:]]
    assert response["embedding_usage"] == {"prompt_tokens": 15}
    assert response["embedding_requests"] == 3
    assert [result["query"] for result in response["results"]] == queries
    assert all("usage" not in result for result in response["results"])
    assert all(result["requests"] == 1 for result in response["results"])

    backend.calls.clear()
    default = await p.retrieve(queries, rpm=1000)
    assert backend.calls == [queries]
    assert default["embedding_usage"] == {"prompt_tokens": 15}
    assert default["embedding_requests"] == 1
    filtered = await p.retrieve(queries, case_filter=lambda query, case: False)
    assert filtered["embedding_usage"] == {} and filtered["embedding_requests"] == 0
    assert backend.calls == [queries]


@pytest.mark.asyncio
async def test_cancel_leaves_extraction_checkpoint_and_releases_writer(tmp_path):
    started = asyncio.Event()

    class Waiting(Embeddings):
        async def embed(self, texts, *, input_type="document"):
            started.set()
            await asyncio.Event().wait()

    worker = Worker()
    task = asyncio.create_task(pipeline(tmp_path, worker, Waiting()).index([raw("a")]))
    await started.wait()
    with pytest.raises(RuntimeError, match="busy"):
        await pipeline(tmp_path).index()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = await pipeline(tmp_path, worker).index()
    assert len(worker.calls) == 1 and result["summary"]["stored_embedded"] == 1


@pytest.mark.asyncio
async def test_no_bm25_empty_and_typed_ids(tmp_path):
    p = pipeline(tmp_path, bm25=False)
    assert (await p.retrieve(["query"]))["results"][0]["candidates"] == []
    result = await p.index([raw(1), raw("1")])
    assert result["summary"]["stored_embedded"] == 2
    assert p._retriever.bm25 is None
    result = await p.retrieve(["query"])
    assert [h["id"] for h in result["results"][0]["candidates"]] in ([1, "1"], ["1", 1])


@pytest.mark.asyncio
async def test_duplicate_inputs_rejected_before_agents_and_rewrite(tmp_path):
    worker, backend = Worker(), Embeddings()
    p = pipeline(tmp_path, worker, backend)
    with pytest.raises(ValueError, match="unique"):
        await p.index([raw("a"), raw("a")])
    assert not worker.calls
    await p.index([raw("a")])
    await p.index([raw("a")], rewrite=True)
    assert len(worker.calls) == len(backend.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [[], ["isolated"], [1, "1", "isolated"]])
async def test_graph_adjacency_includes_isolates_and_preserves_typed_ids(tmp_path, ids):
    backend = Embeddings()
    p = pipeline(tmp_path, graph={
        "backend": backend, "case_to_text": lambda state: state.timeline[-1], "rpm": 1000,
    })
    await p.index([raw(id) for id in ids])
    cache = p.store.cache_dir(p.store.read()) / "graph.json"
    saved = json.loads(cache.read_text())
    saved["result"]["edges"] = (
        [{"source": 1, "target": "1", "weight": 0}] if len(ids) > 1 else []
    )
    save_json(cache, saved)
    calls = len(backend.calls)
    adjacency = await p.graph_adjacency()
    assert set(adjacency) == set(ids)
    if len(ids) > 1:
        assert adjacency == {1: {"1": 0.0}, "1": {1: 0.0}, "isolated": {}}
    else:
        assert adjacency == {id: {} for id in ids}
    assert len(backend.calls) == calls


@pytest.mark.asyncio
async def test_graph_adjacency_requires_existing_catalog(tmp_path):
    p = pipeline(tmp_path / "uninitialized")
    with pytest.raises(ValueError, match=r"index\(\)"):
        await p.graph_adjacency()
    assert not p.store.path.exists()
