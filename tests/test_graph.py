import asyncio
import json
import threading

import pytest
from pydantic import BaseModel

from raft import ExtractedCase, build_case_graph
from raft.defaults import CaseExtraction, case_to_text
from raft.embedding import BM25Index, EmbeddingBatch
from raft.graph import link_cases as link_case_records
from raft.runtime import RetryDecision
from raft.storage import load_jsonl


class State(BaseModel):
    cause: str
    entities: list[str] = []


class Backend:
    name = "fake"
    model = "case-embedding"

    def __init__(self):
        self.calls = []

    async def embed(self, texts, *, input_type="document"):
        self.calls.append(texts)
        if texts == ["provider failure"]:
            raise ValueError("terminal provider failure")
        return [[1.0, float(len(text))] for text in texts]

    def classify_error(self, exc):
        return RetryDecision(isinstance(exc, TimeoutError), "provider_error")


def node(identifier, text=None, vector=None, product="p", entities=None):
    return {
        "id": identifier,
        "case_id": identifier,
        "item_index": 0,
        "text": identifier if text is None else text,
        "embedding": [1.0, 0.0] if vector is None else vector,
        "metadata": {"product": product},
        "output": State(cause=identifier, entities=entities or []),
        "provider": "fake",
        "model": "test",
    }


def neighbor_ids(result, source):
    return [n["id"] for row in result["neighbors"] if row["id"] == source for n in row["neighbors"]]


def link_cases(nodes, embeddings=None, **kwargs):
    # Compact fixture helper; production vectors and cases are separate records.
    if embeddings is None:
        embeddings = [
            {k: n[k] for k in ("id", "text", "embedding", "provider", "model")} for n in nodes
        ]
    return link_case_records(nodes, embeddings, **kwargs)


def test_undirected_union_jaccard_weights_and_order_independence():
    nodes = [node(letter) for letter in "abcd"]
    result = link_cases(nodes, top_k=2)
    assert neighbor_ids(result, "a") == ["b", "c"]
    assert neighbor_ids(result, "d") == ["a", "b"]
    assert result["summary"] == {"nodes": 4, "directed_links": 8, "edges": 5}
    edges = {(e["source"], e["target"]): e for e in result["edges"]}
    assert ("c", "d") not in edges
    assert edges["a", "b"]["shared_neighbors"] == 1
    assert edges["a", "b"]["neighbor_union"] == 3
    assert edges["a", "b"]["weight"] == pytest.approx(1 / 3)
    assert edges["a", "b"]["mutual"]
    assert not edges["a", "d"]["mutual"]
    assert result["edges"] == link_cases(list(reversed(nodes)), top_k=2)["edges"]


def test_zero_weight_edges_retained_and_self_links_excluded():
    result = link_cases([node("a"), node("b")], top_k=10)
    assert len(result["edges"]) == 1
    assert result["edges"][0]["weight"] == 0.0
    assert result["edges"][0]["shared_neighbors"] == 0
    assert neighbor_ids(result, "a") == ["b"]
    assert link_cases([], top_k=1)["edges"] == []
    single = link_cases([node("a")], top_k=1)
    assert single["summary"] == {"nodes": 1, "directed_links": 0, "edges": 0}


def test_filter_before_ranking_with_entity_overlap():
    nodes = [
        node("a", "login", entities=["sso"]),
        node("b", "login", product="excluded", entities=["sso"]),
        node("c", "certificate", vector=[0.0, 1.0], entities=["sso"]),
        node("d", "login", entities=["upload"]),
    ]

    def predicate(source, candidate):
        return source.metadata["product"] == candidate.metadata["product"] and bool(
            set(source.output.entities) & set(candidate.output.entities)
        )

    result = link_cases(nodes, top_k=1, neighbor_filter=predicate)
    assert neighbor_ids(result, "a") == ["c"]  # b/d would win without filtering
    assert neighbor_ids(result, "d") == []


def test_asymmetric_filter_still_forms_union_edge():
    result = link_cases(
        [node("a"), node("b")], neighbor_filter=lambda source, candidate: source.id == "a"
    )
    assert neighbor_ids(result, "b") == []
    assert len(result["edges"]) == 1 and not result["edges"][0]["mutual"]


def test_rrf_uses_dense_and_positive_lexical_ranks():
    nodes = [node("a", "login", [1, 0]), node("b", "other", [1, 0]), node("c", "login", [0, 1])]
    result = link_cases(nodes, top_k=1)
    hit = result["neighbors"][0]["neighbors"][0]
    assert hit["id"] == "c"  # lexical rank 1 plus dense rank 2 beats dense-only rank 1
    assert hit["rrf_score"] == pytest.approx(1 / 61 + 1 / 62)
    assert hit["cosine_similarity"] == 0 and hit["bm25_score"] > 0


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 1], [], [1, 2, 3]])
def test_rejects_invalid_vectors(vector):
    with pytest.raises(ValueError):
        link_cases([node("a"), node("b", vector=vector)])


def test_rejects_mixed_models_duplicate_ids_and_misaligned_index():
    with pytest.raises(ValueError, match="same embedding"):
        link_cases([node("a"), {**node("b"), "model": "different"}])
    with pytest.raises(ValueError, match="unique"):
        link_cases([node("a"), node("a")])
    with pytest.raises(ValueError, match="BM25 corpus"):
        link_cases([node("a")], bm25_index=BM25Index.from_records([node("b")]))


@pytest.mark.asyncio
async def test_runner_callback_once_failures_saved_outputs_and_rebuild(tmp_path):
    backend = Backend()
    calls = []

    def convert(state):
        assert isinstance(state, State)
        calls.append(state.cause)
        return state.cause

    result = await build_case_graph(
        cases=[
            {"id": str(i), "metadata": {"product": "p"}, "output": {"cause": cause}}
            for i, cause in enumerate(["login failure", "login expired", "", "provider failure"])
        ],
        backend=backend,
        case_to_text=convert,
        output_type=State,
        top_k=1,
        batch_size=1,
        retries=0,
        rpm=1000,
    )
    assert len(calls) == 4 and len(backend.calls) == 3
    assert result["summary"] == {
        "total": 4,
        "nodes": 2,
        "directed_links": 2,
        "edges": 1,
        "failed": 2,
    }
    assert {f["error_category"] for f in result["failed_cases"]} == {
        "case_to_text_error",
        "provider_error",
    }
    nodes = result["nodes"]
    assert not list(tmp_path.iterdir())
    from raft.storage import save_graph

    save_graph(result, tmp_path)
    assert load_jsonl(tmp_path / "nodes.jsonl") == [{"id": n.id} for n in nodes]
    assert nodes[0].output.cause == "login failure"
    assert len({n.id for n in nodes}) == 2
    index = BM25Index.load(tmp_path / "bm25")
    embeddings = load_jsonl(tmp_path / "embeddings.jsonl")
    assert [(d["id"], d["text"]) for d in index.documents] == [
        (r["id"], r["text"]) for r in embeddings
    ]
    assert all("metadata" not in row and "output" not in row for row in embeddings)
    assert result["edges"] == load_jsonl(tmp_path / "edges.jsonl")
    assert result["neighbors"] == load_jsonl(tmp_path / "neighbors.jsonl")
    assert result["edges"] == link_cases(nodes, embeddings, top_k=1, bm25_index=index)["edges"]
    assert json.loads((tmp_path / "report.json").read_text())["summary"] == result["summary"]


@pytest.mark.asyncio
async def test_retry_does_not_reinvoke_case_to_text(monkeypatch):
    class Flaky(Backend):
        async def embed(self, texts, *, input_type="document"):
            if not self.calls:
                self.calls.append(texts)
                raise TimeoutError("temporary")
            return await super().embed(texts, input_type=input_type)

    calls = []

    def convert(state):
        calls.append(state)
        return state.cause

    monkeypatch.setattr("raft.embedding._batching._retry_delay", lambda *args: 0)
    backend = Flaky()
    result = await build_case_graph(
        cases=[{"metadata": {}, "id": "a", "output": State(cause="test")}],
        backend=backend,
        case_to_text=convert,
        batch_size=1,
        retries=1,
    )
    assert len(calls) == 1 and len(backend.calls) == 2
    assert result["summary"]["nodes"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("returned", [None, [], ["text"], 1, "", "   "])
async def test_case_to_text_requires_single_nonempty_string(returned):
    backend = Backend()
    result = await build_case_graph(
        cases=[{"metadata": {}, "id": "a", "output": State(cause="test")}],
        backend=backend,
        case_to_text=lambda state: returned,
    )
    assert not backend.calls
    assert result["failed_cases"][0]["error_category"] == "case_to_text_error"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {"top_k": 0},
        {"rrf_constant": 0},
        {"neighbor_filter": "no_such_column = 1"},
        {"neighbor_filter": 123},
        {"batch_size": 0},
        {"batch_size": True},
        {"batch_size": 1.5},
    ],
)
async def test_invalid_configuration_fails_before_api_requests(options):
    backend = Backend()
    with pytest.raises(ValueError):
        await build_case_graph(
            cases=[{"metadata": {}, "id": "a", "output": State(cause="test")}],
            backend=backend,
            case_to_text=lambda state: state.cause,
            **options,
        )
    assert not backend.calls


def test_default_case_to_text():
    narrative = (
        "AUTH-401 blocked portal login after the SSO certificate expired. Support "
        "checked the certificate validity, renewed the expired certificate, and "
        "the customer confirmed that portal login worked again with the new certificate."
    )
    state = CaseExtraction(
        entities=["AUTH-401"],
        timeline=[narrative],
        root_cause="expired certificate",
        resolution_steps="renewed certificate",
    )
    assert case_to_text(state) == "expired certificate\nrenewed certificate"
    state.root_cause = state.resolution_steps = None
    assert case_to_text(state) == narrative


@pytest.mark.asyncio
async def test_custom_model_duplicate_ids_and_model_failure_report(tmp_path):
    cases = [
        {"metadata": {}, "id": 1, "output": {"cause": "login"}},
        {"metadata": {}, "id": "1", "output": {"cause": "upload"}},
        {"metadata": {}, "id": 1, "output": {"cause": "duplicate"}},
    ]
    result = await build_case_graph(
        cases=cases,
        backend=Backend(),
        output_type=State,
        case_to_text=lambda state: state.cause,
    )
    assert result["summary"]["nodes"] == 2 and result["summary"]["failed"] == 1
    assert len({node.id for node in result["nodes"]}) == 2
    assert result["nodes"][0].output.cause == "login"
    result = await build_case_graph(
        cases=[{"metadata": {}, "id": "bad", "output": State(cause="")}],
        backend=Backend(),
        case_to_text=lambda state: state.cause,
    )
    assert result["summary"]["nodes"] == 0
    from raft.storage import save_graph

    save_graph(result, tmp_path)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["failed_cases"][0]["case"]["output"]["cause"] == ""
    assert load_jsonl(tmp_path / "edges.jsonl") == []


@pytest.mark.asyncio
async def test_graph_embedding_concurrency_timeout_and_filter_worker_thread():
    class Slow(Backend):
        active = maximum = 0

        async def embed(self, texts, *, input_type="document"):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            try:
                await asyncio.sleep(0.01 if texts != ["slow"] else 0.5)
                return await super().embed(texts, input_type=input_type)
            finally:
                self.active -= 1

    thread_ids = []
    main_thread = threading.get_ident()

    def allow(source, candidate):
        thread_ids.append(threading.get_ident())
        return True

    backend = Slow()
    result = await build_case_graph(
        cases=[
            {"metadata": {}, "id": str(i), "output": State(cause=cause)}
            for i, cause in enumerate(["login", "upload", "slow"])
        ],
        backend=backend,
        case_to_text=lambda state: state.cause,
        batch_size=1,
        concurrency=2,
        timeout=0.2,
        retries=0,
        rpm=1000,
        neighbor_filter=allow,
    )
    assert backend.maximum == 2
    assert result["summary"]["nodes"] == 2 and result["summary"]["failed"] == 1
    assert thread_ids and all(t != main_thread for t in thread_ids)


@pytest.mark.asyncio
async def test_graph_batches_summaries_preserving_case_mapping_metadata_and_total_usage():
    class Metered(Backend):
        async def embed(self, texts, *, input_type="document"):
            assert input_type == "document"
            self.calls.append(list(texts))
            return EmbeddingBatch(
                [[float(len(text)), 1.0] for text in texts],
                {"prompt_tokens": sum(map(len, texts)), "total_tokens": sum(map(len, texts))},
            )

    first = ExtractedCase(id=1, metadata={"product": "p"}, output=State(cause="a"))
    last = ExtractedCase(id="c", metadata={"product": "q"}, output=State(cause="xyz"))
    seen = []

    def convert(state):
        seen.append(state.cause)
        return state.cause

    backend = Metered()
    result = await build_case_graph(
        cases=[
            first,
            {"id": "blank", "metadata": {}, "output": State(cause="")},
            {"id": "1", "metadata": {"product": "p"}, "output": {"cause": "long"}},
            {"id": 1, "metadata": {}, "output": State(cause="duplicate")},
            last,
            {"id": "invalid", "metadata": [], "output": State(cause="invalid")},
        ],
        backend=backend,
        case_to_text=convert,
        output_type=State,
        batch_size=2,
        concurrency=1,
        rpm=1000,
        neighbor_filter=lambda source, target: (
            source.metadata["product"] == target.metadata["product"]
        ),
    )
    assert seen == ["a", "", "long", "xyz"]
    assert backend.calls == [["a", "long"], ["xyz"]]
    assert result["nodes"][0] is first and result["nodes"][2] is last
    assert [n.id for n in result["nodes"]] == [1, "1", "c"]
    assert [(e["id"], e["text"], e["embedding"]) for e in result["embeddings"]] == [
        (1, "a", [1.0, 1.0]), ("1", "long", [4.0, 1.0]), ("c", "xyz", [3.0, 1.0]),
    ]
    assert neighbor_ids(result, "c") == []
    assert result["embedding_usage"] == {"prompt_tokens": 8, "total_tokens": 8}
    assert result["embedding_requests"] == 2
    assert result["embedding_summary"] == {
        "total": 6, "embedded": 3, "skipped": 0, "failed": 3, "items": 3,
    }
    assert [(f["id"], f["error_category"]) for f in result["failed_cases"]] == [
        ("blank", "case_to_text_error"), (1, "invalid_case"), ("invalid", "invalid_case"),
    ]
    assert all(f["attempts"] == f["requests"] == 0 for f in result["failed_cases"])


@pytest.mark.asyncio
async def test_batch_retry_preserves_summary_preparation_and_counts_actual_responses(monkeypatch):
    class Flaky(Backend):
        async def embed(self, texts, *, input_type="document"):
            self.calls.append(list(texts))
            if len(self.calls) == 1:
                raise TimeoutError("temporary")
            return EmbeddingBatch([[1.0, float(len(text))] for text in texts], {"total_tokens": 11})

    seen = []

    def convert(state):
        seen.append(state.cause)
        return state.cause

    monkeypatch.setattr("raft.embedding._batching._retry_delay", lambda *args: 0)
    backend = Flaky()
    result = await build_case_graph(
        cases=[
            {"id": id, "metadata": {}, "output": State(cause=id)} for id in ("a", "bb")
        ],
        backend=backend,
        case_to_text=convert,
        batch_size=2,
        retries=1,
        rpm=1000,
    )
    assert seen == ["a", "bb"]
    assert backend.calls == [["a", "bb"], ["a", "bb"]]
    assert result["embedding_requests"] == 2
    assert result["embedding_usage"] == {"total_tokens": 11}
    assert result["summary"]["nodes"] == 2 and not result["failed_cases"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 403])
async def test_graph_batch_failure_isolates_bad_input_but_does_not_split_auth_errors(status):
    class ProviderError(ValueError):
        def __init__(self):
            super().__init__("rejected input" if status == 400 else "access denied")
            self.status_code = status

    class Rejecting(Backend):
        def classify_error(self, exc):
            if isinstance(exc, ProviderError):
                return RetryDecision(False, f"http_{status}")
            return super().classify_error(exc)

        async def embed(self, texts, *, input_type="document"):
            self.calls.append(list(texts))
            if "bad" in texts:
                raise ProviderError()
            return EmbeddingBatch([[1.0, float(len(text))] for text in texts], {"total_tokens": 7})

    seen = []

    def convert(state):
        seen.append(state.cause)
        return state.cause

    backend = Rejecting()
    result = await build_case_graph(
        cases=[
            {"id": id, "metadata": {}, "output": State(cause=id)}
            for id in ("good-a", "bad", "good-b")
        ],
        backend=backend,
        case_to_text=convert,
        batch_size=3,
        retries=0,
        rpm=1000,
    )
    assert seen == ["good-a", "bad", "good-b"]
    assert result["embedding_requests"] == len(backend.calls)
    if status == 400:
        assert [n.id for n in result["nodes"]] == ["good-a", "good-b"]
        assert [f["id"] for f in result["failed_cases"]] == ["bad"]
        successes = sum("bad" not in call for call in backend.calls)
        assert result["embedding_usage"] == {"total_tokens": successes * 7}
    else:
        assert not result["nodes"] and len(result["failed_cases"]) == 3
        assert backend.calls == [["good-a", "bad", "good-b"]]
        assert result["embedding_usage"] == {}
    assert all("usage" not in f for f in result["failed_cases"])


@pytest.mark.asyncio
async def test_graph_retains_shared_usage_when_response_vectors_are_invalid():
    class Broken(Backend):
        async def embed(self, texts, *, input_type="document"):
            self.calls.append(list(texts))
            return EmbeddingBatch([], {"prompt_tokens": 9})

    backend = Broken()
    result = await build_case_graph(
        cases=[{"id": id, "metadata": {}, "output": State(cause=id)} for id in ("a", "b")],
        backend=backend,
        case_to_text=lambda state: state.cause,
        batch_size=2,
        retries=0,
        rpm=1000,
    )
    assert not result["nodes"] and len(result["failed_cases"]) == 2
    assert result["embedding_requests"] == 1
    assert result["embedding_usage"] == {"prompt_tokens": 9}
    assert all("usage" not in f for f in result["failed_cases"])
