import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from raft import ExtractedCase, embed_cases
from raft.embedding import BM25Index
from raft.embedding.openai import OpenAIEmbeddings
from raft.runtime import RetryDecision
from raft.storage import load_jsonl, save_embeddings, save_jsonl


class Extraction(BaseModel):
    timeline: list[str]


class Entry(BaseModel):
    narrative: str


class RichExtraction(BaseModel):
    timeline: list[Entry]


def state_to_text(state: Extraction) -> list[str]:
    assert isinstance(state, Extraction)
    return state.timeline


def narratives(state: RichExtraction) -> list[str]:
    assert isinstance(state, RichExtraction)
    return [entry.narrative for entry in state.timeline]


class FakeEmbeddings:
    name = "fake"
    model = "test-embedding"

    def __init__(self):
        self.calls = []

    async def embed(self, texts, *, input_type="document"):
        self.calls.append(texts)
        return [[float(len(text)), 1.0] for text in texts]

    def classify_error(self, exc):
        return RetryDecision(isinstance(exc, TimeoutError), type(exc).__name__)


def options(**kwargs):
    return {
        "backend": FakeEmbeddings(),
        "state_to_text": state_to_text,
        "output_type": Extraction,
        "rpm": 1000,
        "retries": 0,
        **kwargs,
    }


@pytest.mark.asyncio
async def test_should_embed_skips_before_text_conversion_and_can_embed_later():
    cases = [
        ExtractedCase(id=key, metadata={"product": "p"}, output=Extraction(timeline=[key]))
        for key in ("skip", "keep")
    ]
    checked, converted = [], []

    def eligible(state):
        checked.append(state)
        return state.timeline != ["skip"]

    def convert(state):
        converted.append(state)
        return state.timeline

    backend = FakeEmbeddings()
    result = await embed_cases(
        cases=cases, **options(backend=backend, should_embed=eligible, state_to_text=convert)
    )
    assert result["summary"] == {
        "total": 2,
        "embedded": 1,
        "skipped": 1,
        "failed": 0,
        "items": 1,
    }
    assert checked == [case.output for case in cases]
    assert converted == [cases[1].output]
    assert result["skipped_cases"][0] is cases[0]
    assert result["embedded_cases"][0]["case"] is cases[1]
    assert backend.calls == [["keep"]]
    # No predicate means embed every valid case, including previously skipped ones.
    again = await embed_cases(cases=result["skipped_cases"], **options(backend=backend))
    assert again["embedded_cases"][0]["case"] is cases[0]
    assert again["skipped_cases"] == []
    assert backend.calls == [["keep"], ["skip"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [None, 0, 1, "yes", [], RuntimeError("predicate bug")])
async def test_should_embed_errors_are_terminal_and_isolated(invalid):
    backend = FakeEmbeddings()

    def eligible(state):
        if state.timeline == ["bad"]:
            if isinstance(invalid, Exception):
                raise invalid
            return invalid
        return True

    result = await embed_cases(
        cases=[{"id": key, "metadata": {}, "output": {"timeline": [key]}} for key in ("bad", "ok")],
        **options(backend=backend, should_embed=eligible, retries=2),
    )
    assert backend.calls == [["ok"]]
    assert result["skipped_cases"] == []
    failure = result["failed_cases"][0]
    assert failure["error_category"] == "should_embed_error"
    assert failure["attempts"] == 0 and failure["retryable"] is False
    assert isinstance(failure["case"].output, Extraction)


@pytest.mark.asyncio
async def test_should_embed_restores_models_and_runs_once_across_retries(monkeypatch):
    checked = []

    def eligible(state):
        assert isinstance(state, Extraction)
        checked.append(state)
        return True

    class Flaky(FakeEmbeddings):
        async def embed(self, texts, *, input_type="document"):
            self.calls.append(texts)
            if len(self.calls) == 1:
                raise TimeoutError("temporary")
            return [[1.0] for _ in texts]

    backend = Flaky()
    monkeypatch.setattr("raft.embedding.runner._retry_delay", lambda *args: 0)
    result = await embed_cases(
        cases=[{"id": "a", "metadata": {}, "output": {"timeline": ["text"]}}],
        **options(backend=backend, should_embed=eligible, retries=1),
    )
    assert len(checked) == 1 and len(backend.calls) == 2
    assert checked[0] is result["embedded_cases"][0]["case"].output


@pytest.mark.asyncio
async def test_no_predicate_does_not_interpret_extractable_field():
    class Output(Extraction):
        extractable: bool

    case = ExtractedCase(id="rfi", metadata={}, output=Output(timeline=["RFI"], extractable=False))
    result = await embed_cases(cases=[case], **options(output_type=None))
    assert result["embedded_cases"][0]["case"] is case
    assert result["skipped_cases"] == []


@pytest.mark.asyncio
async def test_bm25_uses_successful_embedding_texts_without_reformatting(tmp_path):
    calls = []

    def convert(state):
        calls.append(state)
        return state.timeline

    class Backend(FakeEmbeddings):
        async def embed(self, texts, *, input_type="document"):
            if "failed case" in texts:
                raise ValueError("terminal failure")
            return await super().embed(texts, input_type=input_type)

    result = await embed_cases(
        cases=[
            {"metadata": {}, "id": "ok", "output": {"timeline": ["Login E42", "Reset password"]}},
            {"metadata": {}, "id": "bad", "output": {"timeline": ["failed case"]}},
            {"metadata": {}, "id": "empty", "output": {"timeline": []}},
        ],
        **options(
            backend=Backend(),
            state_to_text=convert,
        ),
    )
    assert len(calls) == 3
    assert result["summary"] == {"total": 3, "embedded": 2, "skipped": 0, "failed": 1, "items": 2}
    assert not list(tmp_path.iterdir())
    info = save_embeddings(
        result, output_path=tmp_path / "vectors.jsonl", bm25_path=tmp_path / "bm25"
    )
    assert info == {"path": str(tmp_path / "bm25"), "documents": 2}
    assert "bm25_index" not in result
    index = BM25Index.load(tmp_path / "bm25")
    vectors = load_jsonl(tmp_path / "vectors.jsonl")
    assert [(d["id"], d["text"]) for d in index.documents] == [
        (d["id"], d["text"]) for d in vectors
    ]
    assert index.search("failed") == []
    assert index.search("E42")[0]["id"] == vectors[0]["id"]


@pytest.mark.asyncio
async def test_bm25_empty_batch_without_vector_file(tmp_path):
    result = await embed_cases(cases=[], **options())
    info = save_embeddings(result, bm25_path=tmp_path / "bm25")
    assert info["documents"] == 0
    assert BM25Index.load(tmp_path / "bm25").search("anything") == []


@pytest.mark.asyncio
async def test_per_item_embedding_order_persistence_and_stable_ids(tmp_path):
    cases = [
        {
            "id": "a",
            "metadata": {"product": "sync"},
            "output": {
                "timeline": [{"narrative": "symptom", "ignored": 4}, {"narrative": "resolution"}]
            },
        },
        {"metadata": {}, "id": "b", "output": {"timeline": [{"narrative": "another"}]}},
    ]
    backend = FakeEmbeddings()
    path = tmp_path / "embeddings.jsonl"
    result = await embed_cases(
        cases=cases,
        **options(
            backend=backend,
            state_to_text=narratives,
            output_type=RichExtraction,
            batch_size=1,
        ),
    )
    assert result["summary"] == {"total": 2, "embedded": 2, "skipped": 0, "failed": 0, "items": 3}
    save_embeddings(result, output_path=path)
    records = load_jsonl(path)
    assert [(r["case_id"], r["item_index"], r["text"]) for r in records] == [
        ("a", 0, "symptom"),
        ("a", 1, "resolution"),
        ("b", 0, "another"),
    ]
    assert "metadata" not in records[0]
    assert result["embedded_cases"][0]["case"].metadata == {"product": "sync"}
    assert records[0]["embedding"] == [7.0, 1.0]
    assert len({row["id"] for row in records}) == 3
    again = await embed_cases(
        cases=cases,
        **options(state_to_text=narratives, output_type=RichExtraction),
    )
    save_embeddings(again, output_path=path)
    assert [r["id"] for r in load_jsonl(path)] == [r["id"] for r in records]


@pytest.mark.asyncio
async def test_custom_models_and_empty_lists():
    class CaseState(BaseModel):
        title: str
        events: list[Entry]

    received = []

    def format_state(state: CaseState) -> list[str]:
        received.append(state)
        return [f"{state.title}: {event.narrative}" for event in state.events]

    backend = FakeEmbeddings()
    result = await embed_cases(
        cases=[
            {
                "metadata": {},
                "id": "a",
                "output": {"title": "Login failure", "events": [{"narrative": "SSO fixed"}]},
            },
            {"metadata": {}, "id": "empty", "output": {"title": "Empty case", "events": []}},
        ],
        **options(backend=backend, output_type=CaseState, state_to_text=format_state),
    )
    assert backend.calls == [["Login failure: SSO fixed"]]
    assert all(isinstance(state, CaseState) for state in received)
    assert result["summary"]["embedded"] == 2
    assert result["embedded_cases"][1]["embeddings"] == []


@pytest.mark.asyncio
async def test_accepts_model_instances_without_reconstruction_type():
    backend = FakeEmbeddings()
    result = await embed_cases(
        cases=[{"metadata": {}, "id": "a", "output": Extraction(timeline=["done"])}],
        **options(backend=backend, output_type=None),
    )
    assert backend.calls == [["done"]]
    assert result["summary"]["embedded"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("returned", ["a single string", None, [42], [""], ["   "], ("tuple",)])
async def test_callback_contract_errors_are_terminal_and_never_call_provider(returned):
    backend = FakeEmbeddings()
    result = await embed_cases(
        cases=[{"metadata": {}, "id": "a", "output": {"timeline": []}}],
        **options(backend=backend, state_to_text=lambda state: returned, retries=2),
    )
    assert backend.calls == []
    failure = result["failed_cases"][0]
    assert failure["error_category"] == "state_to_text_error"
    assert failure["attempts"] == 0 and not failure["retryable"]


@pytest.mark.asyncio
async def test_callback_exception_is_isolated_to_case():
    def convert(state):
        if state.timeline == ["bad"]:
            raise RuntimeError("application formatter error")
        return state.timeline

    result = await embed_cases(
        cases=[
            {"metadata": {}, "id": text, "output": {"timeline": [text]}} for text in ["bad", "ok"]
        ],
        **options(state_to_text=convert),
    )
    assert result["summary"]["failed"] == 1
    assert result["summary"]["embedded"] == 1
    assert result["failed_cases"][0]["error_category"] == "state_to_text_error"


@pytest.mark.asyncio
async def test_invalid_cases_do_not_abort_batch():
    cases = [
        {"metadata": {}, "id": "good", "output": {"timeline": ["ok"]}},
        {"metadata": {}, "id": "good", "output": {"timeline": ["duplicate"]}},
        {"metadata": {}, "id": "bad", "output": {"timeline": None}},
        {"metadata": {}, "id": "blank", "output": {"timeline": [""]}},
        {"metadata": {}, "id": "missing", "output": {}},
    ]
    result = await embed_cases(cases=cases, **options())
    assert result["summary"]["embedded"] == 1 and result["summary"]["failed"] == 4
    assert all(item["attempts"] == 0 for item in result["failed_cases"])


@pytest.mark.asyncio
async def test_retry_resumes_failed_batch(monkeypatch):
    class Flaky(FakeEmbeddings):
        async def embed(self, texts, *, input_type="document"):
            self.calls.append(texts)
            if len(self.calls) == 2:
                raise TimeoutError("temporary")
            return [[1.0] for _ in texts]

    backend = Flaky()
    monkeypatch.setattr("raft.embedding.runner._retry_delay", lambda *args: 0)
    result = await embed_cases(
        cases=[{"metadata": {}, "id": "a", "output": {"timeline": ["a", "b", "c"]}}],
        **options(backend=backend, batch_size=1, retries=1),
    )
    assert backend.calls == [["a"], ["b"], ["b"], ["c"]]
    assert result["embedded_cases"][0]["attempts"] == 2


@pytest.mark.asyncio
async def test_concurrency_timeout_and_no_partial_publication(tmp_path):
    class Slow(FakeEmbeddings):
        active = maximum = 0

        async def embed(self, texts, *, input_type="document"):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            try:
                await asyncio.sleep(0.02)
                return await super().embed(texts, input_type=input_type)
            finally:
                self.active -= 1

    backend = Slow()
    result = await embed_cases(
        cases=[{"metadata": {}, "id": str(i), "output": {"timeline": ["a"]}} for i in range(5)],
        **options(backend=backend, concurrency=2),
    )
    assert backend.maximum == 2 and result["summary"]["embedded"] == 5
    path = tmp_path / "vectors.jsonl"
    result = await embed_cases(
        cases=[{"metadata": {}, "id": "slow", "output": {"timeline": ["a", "b", "c"]}}],
        **options(backend=backend, batch_size=1, timeout=0.03),
    )
    assert result["summary"]["failed"] == 1
    save_embeddings(result, output_path=path)
    assert load_jsonl(path) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("vectors", [[], [[1], [1, 2]], [[float("nan")], [1]], [[], []]])
async def test_rejects_malformed_provider_vectors(vectors):
    class Broken(FakeEmbeddings):
        async def embed(self, texts, *, input_type="document"):
            return vectors

    result = await embed_cases(
        cases=[{"metadata": {}, "id": "a", "output": {"timeline": ["a", "b"]}}],
        **options(backend=Broken()),
    )
    assert result["summary"]["failed"] == 1


def test_atomic_save_preserves_previous_file_on_error(tmp_path):
    path = tmp_path / "records.jsonl"
    save_jsonl(path, [{"old": True}])
    with pytest.raises(TypeError):
        save_jsonl(path, [{"first": 1}, {"bad": object()}])
    assert load_jsonl(path) == [{"old": True}]

@pytest.mark.asyncio
@pytest.mark.parametrize("input_type", ["document", "query"])
async def test_openai_client_arguments_and_index_mapping(input_type):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[2.0]),
                SimpleNamespace(index=0, embedding=[1.0]),
            ]
        )

    backend = OpenAIEmbeddings(
        client=SimpleNamespace(embeddings=SimpleNamespace(create=create)),
        model="custom-model",
        dimensions=1,
    )
    batch = await backend.embed(["a", "b"], input_type=input_type)
    assert batch.vectors == [[1.0], [2.0]]
    assert batch.usage == {}
    assert calls == [
        {"input": ["a", "b"], "model": "custom-model", "encoding_format": "float", "dimensions": 1}
    ]
