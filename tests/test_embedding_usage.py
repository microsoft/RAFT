import asyncio
from types import SimpleNamespace

import pytest
from openai.types import CreateEmbeddingResponse
from pydantic import BaseModel

from raft import ExtractedCase, embed_cases
from raft.embedding import EmbeddingBatch
from raft.embedding.openai import OpenAIEmbeddings
from raft.embedding.voyage import VoyageEmbeddings
from raft.runtime import RetryDecision


class State(BaseModel):
    texts: list[str]


def case(id, texts):
    return ExtractedCase(id=id, metadata={}, output=State(texts=texts))


def options(backend, **overrides):
    return dict(
        backend=backend,
        state_to_text=lambda s: s.texts,
        rpm=1000,
        retries=0,
        batch_size=1,
        **overrides,
    )


@pytest.mark.asyncio
async def test_native_voyage_usage_sums_all_cases_and_batches_concurrently():
    async def create(**kwargs):
        text = kwargs["texts"][0]
        await asyncio.sleep(0.001 if text == "a" else 0)
        tokens = {"a": 3, "b": 5, "c": 11}[text]
        return SimpleNamespace(embeddings=[[1.0]], total_tokens=tokens)

    backend = VoyageEmbeddings(client=SimpleNamespace(embed=create), model="fake")
    first, second = case("first", ["a", "b"]), case("second", ["c"])
    result = await embed_cases(cases=[first, second], **options(backend))
    one, two = result["embedded_cases"]
    assert "usage" not in one and "usage" not in two
    assert one["requests"] == 2 and two["requests"] == 1
    assert result["embedding_usage"] == {"total_tokens": 19}
    assert result["embedding_requests"] == 3
    assert first.execution["usage"] == {}  # Embedding never overwrites extraction usage.


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", [False, True])
async def test_usage_survives_later_failure_or_retry_without_double_counting(monkeypatch, retry):
    class Backend:
        name = model = "fake"
        calls = 0

        async def embed(self, texts, *, input_type="document"):
            self.calls += 1
            if self.calls == 2:
                raise TimeoutError("no response usage")
            return EmbeddingBatch([[1.0]], {"prompt_tokens": 7, "total_tokens": 7})

        def classify_error(self, exc):
            return RetryDecision(True, "timeout")

    monkeypatch.setattr("raft.embedding.runner._retry_delay", lambda *args: 0)
    config = options(Backend())
    config["retries"] = int(retry)
    result = await embed_cases(cases=[case("a", ["first", "second"])], **config)
    item = result["embedded_cases" if retry else "failed_cases"][0]
    tokens = 14 if retry else 7
    assert "usage" not in item
    assert item["requests"] == (3 if retry else 2)
    assert item["attempts"] == (2 if retry else 1)
    assert result["embedding_usage"] == {"prompt_tokens": tokens, "total_tokens": tokens}
    assert result["embedding_requests"] == item["requests"]


@pytest.mark.asyncio
async def test_usage_retained_when_returned_vectors_fail_validation():
    class Backend:
        name = model = "fake"

        async def embed(self, texts, *, input_type="document"):
            return EmbeddingBatch([[float("nan")]], {"prompt_tokens": 9})

        def classify_error(self, exc):
            return RetryDecision(False, "invalid_vectors")

    result = await embed_cases(cases=[case("a", ["text"])], **options(Backend()))
    assert "usage" not in result["failed_cases"][0]
    assert result["embedding_usage"] == {"prompt_tokens": 9}


@pytest.mark.asyncio
async def test_unavailable_usage_is_not_invented():
    class Backend:
        name = model = "fake"

        async def embed(self, texts, *, input_type="document"):
            return [[1.0] for _ in texts]

        def classify_error(self, exc):
            return RetryDecision(False, "error")

    result = await embed_cases(cases=[case("a", ["text"]), case("empty", [])], **options(Backend()))
    assert all("usage" not in item for item in result["embedded_cases"])
    assert result["embedding_usage"] == {}
    assert result["embedding_requests"] == 1

@pytest.mark.asyncio
async def test_native_openai_usage_sums_all_cases_and_batches_concurrently():
    async def create(**kwargs):
        text = kwargs["input"][0]
        await asyncio.sleep(0.001 if text == "a" else 0)
        tokens = {"a": 3, "b": 5, "c": 11}[text]
        return CreateEmbeddingResponse(
            data=[{"index": 0, "embedding": [1.0], "object": "embedding"}],
            model="fake",
            object="list",
            usage={"prompt_tokens": tokens, "total_tokens": tokens},
        )

    backend = OpenAIEmbeddings(
        client=SimpleNamespace(embeddings=SimpleNamespace(create=create)), model="fake"
    )
    first, second = case("first", ["a", "b"]), case("second", ["c"])
    result = await embed_cases(cases=[first, second], **options(backend))
    one, two = result["embedded_cases"]
    assert "usage" not in one and "usage" not in two
    assert result["embedding_usage"] == {"prompt_tokens": 19, "total_tokens": 19}
    assert result["embedding_requests"] == 3
    assert one["requests"] == 2 and two["requests"] == 1
    assert first.execution["usage"] == {}  # Embedding never overwrites extraction usage.
