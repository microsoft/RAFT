# ruff: noqa: E402
# Optional SDK must be checked before importing its adapter.
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

voyageai = pytest.importorskip("voyageai")
from voyageai import error

from raft import ExtractedCase, LocalRetriever, embed_cases
from raft.embedding.voyage import VoyageEmbeddings


@pytest.mark.asyncio
async def test_real_sdk_request_and_response_without_network(monkeypatch):
    request = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, 0.0])],
            usage=SimpleNamespace(total_tokens=3),
        )
    )
    monkeypatch.setattr(voyageai.Embedding, "acreate", request)
    backend = VoyageEmbeddings(
        voyageai.AsyncClient(api_key="offline-test", max_retries=0),
        model="voyage-4",
        dimensions=256,
    )
    result = await backend.embed(["text"], input_type="query")
    args = request.call_args.kwargs
    assert args["input"] == ["text"]
    assert args["input_type"] == "query"
    assert args["model"] == "voyage-4"
    assert args["output_dimension"] == 256
    assert args["truncation"] is False
    assert args["output_dtype"] == "float"
    assert result.vectors == [[1.0, 0.0]]
    assert result.usage == {"total_tokens": 3}


@pytest.mark.asyncio
async def test_index_and_retrieve_select_distinct_input_types():
    class State(BaseModel):
        text: str

    calls = []

    async def embed(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(embeddings=[[1.0, 0.0] for _ in kwargs["texts"]], total_tokens=2)

    backend = VoyageEmbeddings(SimpleNamespace(embed=embed), model="voyage-4")
    result = await embed_cases(
        cases=[ExtractedCase(id="a", metadata={}, output=State(text="document"))],
        backend=backend,
        state_to_text=lambda state: [state.text],
        retries=0,
    )
    assert not result["failed_cases"]
    retriever = LocalRetriever.from_embeddings(result["embedded_cases"])
    hits = await retriever.retrieve(["question"], backend=backend, retries=0)
    assert hits["results"][0]["candidates"][0]["id"] == "a"
    assert [c["input_type"] for c in calls] == ["document", "query"]
    assert all("output_dimension" not in c for c in calls)
    assert hits["embedding_usage"] == {"total_tokens": 2}
    assert hits["embedding_requests"] == 1


@pytest.mark.parametrize(
    "exc,retryable,category",
    [
        (
            error.RateLimitError("slow", http_status=429, headers={"retry-after": "2"}),
            True,
            "rate_limit",
        ),
        (error.APIError("server", http_status=503), True, "http_503"),
        (error.AuthenticationError("invalid", http_status=401), False, "http_401"),
        (error.InvalidRequestError("too large", http_status=400), False, "http_400"),
        (error.APIConnectionError("offline"), True, "connection"),
        (error.Timeout("timeout"), True, "timeout"),
        (ValueError("invalid vectors"), False, "unexpected_error"),
    ],
)
def test_error_classification(exc, retryable, category):
    decision = VoyageEmbeddings(None, "voyage-4").classify_error(exc)
    assert decision.retryable is retryable
    assert decision.category == category
    if category == "rate_limit":
        assert decision.retry_after == 2


@pytest.mark.asyncio
async def test_malformed_response_rejected():
    client = SimpleNamespace(embed=AsyncMock(return_value=SimpleNamespace(embeddings=[])))
    with pytest.raises(ValueError, match="count"):
        await VoyageEmbeddings(client, "voyage-4").embed(["text"])
