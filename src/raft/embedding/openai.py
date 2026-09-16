from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from openai import AsyncOpenAI

from raft._openai_errors import classify_error
from raft.runtime import RetryDecision

from .backend import EmbeddingBatch


@dataclass
class OpenAIEmbeddings:
    """Use the caller's async client; the caller also owns its lifetime.

    Set max_retries=0 on the client to let embed_cases own the retry budget.
    Explicit float encoding also works with OpenAI-compatible embedding endpoints.
    """

    client: AsyncOpenAI
    model: str
    dimensions: int | None = None
    name: str = "openai"

    async def embed(
        self, texts: list[str], *, input_type: Literal["document", "query"] = "document"
    ) -> EmbeddingBatch:
        kwargs = {"dimensions": self.dimensions} if self.dimensions is not None else {}
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
            **kwargs,
        )
        data = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in data] != list(range(len(texts))):
            raise ValueError("Embedding response indices do not match the input batch")
        usage = getattr(response, "usage", None)
        return EmbeddingBatch(
            vectors=[item.embedding for item in data],
            usage=usage.model_dump(exclude_none=True) if usage is not None else {},
        )

    def classify_error(self, exc: Exception) -> RetryDecision:
        return classify_error(exc)
