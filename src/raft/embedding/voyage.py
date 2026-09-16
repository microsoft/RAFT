from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from voyageai import AsyncClient, error

from raft.runtime import RetryDecision, _retry_after

from .backend import EmbeddingBatch


@dataclass
class VoyageEmbeddings:
    """Reuse a caller-owned AsyncClient; set max_retries=0 for runner-owned retries.

    Inputs are never silently truncated. Returns dense float vectors only.
    """

    client: AsyncClient
    model: str
    dimensions: int | None = None
    name: str = "voyage"

    async def embed(
        self, texts: list[str], *, input_type: Literal["document", "query"] = "document"
    ) -> EmbeddingBatch:
        if input_type not in {"document", "query"}:
            raise ValueError("input_type must be document or query")
        kwargs = {"output_dimension": self.dimensions} if self.dimensions is not None else {}
        response = await self.client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type,
            truncation=False,
            output_dtype="float",
            **kwargs,
        )
        if len(response.embeddings) != len(texts):
            raise ValueError("Embedding response count does not match the input batch")
        tokens = getattr(response, "total_tokens", None)
        return EmbeddingBatch(
            vectors=response.embeddings,
            usage={"total_tokens": tokens} if tokens is not None else {},
        )

    def classify_error(self, exc: Exception) -> RetryDecision:
        if isinstance(exc, (TimeoutError, error.Timeout)):
            return RetryDecision(True, "timeout")
        if isinstance(exc, error.APIConnectionError):
            return RetryDecision(True, "connection")
        if isinstance(exc, error.VoyageError):
            status = exc.http_status
            retry_after = _retry_after(exc.headers)
            if isinstance(exc, error.RateLimitError):
                return RetryDecision(True, "rate_limit", retry_after)
            if isinstance(status, int):
                return RetryDecision(
                    status in {408, 409, 429} or status >= 500,
                    "rate_limit" if status == 429 else f"http_{status}",
                    retry_after,
                )
            if isinstance(exc, (error.ServerError, error.ServiceUnavailableError, error.TryAgain)):
                return RetryDecision(True, "server_error", retry_after)
            return RetryDecision(False, type(exc).__name__)
        return RetryDecision(False, "unexpected_error")
