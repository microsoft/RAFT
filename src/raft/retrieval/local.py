"""Reusable local exact-search snapshot. No storage writes or pipeline orchestration."""

from __future__ import annotations

import asyncio
import inspect
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import BaseModel

from raft._json import _id_key
from raft.cases import ExtractedCase, load_cases, restore_case
from raft.defaults import format_case as default_format_case
from raft.embedding import BM25Index, EmbeddingBackend
from raft.embedding._batching import embed_text_batches, validate_batch_size
from raft.progress import CaseProgress
from raft.runtime import _retry_delay, map_concurrent, validate_limits
from raft.storage import load_jsonl

from .ranking import cap_cases, normalize, rank_cases
from .types import CaseFormatter

CaseFilter = Callable[[str, ExtractedCase], bool]


class LocalRetriever:
    """Load once, then retrieve repeatedly. Treat source cases/indexes as read-only.

    Exact cosine search is in-memory, not an ANN/database service. BM25 is optional
    and must describe the same entry snapshot. Case payloads are reused, not copied.
    """

    def __init__(
        self,
        *,
        cases: Sequence[ExtractedCase],
        embeddings: Sequence[dict[str, Any]],
        bm25_index: BM25Index | None = None,
    ):
        self.cases = {}
        for case in cases:
            case = restore_case(case)
            key = _id_key(case.id)
            if key in self.cases:
                raise ValueError("Case IDs must be unique")
            self.cases[key] = case
        self.rows = [dict(row) for row in embeddings]
        self.case_rows: dict[str, list[int]] = {}
        seen, positions = set(), set()
        for row_number, row in enumerate(self.rows):
            key = _id_key(row["case_id"])
            if key not in self.cases:
                raise ValueError("Embedding case_id must reference a supplied case")
            if row["id"] is None or row["id"] == "" or _id_key(row["id"]) in seen:
                raise ValueError("Embedding IDs must be nonempty and unique")
            if type(row["item_index"]) is not int or row["item_index"] < 0:
                raise ValueError("item_index must be a nonnegative integer")
            position = (key, row["item_index"])
            if position in positions:
                raise ValueError("Each case/item_index must be unique")
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError("Embedding text must be nonempty")
            seen.add(_id_key(row["id"]))
            positions.add(position)
            self.case_rows.setdefault(key, []).append(row_number)
        self.vectors = normalize([row["embedding"] for row in self.rows]) if self.rows else None
        spaces = {(row["provider"], row["model"]) for row in self.rows}
        if len(spaces) > 1:
            raise ValueError("Vectors must use the same embedding provider and model")
        self.space = next(iter(spaces), None)
        if self.rows and any(row["dimensions"] != self.vectors.shape[1] for row in self.rows):
            raise ValueError("Stored dimensions must match vector dimensions")
        self.bm25 = bm25_index
        if self.bm25 is not None and [
            (_id_key(d["id"]), d["text"]) for d in self.bm25.documents
        ] != [(_id_key(r["id"]), r["text"]) for r in self.rows]:
            raise ValueError("BM25 IDs, texts and order must match the entry snapshot")

    @classmethod
    def from_embeddings(cls, embedded_cases, *, output_type=None, **kwargs):
        """Accept embed_cases()['embedded_cases']; optionally restore serialized outputs."""
        return cls(
            cases=[restore_case(item["case"], output_type) for item in embedded_cases],
            embeddings=[row for item in embedded_cases for row in item["embeddings"]],
            **kwargs,
        )

    @classmethod
    async def load(cls, output_dir: str | Path, *, output_type: type[BaseModel]):
        """Read existing extraction.json + embeddings.jsonl and optional bm25/.

        Uses the current snapshot helper layout. Does not create, modify, merge or
        repair files. Do not read a directory while another process is writing it.
        """

        def read():
            directory = Path(output_dir)
            return cls(
                cases=load_cases(directory / "extraction.json", output_type=output_type),
                embeddings=load_jsonl(directory / "embeddings.jsonl"),
                bm25_index=BM25Index.load(directory / "bm25")
                if (directory / "bm25").exists()
                else None,
            )

        return await asyncio.to_thread(read)

    def _eligible(self, query, case_filter):
        if case_filter is None:
            return list(range(len(self.rows)))
        eligible = []
        for key, positions in self.case_rows.items():
            allowed = case_filter(query, self.cases[key])
            if not isinstance(allowed, bool):
                raise ValueError("case_filter must return a bool")
            if allowed:
                eligible.extend(positions)
        return eligible

    def _rank(self, query, vector, eligible, top_k, rrf_constant, max_chars, format_case):
        query_vector = normalize([vector])[0]
        if query_vector.shape[0] != self.vectors.shape[1]:
            raise ValueError("Query dimensions must match stored vectors")
        lexical = self.bm25.scores(query) if self.bm25 is not None else None
        hits = rank_cases(
            self.rows, self.vectors, query_vector, eligible, self.cases, lexical, rrf_constant
        )
        return cap_cases(hits[:top_k], max_chars, format_case)

    async def retrieve(
        self,
        queries: list[str],
        *,
        backend: EmbeddingBackend,
        top_k: int = 5,
        case_filter: CaseFilter | None = None,
        max_chars: int | None = None,
        format_case: CaseFormatter = default_format_case,
        rrf_constant: int = 60,
        batch_size: int = 64,
        concurrency: int = 4,
        timeout: float = 120,
        retries: int = 1,
        rpm: int = 60,
        show_progress: bool = False,
    ) -> dict[str, Any]:
        """Return ordered query results and exact shared embedding usage.

        results contains one result per query; top_k counts distinct cases.
        batch_size caps query texts per request, including the final short batch.
        Only queries with eligible entries are embedded. embedding_usage and
        embedding_requests count responses/calls once across the whole operation;
        per-query requests/attempts count participation in potentially shared work.

        Filtering precedes ranking. With BM25, score is equal-weight RRF;
        otherwise cosine. A case's item_index anchors its best-ranked entry
        (the state_to_text list's zero-based index).

        format_case(hit) is a synchronous string formatter receiving the full hit:
        case, id, zero-based item_index, entry_id, scores, and source. The default
        serializes only id, metadata, output, and item_index, excluding execution
        history and review diagnostics. Structured candidates retain the full case.
        The formatter sees at most top_k distinct cases in ranking order.

        formatted_context joins selected case strings with two newlines.
        max_chars caps its exact character length, including separators; used_chars
        equals len(formatted_context). Stop before the first case that would exceed
        the budget, without truncating a case or trying smaller lower-ranked cases.
        truncated reports removal by the character budget, not the top_k limit.
        Empty results and failed queries have formatted_context="" and used_chars=0.

        CPU work runs in worker threads; callbacks must be pure/thread-safe.
        Per-query failures are returned in error; caller cancellation propagates.
        concurrency bounds batch workers and query filtering/ranking workers in
        their respective phases. timeout bounds each filtering, embedding, or
        ranking attempt; embedding attempts include their RPM wait. Ranking
        retries reuse vectors instead of requesting embeddings again.
        A cancelled await cannot forcibly stop already-running thread work.
        """
        validate_limits(concurrency, timeout, retries, rpm)
        validate_batch_size(batch_size)
        for name, value in (("top_k", top_k), ("rrf_constant", rrf_constant)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_chars is not None and (type(max_chars) is not int or max_chars < 0):
            raise ValueError("max_chars must be nonnegative or None")
        if case_filter is not None and not callable(case_filter):
            raise ValueError("case_filter must be callable or None")
        if (
            not callable(format_case)
            or inspect.iscoroutinefunction(format_case)
            or inspect.isasyncgenfunction(format_case)
            or inspect.iscoroutinefunction(getattr(format_case, "__call__", None))
            or inspect.isasyncgenfunction(getattr(format_case, "__call__", None))
        ):
            raise ValueError("format_case must be a synchronous callable")
        if not isinstance(queries, list) or any(
            not isinstance(q, str) or not q.strip() for q in queries
        ):
            raise ValueError("queries must be a list of nonempty strings")
        if self.space is not None and (backend.name, backend.model) != self.space:
            raise ValueError("Query backend must match the stored embedding provider and model")
        started: dict[int, float] = {}
        results = [
            {
                "query": query,
                "candidates": [],
                "formatted_context": "",
                "used_chars": 0,
                "truncated": False,
                "requests": 0,
                "attempts": 0,
                "error": None,
            }
            for query in queries
        ]

        def error_record(exc, decision):
            return {
                "type": type(exc).__name__,
                "message": str(exc),
                "category": decision.category,
                "retryable": decision.retryable,
            }

        def finish(index):
            result = results[index]
            result["elapsed_seconds"] = round(time.monotonic() - started[index], 3)
            progress.advance("failed" if result["error"] else "succeeded")

        async def prepare(index):
            started[index] = time.monotonic()
            result = results[index]
            for attempt in range(retries + 1):
                result["attempts"] = attempt + 1
                try:
                    async with asyncio.timeout(timeout):
                        eligible = await asyncio.to_thread(
                            self._eligible, queries[index], case_filter
                        )
                    if eligible:
                        return index, eligible
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    decision = backend.classify_error(exc)
                    if not decision.retryable or attempt == retries:
                        result["error"] = error_record(exc, decision)
                        break
                    await asyncio.sleep(_retry_delay(decision, attempt + 1))
            finish(index)
            return None

        async def rank(entry):
            (index, eligible), embedded = entry
            result = results[index]
            result["attempts"] += embedded["attempts"] - 1
            result["requests"] = embedded["requests"]
            if embedded["error"] is not None:
                result["error"] = error_record(embedded["error"], embedded["decision"])
                finish(index)
                return
            for attempt in range(retries + 1):
                if attempt:
                    result["attempts"] += 1
                try:
                    async with asyncio.timeout(timeout):
                        hits, context, truncated = await asyncio.to_thread(
                            self._rank,
                            queries[index],
                            embedded["embedding"],
                            eligible,
                            top_k,
                            rrf_constant,
                            max_chars,
                            format_case,
                        )
                        result.update(
                            candidates=hits, formatted_context=context,
                            used_chars=len(context), truncated=truncated,
                        )
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    decision = backend.classify_error(exc)
                    if not decision.retryable or attempt == retries:
                        result["error"] = error_record(exc, decision)
                        break
                    await asyncio.sleep(_retry_delay(decision, attempt + 1))
            finish(index)

        with CaseProgress(
            len(queries), enabled=show_progress, desc="Retrieval", unit="query"
        ) as progress:
            prepared = await map_concurrent(list(range(len(queries))), prepare, concurrency)
            ready = [entry for entry in prepared if entry is not None]
            embedded = await embed_text_batches(
                [queries[index] for index, _ in ready],
                backend=backend,
                input_type="query",
                batch_size=batch_size,
                concurrency=concurrency,
                timeout=timeout,
                retries=retries,
                rpm=rpm,
            )
            await map_concurrent(list(zip(ready, embedded["items"], strict=True)), rank, concurrency)
        return {
            "results": results,
            "embedding_usage": embedded["usage"],
            "embedding_requests": embedded["requests"],
        }
