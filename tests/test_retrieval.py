import asyncio
import json
import threading

import pytest
from pydantic import BaseModel, TypeAdapter

from raft import ExtractedCase, LocalRetriever
from raft.defaults import format_case
from raft.embedding import BM25Index, EmbeddingBatch
from raft.runtime import RetryDecision
from raft.storage import save_json, save_jsonl


class State(BaseModel):
    timeline: list[str]
    entities: list[str] = []


class Backend:
    name = "fake"
    model = "test"

    def __init__(self):
        self.calls = []

    async def embed(self, texts, *, input_type="document"):
        self.calls.append(list(texts))
        vectors = [[0.0, 1.0] if text == "vertical" else [1.0, 0.0] for text in texts]
        return EmbeddingBatch(
            vectors, {"prompt_tokens": len(texts) * 3, "total_tokens": len(texts) * 3}
        )

    def classify_error(self, exc):
        return RetryDecision(isinstance(exc, TimeoutError), "test_error")


def fixture():
    cases = [
        ExtractedCase(
            id=id, metadata={"product": product}, output=State(timeline=texts, entities=["sso"])
        )
        for id, product, texts in [
            ("a", "portal", ["opening", "certificate repaired"]),
            ("b", "upload", ["upload stalls"]),
            ("c", "portal", ["certificate expired"]),
        ]
    ]
    vectors = [[0.9, 0.1], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    rows = []
    for case in cases:
        for index, text in enumerate(case.output.timeline):
            rows.append(
                {
                    "id": f"{case.id}-{index}",
                    "case_id": case.id,
                    "item_index": index,
                    "text": text,
                    "embedding": vectors[len(rows)],
                    "provider": "fake",
                    "model": "test",
                    "dimensions": 2,
                }
            )
    return cases, rows


async def search(retriever, queries=None, **kwargs):
    response = await retriever.retrieve(
        ["query"] if queries is None else queries,
        backend=kwargs.pop("backend", Backend()),
        batch_size=kwargs.pop("batch_size", 1),
        rpm=1000,
        **kwargs,
    )
    return response["results"]


@pytest.mark.asyncio
async def test_entry_ranking_promotes_unique_cases_and_keeps_best_anchor_and_metadata():
    cases, rows = fixture()
    result = (await search(LocalRetriever(cases=cases, embeddings=rows), top_k=3))[0]
    hits = result["candidates"]
    assert [h["id"] for h in hits] == ["a", "b", "c"]
    assert hits[0]["item_index"] == 1 and hits[0]["entry_id"] == "a-1"
    assert hits[0]["case"] is cases[0]
    assert hits[0]["case"].metadata == {"product": "portal"}
    assert hits[0]["case"].output.timeline == ["opening", "certificate repaired"]
    assert hits[0]["score"] == pytest.approx(1.0)
    assert hits[-1]["score"] == pytest.approx(-1.0)
    assert all(h["bm25_score"] is None for h in hits)
    assert "usage" not in result
    assert result["error"] is None


@pytest.mark.asyncio
async def test_bm25_rrf_matches_full_entry_reference_then_deduplicates():
    cases, rows = fixture()
    bm25 = BM25Index.from_records(rows)
    retriever = LocalRetriever(cases=cases, embeddings=rows, bm25_index=bm25)
    result = (await search(retriever, ["certificate expired"], top_k=3))[0]
    lexical = bm25.scores("certificate expired")
    dense_order = [1, 0, 2, 3]
    lexical_order = sorted(
        (i for i in range(4) if lexical[i] > 0), key=lambda i: (-lexical[i], rows[i]["id"])
    )
    scores = {i: 1 / (60 + rank) for rank, i in enumerate(dense_order, 1)}
    for rank, i in enumerate(lexical_order, 1):
        scores[i] += 1 / (60 + rank)
    expected = []
    for i in sorted(scores, key=lambda i: (-scores[i], rows[i]["id"])):
        if rows[i]["case_id"] not in [rows[j]["case_id"] for j in expected]:
            expected.append(i)
    assert [hit["entry_id"] for hit in result["candidates"]] == [rows[i]["id"] for i in expected]
    assert [hit["score"] for hit in result["candidates"]] == pytest.approx(
        [scores[i] for i in expected]
    )


@pytest.mark.asyncio
async def test_prefilter_fills_topk_and_filters_once_per_case():
    cases, rows = fixture()
    seen = []

    def allow(query, case):
        seen.append((query, case))
        return case.metadata["product"] == "portal" and "sso" in case.output.entities

    retriever = LocalRetriever(
        cases=cases, embeddings=rows, bm25_index=BM25Index.from_records(rows)
    )
    result = (await search(retriever, top_k=2, case_filter=allow))[0]
    assert [hit["id"] for hit in result["candidates"]] == ["a", "c"]
    assert len(seen) == len(cases) and seen[0][1] is cases[0]
    assert result["candidates"][1]["score"] == pytest.approx(1 / 63)
    backend = Backend()
    result = (await search(retriever, backend=backend, case_filter=lambda q, c: False))[0]
    assert result["candidates"] == [] and not backend.calls


@pytest.mark.asyncio
async def test_cases_without_entries_are_not_filter_candidates():
    cases, rows = fixture()
    cases.append(ExtractedCase(id="no-vector", metadata={}, output=State(timeline=[])))

    def allow(query, case):
        assert case.id != "no-vector"
        return True

    retriever = LocalRetriever(cases=cases, embeddings=rows)
    result = (await search(retriever, case_filter=allow))[0]
    assert result["error"] is None and len(result["candidates"]) == 3


@pytest.mark.asyncio
async def test_ties_and_case_id_types_are_stable_independent_of_row_order():
    cases = [ExtractedCase(id=id, metadata={}, output=State(timeline=["same"])) for id in (1, "1")]
    rows = [
        {
            "id": f"row-{i}",
            "case_id": c.id,
            "item_index": 0,
            "text": "same",
            "embedding": [1.0, 0.0],
            "provider": "fake",
            "model": "test",
            "dimensions": 2,
        }
        for i, c in enumerate(cases)
    ]
    first = (await search(LocalRetriever(cases=cases, embeddings=rows)))[0]
    again = (await search(LocalRetriever(cases=cases[::-1], embeddings=rows[::-1])))[0]
    assert [h["id"] for h in first["candidates"]] == [1, "1"]
    assert [h["id"] for h in again["candidates"]] == [1, "1"]


@pytest.mark.asyncio
async def test_query_batch_order_duplicates_empty_batch_and_empty_index():
    cases, rows = fixture()
    retriever = LocalRetriever(cases=cases, embeddings=rows)
    results = await search(retriever, ["vertical", "query", "vertical"], top_k=1)
    assert [r["query"] for r in results] == ["vertical", "query", "vertical"]
    assert [r["candidates"][0]["id"] for r in results] == ["b", "a", "b"]
    assert await search(retriever, []) == []
    backend = Backend()
    empty = (await search(LocalRetriever(cases=[], embeddings=[]), backend=backend))[0]
    assert empty["candidates"] == [] and not backend.calls


@pytest.mark.asyncio
async def test_char_budget_is_strict_whole_case_ranked_prefix():
    cases, rows = fixture()
    retriever = LocalRetriever(cases=cases, embeddings=rows)
    unlimited = (await search(retriever))[0]
    size = len(format_case(unlimited["candidates"][0]))
    result = (await search(retriever, context_budget={"unit": "chars", "limit": size}, top_k=3))[0]
    assert [hit["id"] for hit in result["candidates"]] == ["a"]
    assert result["used_chars"] == size and result["truncated"]
    assert result["formatted_context"] == format_case(result["candidates"][0])
    result = (await search(retriever, context_budget={"unit": "chars", "limit": size - 1}))[0]
    assert result["candidates"] == [] and result["truncated"]
    assert result["used_chars"] == 0
    assert result["formatted_context"] == ""
    assert not unlimited["truncated"]
    assert unlimited["formatted_context"] == "\n\n".join(
        format_case(hit) for hit in unlimited["candidates"]
    )
    assert unlimited["used_chars"] == len(unlimited["formatted_context"])


@pytest.mark.asyncio
async def test_default_context_contains_state_and_anchor_but_no_diagnostics():
    cases, rows = fixture()
    cases[0].execution = {"private_debug": "history" * 10_000}
    cases[0].review = {"private_review": "not retrieval evidence"}
    original = cases[0].model_dump()
    result = (await search(LocalRetriever(cases=cases, embeddings=rows), top_k=1))[0]
    expected = {
        "id": "a", "metadata": cases[0].metadata, "output": cases[0].output.model_dump(),
        "item_index": 1,
    }
    assert json.loads(result["formatted_context"]) == expected
    assert result["used_chars"] < len(cases[0].model_dump_json())
    assert "private_debug" not in result["formatted_context"]
    assert "private_review" not in result["formatted_context"]
    assert result["candidates"][0]["case"] is cases[0]
    assert cases[0].model_dump() == original
    budgeted = (await search(
        LocalRetriever(cases=cases, embeddings=rows),
        context_budget={"unit": "chars", "limit": result["used_chars"]}, top_k=1,
    ))[0]
    assert budgeted["formatted_context"] == result["formatted_context"]
    assert len(budgeted["candidates"]) == 1
    assert not budgeted["truncated"]


@pytest.mark.asyncio
async def test_default_matched_index_is_query_specific_for_the_same_parent_case():
    cases, rows = fixture()
    results = await search(
        LocalRetriever(cases=cases, embeddings=rows), ["query", "vertical"], top_k=1,
        case_filter=lambda query, case: case.id == "a",
    )
    assert [r["candidates"][0]["id"] for r in results] == ["a", "a"]
    assert [json.loads(r["formatted_context"])["item_index"] for r in results] == [1, 0]
    assert [r["candidates"][0]["entry_id"] for r in results] == ["a-1", "a-0"]


@pytest.mark.asyncio
async def test_custom_formatter_receives_full_hit_and_runs_in_worker_thread():
    cases, rows = fixture()
    cases[0].review = {"keep": True}
    seen = []
    main_thread = threading.get_ident()

    def formatter(hit):
        seen.append((hit, threading.get_ident()))
        assert hit["case"].execution is cases[0].execution
        assert hit["case"].review == {"keep": True}
        return f"{hit['id']}:{hit['item_index']}:{hit['entry_id']}"

    result = (await search(
        LocalRetriever(cases=cases, embeddings=rows), top_k=1, format_case=formatter,
    ))[0]
    assert result["formatted_context"] == "a:1:a-1"
    assert result["used_chars"] == 7
    assert len(seen) == 1
    hit, thread = seen[0]
    assert thread != main_thread
    assert hit is result["candidates"][0]
    assert set(hit) == {
        "id", "case", "item_index", "entry_id", "score",
        "cosine_similarity", "bm25_score", "source",
    }
    assert hit["score"] == hit["cosine_similarity"] == 1.0
    assert hit["bm25_score"] is None and hit["source"] == "direct"


@pytest.mark.asyncio
@pytest.mark.parametrize("limit,ids,text,truncated", [
    (None, ["a", "b", "c"], "aaa\n\nbb\n\nc", False),
    (0, [], "", True),
    (2, [], "", True),
    (3, ["a"], "aaa", True),
    (6, ["a"], "aaa", True),
    (7, ["a", "b"], "aaa\n\nbb", True),
    (9, ["a", "b"], "aaa\n\nbb", True),
    (10, ["a", "b", "c"], "aaa\n\nbb\n\nc", False),
])
async def test_formatted_budget_counts_exact_text_and_separators(limit, ids, text, truncated):
    cases, rows = fixture()
    result = (await search(
        LocalRetriever(cases=cases, embeddings=rows), top_k=3,
        context_budget=None if limit is None else {"unit": "chars", "limit": limit},
        format_case=lambda hit: {"a": "aaa", "b": "bb", "c": "c"}[hit["id"]],
    ))[0]
    assert [hit["id"] for hit in result["candidates"]] == ids
    assert result["formatted_context"] == text
    assert result["used_chars"] == len(text)
    assert result["truncated"] is truncated
    assert result["error"] is None


@pytest.mark.asyncio
async def test_budget_stops_at_first_oversize_case_and_formats_no_lower_candidates():
    cases, rows = fixture()
    seen = []

    def formatter(hit):
        seen.append(hit["id"])
        return {"a": "aaa", "b": "too long", "c": "c"}[hit["id"]]

    result = (await search(
        LocalRetriever(cases=cases, embeddings=rows), context_budget={"unit": "chars", "limit": 6}, format_case=formatter,
    ))[0]
    assert seen == ["a", "b"]
    assert [hit["id"] for hit in result["candidates"]] == ["a"]
    assert result["formatted_context"] == "aaa"
    # A smaller third case would fit, but ranked-prefix semantics must not skip the second.
    assert len("aaa\n\nc") == 6


@pytest.mark.asyncio
async def test_unicode_budget_is_character_length_not_bytes_and_applies_per_query():
    cases, rows = fixture()
    text = "café \U0001f600"
    results = await search(
        LocalRetriever(cases=cases, embeddings=rows), ["query", "vertical"], top_k=1,
        context_budget={"unit": "chars", "limit": len(text)}, format_case=lambda hit: text,
    )
    assert len(text.encode("utf-8")) > len(text)
    assert [r["candidates"][0]["id"] for r in results] == ["a", "b"]
    assert all(r["used_chars"] == len(text) and r["formatted_context"] == text for r in results)
    assert all(not r["truncated"] for r in results)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, 3, ["text"], {"text": "bad"}, b"bytes"])
async def test_invalid_formatter_output_is_a_query_error_without_partial_context(value):
    cases, rows = fixture()

    def formatter(hit):
        return "valid first case" if hit["id"] == "a" else value

    result = (await search(
        LocalRetriever(cases=cases, embeddings=rows), format_case=formatter,
    ))[0]
    assert result["error"]["type"] == "TypeError"
    assert "format_case" in result["error"]["message"]
    assert result["candidates"] == []
    assert result["formatted_context"] == "" and result["used_chars"] == 0


@pytest.mark.asyncio
async def test_formatter_failures_are_isolated_and_do_not_repeat_embeddings(monkeypatch):
    cases, rows = fixture()
    seen = []

    def formatter(hit):
        seen.append(hit["id"])
        if hit["id"] == "b":
            raise ValueError("formatter rejected case b")
        if len(seen) == 1:
            raise TimeoutError("temporary formatter failure")
        return hit["id"]

    monkeypatch.setattr("raft.retrieval.local._retry_delay", lambda *args: 0)
    backend = Backend()
    results = await search(
        LocalRetriever(cases=cases, embeddings=rows), ["query", "vertical"],
        backend=backend, top_k=1, format_case=formatter, concurrency=1,
    )
    assert backend.calls == [["query"], ["vertical"]]
    assert results[0]["formatted_context"] == "a" and results[0]["attempts"] == 2
    assert results[1]["error"]["message"] == "formatter rejected case b"
    assert results[1]["formatted_context"] == "" and results[1]["candidates"] == []


@pytest.mark.asyncio
async def test_async_and_noncallable_formatters_fail_before_api_calls():
    async def async_formatter(hit):
        return "text"

    async def async_generator(hit):
        yield "text"

    class AsyncFormatter:
        async def __call__(self, hit):
            return "text"

    cases, rows = fixture()
    backend = Backend()
    for formatter in (None, "not callable", 1, async_formatter, async_generator, AsyncFormatter()):
        with pytest.raises(ValueError, match="synchronous callable"):
            await search(
                LocalRetriever(cases=cases, embeddings=rows),
                backend=backend, format_case=formatter,
            )
    assert not backend.calls


@pytest.mark.asyncio
async def test_hidden_coroutine_return_is_rejected_without_unawaited_coroutine():
    async def make_text():
        return "text"

    cases, rows = fixture()
    result = (await search(
        LocalRetriever(cases=cases, embeddings=rows), format_case=lambda hit: make_text(),
    ))[0]
    assert result["error"]["type"] == "TypeError"
    assert result["formatted_context"] == ""


@pytest.mark.asyncio
async def test_empty_index_and_filtered_results_have_empty_context_without_formatting():
    def formatter(hit):
        pytest.fail("There are no hits to format")

    cases, rows = fixture()
    for retriever in (
        LocalRetriever(cases=[], embeddings=[]),
        LocalRetriever(cases=cases, embeddings=rows),
    ):
        result = (await search(
            retriever, case_filter=lambda query, case: False, format_case=formatter,
        ))[0]
        assert result["formatted_context"] == ""
        assert result["used_chars"] == 0 and not result["truncated"]
        assert result["error"] is None


@pytest.mark.asyncio
async def test_output_only_custom_formatter_and_empty_string_are_preserved():
    cases, rows = fixture()
    retriever = LocalRetriever(cases=cases, embeddings=rows)
    result = (await search(
        retriever, top_k=1, format_case=lambda hit: hit["case"].output.model_dump_json(),
    ))[0]
    assert json.loads(result["formatted_context"]) == cases[0].output.model_dump()
    result = (await search(retriever, top_k=1, context_budget={"unit": "chars", "limit": 0}, format_case=lambda hit: ""))[0]
    assert len(result["candidates"]) == 1 and not result["truncated"]
    assert result["formatted_context"] == "" and result["used_chars"] == 0


@pytest.mark.asyncio
async def test_query_retrieval_rejects_graph_configuration_without_model_calls():
    cases, rows = fixture()
    with pytest.raises(TypeError, match="graph_edges"):
        LocalRetriever(cases=cases, embeddings=rows, graph_edges=[])
    retriever = LocalRetriever(cases=cases, embeddings=rows)
    backend = Backend()
    with pytest.raises(TypeError, match="graph_slots"):
        await search(retriever, backend=backend, graph_slots=1)
    assert not backend.calls


@pytest.mark.asyncio
async def test_load_local_snapshot_and_serialized_embedding_results(tmp_path):
    cases, rows = fixture()
    save_json(tmp_path / "extraction.json", {"extracted_cases": cases})
    save_jsonl(tmp_path / "embeddings.jsonl", rows)
    BM25Index.from_records(rows).save(tmp_path / "bm25")
    # Query-only loading must not parse or depend on a separately saved graph.
    save_jsonl(tmp_path / "case_graph/edges.jsonl", [{"invalid_graph": "ignored"}])
    retriever = await LocalRetriever.load(tmp_path, output_type=State)
    result = (await search(retriever))[0]
    assert isinstance(result["candidates"][0]["case"].output, State)
    assert retriever.bm25 is not None
    assert json.loads(TypeAdapter(dict).dump_json(result))["candidates"][0]["case"]["metadata"]
    items = [{"case": c, "embeddings": [r for r in rows if r["case_id"] == c.id]} for c in cases]
    restored = LocalRetriever.from_embeddings(
        TypeAdapter(list).dump_python(items, mode="json"), output_type=State
    )
    assert (await search(restored))[0]["candidates"][0]["id"] == "a"


@pytest.mark.asyncio
async def test_retry_usage_failure_isolation_and_cancellation(monkeypatch):
    cases, rows = fixture()
    retriever = LocalRetriever(cases=cases, embeddings=rows)

    class Flaky(Backend):
        async def embed(self, texts, *, input_type="document"):
            if texts == ["cancel"]:
                raise asyncio.CancelledError
            if texts == ["bad"]:
                raise ValueError("terminal")
            if not self.calls:
                self.calls.append(texts)
                raise TimeoutError("temporary")
            return await super().embed(texts, input_type=input_type)

    monkeypatch.setattr("raft.embedding._batching._retry_delay", lambda *args: 0)
    backend = Flaky()
    result = await search(retriever, ["query", "bad"], backend=backend)
    assert result[0]["attempts"] == 2 and result[0]["requests"] == 2
    assert "usage" not in result[0]
    assert result[1]["error"]["type"] == "ValueError" and result[1]["candidates"] == []
    with pytest.raises(asyncio.CancelledError):
        await search(retriever, ["cancel"], backend=backend)


@pytest.mark.asyncio
async def test_query_error_retains_provider_diagnostics():
    import httpx2
    import openai

    from raft._openai_errors import classify_error

    class Unavailable(Backend):
        async def embed(self, texts, *, input_type="document"):
            raise openai.NotFoundError(
                "No such embedding deployment",
                response=httpx2.Response(
                    404, headers={"x-request-id": "embed-test"},
                    request=httpx2.Request("POST", "https://provider.example"),
                ),
                body={"code": "DeploymentNotFound"},
            )

        def classify_error(self, exc):
            return classify_error(exc)

    cases, rows = fixture()
    result = (await search(
        LocalRetriever(cases=cases, embeddings=rows), backend=Unavailable(), retries=3,
    ))[0]
    assert result["attempts"] == result["requests"] == 1
    assert result["error"]["details"] == {
        "status_code": 404, "code": "DeploymentNotFound", "request_id": "embed-test",
    }
    assert result["error"]["retryable"] is False
    assert result["candidates"] == [] and result["formatted_context"] == ""


@pytest.mark.asyncio
async def test_concurrency_and_timeout():
    class Slow(Backend):
        active = maximum = 0

        async def embed(self, texts, *, input_type="document"):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            try:
                await asyncio.sleep(0.01)
                return await super().embed(texts, input_type=input_type)
            finally:
                self.active -= 1

    cases, rows = fixture()
    backend = Slow()
    retriever = LocalRetriever(cases=cases, embeddings=rows)
    results = await search(retriever, ["query"] * 5, backend=backend, concurrency=2)
    assert backend.maximum == 2 and all(r["error"] is None for r in results)
    failed = (await search(retriever, backend=backend, timeout=0.001, retries=0))[0]
    assert failed["error"]["type"] == "TimeoutError" and backend.active == 0


def test_rejects_invalid_or_misaligned_snapshots():
    cases, rows = fixture()
    with pytest.raises(ValueError, match="BM25"):
        LocalRetriever(cases=cases, embeddings=rows, bm25_index=BM25Index.from_records(rows[::-1]))
    for bad in (
        {"embedding": [0, 0]},
        {"embedding": [float("nan"), 1]},
        {"model": "other"},
        {"dimensions": 3},
        {"case_id": "missing"},
    ):
        with pytest.raises(ValueError):
            LocalRetriever(cases=cases, embeddings=[{**rows[0], **bad}, *rows[1:]])


@pytest.mark.asyncio
async def test_rejects_wrong_query_model_dimensions_and_filter_errors():
    cases, rows = fixture()
    retriever = LocalRetriever(cases=cases, embeddings=rows)
    backend = Backend()
    backend.model = "wrong"
    with pytest.raises(ValueError, match="backend"):
        await search(retriever, backend=backend)
    assert not backend.calls
    result = (await search(retriever, case_filter=lambda q, c: 1))[0]
    assert result["error"]["type"] == "ValueError"
    class Wrong(Backend):
        async def embed(self, texts, *, input_type="document"):
            return [[1.0]]

    result = (await search(retriever, backend=Wrong()))[0]
    assert result["error"]["message"] == "Query dimensions must match stored vectors"


@pytest.mark.asyncio
async def test_default_query_batching_keeps_duplicates_and_exact_operation_usage():
    cases, rows = fixture()
    retriever = LocalRetriever(cases=cases, embeddings=rows)
    backend = Backend()
    queries = ["vertical", "query", "vertical"] * 22 + ["query"]
    response = await retriever.retrieve(queries, backend=backend, top_k=1, rpm=1000)
    assert backend.calls == [queries[:64], queries[64:]]
    assert response["embedding_requests"] == 2
    assert response["embedding_usage"] == {"prompt_tokens": 201, "total_tokens": 201}
    assert [result["query"] for result in response["results"]] == queries
    assert [result["candidates"][0]["id"] for result in response["results"]] == [
        "b" if query == "vertical" else "a" for query in queries
    ]
    assert all("usage" not in result for result in response["results"])
    assert all(result["requests"] == result["attempts"] == 1 for result in response["results"])
    assert await retriever.retrieve([], backend=backend) == {
        "results": [], "embedding_usage": {}, "embedding_requests": 0,
    }
    assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_query_filters_and_filter_failures_are_resolved_before_embedding():
    cases, rows = fixture()
    retriever = LocalRetriever(cases=cases, embeddings=rows)
    backend = Backend()

    def allow(query, case):
        if query == "bad filter":
            raise ValueError("filter failed")
        return query != "none"

    queries = ["none", "vertical", "bad filter", "query", "vertical"]
    response = await retriever.retrieve(
        queries, backend=backend, case_filter=allow, top_k=1, rpm=1000,
    )
    assert backend.calls == [["vertical", "query", "vertical"]]
    assert response["embedding_requests"] == 1
    assert response["embedding_usage"] == {"prompt_tokens": 9, "total_tokens": 9}
    results = response["results"]
    assert [result["query"] for result in results] == queries
    assert results[0]["candidates"] == [] and results[0]["error"] is None
    assert results[2]["error"]["message"] == "filter failed"
    assert results[0]["requests"] == results[2]["requests"] == 0
    assert [results[index]["candidates"][0]["id"] for index in (1, 3, 4)] == ["b", "a", "b"]
    empty = await retriever.retrieve(
        ["query", "vertical"], backend=backend, case_filter=lambda query, case: False,
    )
    assert empty["embedding_usage"] == {} and empty["embedding_requests"] == 0
    assert all(result["candidates"] == [] for result in empty["results"])
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_ranking_retries_reuse_embeddings_and_isolate_ranking_failures(monkeypatch):
    cases, rows = fixture()
    retriever = LocalRetriever(cases=cases, embeddings=rows)
    backend = Backend()
    counts = {}
    original = retriever._rank

    def rank(query, *args):
        counts[query] = counts.get(query, 0) + 1
        if query == "retry" and counts[query] == 1:
            raise TimeoutError("temporary ranking failure")
        if query == "invalid":
            raise ValueError("ranking failed")
        return original(query, *args)

    monkeypatch.setattr(retriever, "_rank", rank)
    monkeypatch.setattr("raft.retrieval.local._retry_delay", lambda *args: 0)
    response = await retriever.retrieve(
        ["retry", "invalid", "vertical"], backend=backend, top_k=1, retries=1, rpm=1000,
    )
    assert counts == {"retry": 2, "invalid": 1, "vertical": 1}
    assert backend.calls == [["retry", "invalid", "vertical"]]
    assert response["embedding_requests"] == 1
    assert response["embedding_usage"] == {"prompt_tokens": 9, "total_tokens": 9}
    first, failed, last = response["results"]
    assert first["candidates"][0]["id"] == "a" and first["attempts"] == 2
    assert failed["error"]["message"] == "ranking failed"
    assert failed["candidates"] == []
    assert last["candidates"][0]["id"] == "b"
    assert all(result["requests"] == 1 for result in response["results"])


@pytest.mark.asyncio
async def test_concurrency_bounds_each_filter_embedding_and_rank_phase(monkeypatch):
    cases, rows = fixture()
    retriever = LocalRetriever(cases=cases, embeddings=rows)
    lock = threading.Lock()
    active = {"filter": 0, "rank": 0, "embed": 0}
    peak = dict(active)
    finished = dict(active)
    gates = {phase: threading.Event() for phase in ("filter", "rank")}
    embedding_gate = asyncio.Event()

    def bounded(phase, operation):
        def call(*args):
            with lock:
                if phase == "rank":
                    assert finished["embed"] == 3
                active[phase] += 1
                peak[phase] = max(peak[phase], active[phase])
                if active[phase] == 2:
                    gates[phase].set()
            try:
                assert gates[phase].wait(1), f"{phase} did not run concurrently"
                return operation(*args)
            finally:
                with lock:
                    active[phase] -= 1
                    finished[phase] += 1
        return call

    class BoundedBackend(Backend):
        async def embed(self, texts, *, input_type="document"):
            assert finished["filter"] == 6
            active["embed"] += 1
            peak["embed"] = max(peak["embed"], active["embed"])
            if active["embed"] == 2:
                embedding_gate.set()
            try:
                await asyncio.wait_for(embedding_gate.wait(), 1)
                return await super().embed(texts, input_type=input_type)
            finally:
                active["embed"] -= 1
                finished["embed"] += 1

    monkeypatch.setattr(retriever, "_eligible", bounded("filter", retriever._eligible))
    monkeypatch.setattr(retriever, "_rank", bounded("rank", retriever._rank))
    response = await retriever.retrieve(
        ["query"] * 6, backend=BoundedBackend(), batch_size=2, concurrency=2,
        timeout=2, retries=0, rpm=1000,
    )
    assert peak == {"filter": 2, "rank": 2, "embed": 2}
    assert active == {"filter": 0, "rank": 0, "embed": 0}
    assert finished == {"filter": 6, "rank": 6, "embed": 3}
    assert all(result["error"] is None for result in response["results"])


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["filter", "rank"])
async def test_cpu_phase_timeout_does_not_repeat_or_invent_embedding_requests(monkeypatch, phase):
    cases, rows = fixture()
    retriever = LocalRetriever(cases=cases, embeddings=rows)
    backend = Backend()
    released, exited = threading.Event(), threading.Event()
    attribute = "_eligible" if phase == "filter" else "_rank"
    original = getattr(retriever, attribute)

    def slow(*args):
        try:
            assert released.wait(1)
            return original(*args)
        finally:
            exited.set()

    monkeypatch.setattr(retriever, attribute, slow)
    try:
        response = await retriever.retrieve(
            ["query"], backend=backend, timeout=0.02, retries=0, rpm=1000,
        )
    finally:
        released.set()
    assert await asyncio.to_thread(exited.wait, 1)
    assert response["results"][0]["error"]["type"] == "TimeoutError"
    expected = int(phase == "rank")
    assert response["embedding_requests"] == len(backend.calls) == expected
    assert response["embedding_usage"] == (
        {"prompt_tokens": 3, "total_tokens": 3} if expected else {}
    )
