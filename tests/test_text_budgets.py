"""Character and real-token budgets through extraction and retrieval APIs."""

import json
import threading

import pytest
import tiktoken
from test_batching import Backend as ExtractionBackend
from test_batching import data, extract, finish
from test_retrieval import Backend, fixture, search

from raft import LocalRetriever
from raft._json import _to_json


@pytest.fixture(scope="module")
def encoding():
    return tiktoken.get_encoding("o200k_base")


async def test_token_retrieval_budget_counts_joined_context_not_separate_cases(encoding):
    cases, rows = fixture()
    texts = {"a": "hello\n", "b": "\nworld", "c": "last " * 100}
    expected = texts["a"] + "\n\n" + texts["b"]
    def count_tokens(text):
        return len(encoding.encode_ordinary(text))
    limit = count_tokens(expected)
    assert limit != count_tokens(texts["a"]) + count_tokens("\n\n") + count_tokens(texts["b"])
    budget = {"unit": "tokens", "limit": limit, "count_tokens": count_tokens}

    result = (await search(
        LocalRetriever(cases=cases, embeddings=rows),
        top_k=3, context_budget=budget, format_case=lambda hit: texts[hit["id"]],
    ))[0]

    assert result["error"] is None
    assert [hit["id"] for hit in result["candidates"]] == ["a", "b"]
    assert result["formatted_context"] == expected
    assert count_tokens(result["formatted_context"]) == limit
    assert result["used_chars"] == len(expected)
    assert result["truncated"]
    assert budget == {"unit": "tokens", "limit": limit, "count_tokens": count_tokens}


async def test_token_extraction_delivers_whole_artifacts_with_serializable_budget(encoding):
    def count_tokens(text):
        return len(encoding.encode_ordinary(text))
    source = [_to_json(item) for item in sorted(data()["items"], key=lambda item: item["seq"])]
    limit = max(map(count_tokens, source))
    budget = {"unit": "tokens", "limit": limit, "count_tokens": count_tokens}

    def handler(context, prompt, count):
        assert prompt["batch_budget"] == {"unit": "tokens", "limit": limit}
        items = prompt["batch"]["items"]
        assert count_tokens("".join(item["artifact_json"] for item in items)) <= limit
        assert all(item["end_char_exclusive"] == len(item["artifact_json"]) for item in items)
        fragments = prompt["current_state"].get("fragments", [])
        assert finish(context, {"fragments": fragments + [
            item["artifact_json"] for item in items
        ]})["ok"]

    backend = ExtractionBackend(handler)
    result = await extract(backend, batch_budget=budget)
    assert not result["failed_cases"], result
    assert result["extracted_cases"][0].output.fragments == source
    assert len(backend.prompts) > 1


@pytest.mark.parametrize("difference", [0, -1])
async def test_token_sql_response_budget_counts_complete_json_and_review(encoding, difference):
    expected = {"columns": ["text"], "rows": [{"text": "hello\n\nworld"}], "row_count": 1}
    def count_tokens(text):
        return len(encoding.encode_ordinary(text))
    limit = count_tokens(json.dumps(expected, ensure_ascii=False, default=str)) + difference
    budget = {"unit": "tokens", "limit": limit, "count_tokens": count_tokens}
    seen = []

    def check(context):
        result = context.query("SELECT 'hello\n\nworld' AS text")
        seen.append(context.stage)
        if difference == 0:
            assert result == expected
        else:
            assert result["error"] == "query_result_too_large"
            assert result["query_budget"] == {"unit": "tokens", "limit": limit}
            assert "rows" not in result

    class Backend(ExtractionBackend):
        async def review(self, agent, prompt, *, context, **kwargs):
            check(context)
            return {"keep": True}

    def handler(context, prompt, count):
        check(context)
        assert finish(context, {"fragments": []})["ok"]

    result = await extract(Backend(handler), query_budget=budget)
    assert not result["failed_cases"], result
    assert seen == ["worker", "reviewer"]


@pytest.mark.parametrize("text", ["お誕生日おめでとう", '中文🙂\n"\\', "user_id=ab12-cd34"])
@pytest.mark.parametrize("difference", [0, -1])
async def test_token_retrieval_exact_boundary_and_unicode(encoding, text, difference):
    cases, rows = fixture()
    def count_tokens(value):
        return len(encoding.encode_ordinary(value))
    budget = {"unit": "tokens", "limit": count_tokens(text) + difference,
              "count_tokens": count_tokens}
    results = await search(
        LocalRetriever(cases=cases, embeddings=rows), ["query", "vertical"],
        top_k=1, context_budget=budget, format_case=lambda hit: text,
    )
    for result in results:
        assert result["error"] is None
        assert result["formatted_context"] == (text if difference == 0 else "")
        assert len(result["candidates"]) == (1 if difference == 0 else 0)
        assert result["truncated"] is (difference == -1)


async def test_token_extraction_preflights_all_artifacts_before_running_agents(encoding):
    raw = data()
    def count_tokens(text):
        return len(encoding.encode_ordinary(text))
    limit = max(count_tokens(_to_json(item)) for item in raw["items"])
    raw["items"][-1]["text"] = "unusually long artifact " * 200
    backend = ExtractionBackend(lambda *args: pytest.fail("Oversized case must not run"))
    result = await extract(backend, batch_budget={
        "unit": "tokens", "limit": limit, "count_tokens": count_tokens,
    }, case=raw, retries=2)
    failure = result["failed_cases"][0]
    assert failure["error_category"] == "artifact_too_large"
    assert failure["execution"]["attempts"] == 0 and not backend.prompts
    assert failure["details"]["artifact_size"] == count_tokens(_to_json(raw["items"][-1]))
    assert failure["details"]["batch_budget"] == {"unit": "tokens", "limit": limit}
    json.dumps(failure)


@pytest.mark.parametrize("parameter", ["batch_budget", "query_budget", "context_budget"])
@pytest.mark.parametrize("budget", [
    10, {}, {"limit": 10}, {"unit": "chars"}, {"unit": "bytes", "limit": 10},
    {"unit": "chars", "limit": -1}, {"unit": "chars", "limit": True},
    {"unit": "chars", "limit": 1.5}, {"unit": "chars", "limit": "10"},
    {"unit": "chars", "limit": 10, "count_tokens": None},
    {"unit": "chars", "limit": 10, "unexpected": True},
    {"unit": "tokens", "limit": 10},
    {"unit": "tokens", "limit": 10, "count_tokens": 1},
])
async def test_invalid_budget_dicts_fail_before_processing(parameter, budget):
    if parameter == "context_budget":
        backend = Backend()
        with pytest.raises(ValueError, match=parameter):
            await search(LocalRetriever(cases=[], embeddings=[]),
                         backend=backend, context_budget=budget)
        assert not backend.calls
    else:
        backend = ExtractionBackend(lambda *args: pytest.fail("Invalid budget must not run"))
        with pytest.raises(ValueError, match=parameter):
            await extract(backend, **{parameter: budget})
        assert not backend.prompts


@pytest.mark.parametrize("parameter", ["batch_budget", "query_budget", "context_budget"])
async def test_async_counters_are_rejected_at_configuration_time(parameter):
    async def counter(text):
        return len(text)

    async def generator(text):
        yield len(text)

    class AsyncCounter:
        async def __call__(self, text):
            return len(text)

    for callback in (counter, generator, AsyncCounter()):
        budget = {"unit": "tokens", "limit": 100, "count_tokens": callback}
        with pytest.raises(ValueError, match="synchronous callable"):
            if parameter == "context_budget":
                await search(LocalRetriever(cases=[], embeddings=[]), context_budget=budget)
            else:
                await extract(ExtractionBackend(lambda *args: None), **{parameter: budget})


@pytest.mark.parametrize("parameter", ["batch_budget", "query_budget", "context_budget"])
@pytest.mark.parametrize("count", [-1, True, 1.5, "10", None])
async def test_invalid_token_counts_are_explicit_failures(parameter, count):
    budget = {"unit": "tokens", "limit": 100, "count_tokens": lambda text: count}
    if parameter == "context_budget":
        cases, rows = fixture()
        result = (await search(
            LocalRetriever(cases=cases, embeddings=rows), context_budget=budget,
        ))[0]
        assert result["error"]["type"] == "TypeError"
        assert "nonnegative integer" in result["error"]["message"]
        assert result["candidates"] == [] and result["formatted_context"] == ""
    else:
        def handler(context, prompt, invocation):
            context.query("SELECT 1")
            finish(context, {"fragments": []})

        result = await extract(ExtractionBackend(handler), **{parameter: budget})
        failure = result["failed_cases"][0]
        assert failure["error_type"] == "TypeError"
        assert "nonnegative integer" in failure["error_message"]


async def test_retrieval_counter_is_local_to_selected_candidates_and_thread_safe(encoding):
    cases, rows = fixture()
    seen = []
    main_thread = threading.get_ident()

    def counter(text):
        seen.append((text, threading.get_ident()))
        return len(encoding.encode_ordinary(text))

    budget = {"unit": "tokens", "limit": 100, "count_tokens": counter}
    result = (await search(
        LocalRetriever(cases=cases, embeddings=rows), top_k=1,
        context_budget=budget, format_case=lambda hit: hit["id"],
    ))[0]
    assert result["formatted_context"] == "a"
    assert len(seen) == 1 and seen[0][0] == "a"
    assert seen[0][1] != main_thread


async def test_hidden_async_counter_return_is_closed_and_reported():
    async def count(text):
        return len(text)

    cases, rows = fixture()
    result = (await search(
        LocalRetriever(cases=cases, embeddings=rows),
        context_budget={"unit": "tokens", "limit": 100, "count_tokens": lambda text: count(text)},
    ))[0]
    assert result["error"]["type"] == "TypeError"
    assert result["formatted_context"] == ""


@pytest.mark.parametrize("row_count", [0, 9, 10, 99, 100])
async def test_token_query_boundaries_include_row_count_digits(encoding, row_count):
    rows = [{"n": number} for number in range(1, row_count + 1)]
    expected = {"columns": ["n"], "rows": rows, "row_count": row_count}
    query = (
        "WITH RECURSIVE t(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM t WHERE n<100) "
        f"SELECT n FROM t ORDER BY n LIMIT {row_count}"
    )

    def count_tokens(text):
        return len(encoding.encode_ordinary(text))

    size = count_tokens(json.dumps(expected, ensure_ascii=False, default=str))
    for limit in (size, size - 1):
        def handler(context, prompt, count):
            response = context.query(query)
            if limit == size:
                assert response == expected
            else:
                assert response["error"] == "query_result_too_large"
                assert "rows" not in response
            assert finish(context, {"fragments": []})["ok"]

        result = await extract(ExtractionBackend(handler), query_budget={
            "unit": "tokens", "limit": limit, "count_tokens": count_tokens,
        })
        assert not result["failed_cases"], result


async def test_token_batch_retry_keeps_same_budget_and_source(encoding, monkeypatch):
    def count_tokens(text):
        return len(encoding.encode_ordinary(text))

    source = [_to_json(item) for item in sorted(data()["items"], key=lambda item: item["seq"])]
    budget = {"unit": "tokens", "limit": max(map(count_tokens, source)),
              "count_tokens": count_tokens}

    def handler(context, prompt, count):
        fragments = prompt["current_state"].get("fragments", [])
        fragments += [item["artifact_json"] for item in prompt["batch"]["items"]]
        assert finish(context, {"fragments": fragments})["ok"]
        if count == 2:
            raise TimeoutError("Retry without advancing the source")

    monkeypatch.setattr("raft.extraction.runner._retry_delay", lambda *args: 0)
    backend = ExtractionBackend(handler)
    result = await extract(backend, batch_budget=budget, retries=1)
    assert not result["failed_cases"], result
    assert backend.prompts[1] == backend.prompts[2]
    assert result["extracted_cases"][0].output.fragments == source
