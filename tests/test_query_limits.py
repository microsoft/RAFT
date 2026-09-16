"""SQL result limits preserve data and remain independent of previous calls."""

import json
from contextlib import contextmanager

import pytest
from agent_helpers import run_cases

from raft.extraction.context import _build_case_context


@contextmanager
def case_context(limit):
    context = _build_case_context(
        {"id": "case", "meta": {}, "items": [{"text": "abcdefghij"}]},
        id_field="id", metadata_field="meta", artifacts_field="items",
        artifact_sort_field=None, max_query_chars=limit,
    )
    try:
        yield context
    finally:
        context.close()


def response_size(columns, rows):
    return len(json.dumps(
        {"columns": columns, "rows": rows, "row_count": len(rows)},
        ensure_ascii=False, default=str,
    ))


@pytest.mark.parametrize("count", [0, 1, 9, 10, 99, 100, 250])
def test_exact_serialized_boundary_includes_wrapper_and_count_digits(count):
    rows = [{"n": i} for i in range(1, count + 1)]
    limit = response_size(["n"], rows)
    query = (
        "WITH RECURSIVE t(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM t WHERE n<250) "
        f"SELECT n FROM t ORDER BY n LIMIT {count}"
    )
    with case_context(limit) as context:
        for _ in range(3):
            result = context.query(query)
            assert result == {"columns": ["n"], "rows": rows, "row_count": count}
            assert len(json.dumps(result, ensure_ascii=False)) == limit
        context.begin_pass({})
        assert context.query(query) == result
    with case_context(limit - 1) as context:
        result = context.query(query)
        assert result["error"] == "query_result_too_large"
        assert "rows" not in result


def test_json_escaping_unicode_and_column_names_count():
    query = "SELECT 'é\"' AS long_column_name"
    expected = {"columns": ["long_column_name"],
                "rows": [{"long_column_name": 'é"'}], "row_count": 1}
    limit = len(json.dumps(expected, ensure_ascii=False))
    with case_context(limit) as context:
        assert context.query(query) == expected
    with case_context(limit - 1) as context:
        assert context.query(query)["error"] == "query_result_too_large"


def test_oversized_result_has_no_partial_rows_and_pagination_recovers():
    with case_context(response_size(["n"], [{"n": 1}])) as context:
        query = "WITH t(n) AS (VALUES (1),(2),(3)) SELECT n FROM t ORDER BY n"
        result = context.query(query)
        assert set(result) == {"error", "max_query_chars", "suggestion"}
        assert result["error"] == "query_result_too_large"
        for offset in range(3):
            result = context.query(query + f" LIMIT 1 OFFSET {offset}")
            assert result["rows"] == [{"n": offset + 1}]
        assert context.pending_state == {} and not context.pass_finished


def test_oversized_field_errors_and_substr_reads_entire_field():
    with case_context(response_size(["text"], [{"text": "abc"}])) as context:
        expression = "json_extract(artifact_json, '$.text')"
        result = context.query(f"SELECT {expression} AS text FROM artifacts")
        assert result["error"] == "query_result_too_large"
        assert "rows" not in result and "substr" in result["suggestion"]
        chunks = []
        for start in (1, 4, 7, 10):
            result = context.query(f"SELECT substr({expression}, {start}, 3) AS text FROM artifacts")
            chunks.append(result["rows"][0]["text"])
        assert "".join(chunks) == "abcdefghij"


def test_tiny_limit_returns_actionable_error_and_closes_cursor():
    with case_context(1) as context:
        result = context.query("SELECT position FROM artifacts WHERE 0")
        assert result["error"] == "query_result_too_large"
        assert len(json.dumps(result)) > 1  # Errors are exempt.
        assert "error" in context.query("SELECT nonexistent FROM artifacts")
        assert "error" in context.query("DELETE FROM artifacts")
        context.max_query_chars = 100
        assert context.query("SELECT 1 AS n")["rows"] == [{"n": 1}]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "100", None])
async def test_invalid_query_limit_rejected_before_agent_preparation(limit):
    from pydantic import BaseModel

    with pytest.raises(ValueError, match="max_query_chars must be a positive integer"):
        await run_cases(
            cases=[], worker_agent=None, reviewer_agent=None, output_type=BaseModel,
            id_field="id", artifacts_field="items", metadata_field="meta",
            max_query_chars=limit,
        )
