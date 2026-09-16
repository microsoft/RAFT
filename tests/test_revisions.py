"""Committed history, role-scoped SQL, and transactional reviewer corrections."""

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest
from agent_helpers import run_cases
from agents import Agent
from jsonpath import patch as json_patch
from pydantic import RootModel
from test_extraction import Output, case, edit, options
from test_query_limits import case_context
from test_review import Review

from raft.cases import restore_case
from raft.extraction import _agent as sdk
from raft.extraction.context import _build_case_context
from raft.extraction.state import EvidenceReference, apply_edit
from raft.storage import save_json
from raft.tools import edit_state, query_case_sql


def test_history_keeps_original_patch_values_when_later_operations_modify_them():
    operations = [
        {"op": "add", "path": "", "value": {"timeline": []}},
        {"op": "add", "path": "/timeline/-", "value": "one"},
    ]
    with case_context(1000) as context:
        assert apply_edit(context=context, patch_json=json.dumps(operations))["ok"]
        assert context.pending_state == {"timeline": ["one"]}
        assert context.pending_edits[0]["patch"] == operations
        context.commit_revision(context.pending_state, pass_number=1)
        revision = context.revisions()[0]
        assert json_patch.patched(revision["edits"][0]["patch"], {}) == revision["state"]


@pytest.mark.parametrize("query", [
    "SELECT * FROM state_revisions",
    "SELECT count(*) AS n FROM state_revisions",
    "SELECT 1 FROM state_revisions",
    "WITH history AS (SELECT state_json FROM state_revisions) SELECT * FROM history",
    "SELECT a.position FROM artifacts a JOIN state_revisions h ON h.revision_id = 1",
    'SELECT state_json FROM main."STATE_REVISIONS"',
])
def test_worker_cannot_read_history_even_after_reviewer_caches_query(query):
    with case_context(10_000) as context:
        context.commit_revision({"secret": "old conclusion"}, pass_number=1)
        reviewer = context.for_review({})
        try:
            for _ in range(2):
                assert "error" not in reviewer.query(query)
                assert "error" in context.query(query)
            assert context.query("SELECT position FROM artifacts")["row_count"] == 1
            for sql in (
                "DELETE FROM state_revisions RETURNING revision_id",
                "UPDATE state_revisions SET state_json = '{}' RETURNING revision_id",
                "DROP TABLE state_revisions",
                "PRAGMA query_only = OFF",
                "DELETE FROM artifacts RETURNING position",
            ):
                assert "error" in reviewer.query(sql)
            assert reviewer.query("SELECT count(*) AS n FROM state_revisions")["rows"] == [{"n": 1}]
        finally:
            reviewer.close()
        # The worker and private writer survive closing the review attempt.
        context.commit_revision({"secret": "new conclusion"}, pass_number=2)
        assert len(context.revisions()) == 2


def test_history_query_limit_and_snapshots_are_independent_of_pending_edits():
    with case_context(150) as context:
        state = {"text": "x" * 500}
        context.commit_revision(state, pass_number=1)
        state["text"] = "changed after commit"
        reviewer = context.for_review({"text": "draft"})
        try:
            result = reviewer.query("SELECT state_json FROM state_revisions")
            assert result["error"] == "query_result_too_large" and "rows" not in result
            assert reviewer.query(
                "SELECT substr(json_extract(state_json, '$.text'), 1, 5) AS text "
                "FROM state_revisions"
            )["rows"] == [{"text": "xxxxx"}]
            assert context.revisions()[0]["state"] == {"text": "x" * 500}
        finally:
            reviewer.close()


@pytest.mark.parametrize("reference", [
    {"artifact_position": 99, "json_pointer": ""},
    {"artifact_position": -1, "json_pointer": "/text"},
    {"artifact_position": 0, "json_pointer": "text"},
    {"artifact_position": 0, "json_pointer": "/missing"},
    {"artifact_position": 0, "json_pointer": "/text/missing"},
    {"artifact_position": 0, "json_pointer": "/~2"},
])
def test_bad_evidence_does_not_apply_patch_or_record_edit(reference):
    with case_context(1000) as context:
        result = apply_edit(
            context=context,
            patch_json='[{"op":"add","path":"/text","value":"new"}]',
            evidence=[reference],
        )
        assert not result["ok"]
        assert context.pending_state == {} and context.pending_edits == []


def test_evidence_supports_whole_artifacts_escaped_keys_and_array_items():
    context = _build_case_context(
        {"id": "a", "meta": {}, "items": [{"a/b~c": ["evidence"]}]},
        id_field="id", artifacts_field="items", metadata_field="meta", artifact_sort_field=None,
        final_output_type=RootModel[dict],
    )
    try:
        references = [
            EvidenceReference(artifact_position=0),
            EvidenceReference(artifact_position=0, json_pointer="/a~1b~0c/0"),
        ]
        assert apply_edit(context=context, patch_json="[]", evidence=references)["ok"]
        assert context.pending_edits[0]["evidence"] == [r.model_dump() for r in references]
    finally:
        context.close()


@pytest.mark.parametrize("keep", [True, False])
async def test_reviewer_recovers_deleted_content_and_filter_sees_corrected_state(
    monkeypatch, tmp_path, keep,
):
    reviewer = Agent(name="review", tools=[query_case_sql, edit_state], output_type=Review)

    async def run(agent, prompt, *, context, **kwargs):
        if agent is reviewer:
            payload = json.loads(prompt.split("Review context:\n")[1])
            assert payload["worker_final_revision"] == 2
            assert payload["coverage"]["complete"]
            assert payload["target_output_schema"] == Output.model_json_schema()
            assert "state_revisions" not in payload and "query_budget" not in payload
            assert payload["output"]["timeline"] == ["later"]
            rows = context.query(
                "SELECT revision_id, json_extract(state_json, '$.timeline') AS timeline "
                "FROM state_revisions ORDER BY revision_id"
            )["rows"]
            assert [json.loads(row["timeline"]) for row in rows] == [["earlier"], ["later"]]
            assert apply_edit(
                context=context,
                patch_json='[{"op":"add","path":"/timeline/0","value":"earlier"}]',
                edit_note="Restore the earlier observation after checking the source",
                evidence=[EvidenceReference(artifact_position=0, json_pointer="/text")],
            )["ok"]
            assert not context.pass_finished  # Structured response completes review.
            assert context.query("SELECT count(*) AS n FROM state_revisions")["rows"] == [{"n": 2}]
            response = Review(keep=keep, reason="Corrected chronology")
        else:
            payload = json.loads(prompt.split("Pass context:\n")[1])
            assert "error" in context.query("SELECT * FROM state_revisions")
            text = "earlier" if payload["pass_number"] == 1 else "later"
            assert edit(context, [{"op": "add", "path": "", "value": {
                "extractable": True, "timeline": [text],
            }}])["ok"]
            response = "done"
        return SimpleNamespace(new_items=[], final_output=response)

    def should_keep(record):
        assert record.output.timeline == ["earlier", "later"]
        assert record.execution["revisions"][-1]["stage"] == "reviewer"
        return record.review.keep

    monkeypatch.setattr(sdk.Runner, "run", run)
    result = await run_cases(cases=[case()], **options(
        reviewer_agent=reviewer, batch_budget={"unit": "chars", "limit": 31}, should_keep=should_keep,
    ))
    assert not result["failed_cases"], result
    record = result["extracted_cases" if keep else "filtered_cases"][0]
    history = record.execution["revisions"]
    assert [r["revision_id"] for r in history] == [1, 2, 3]
    assert [r["pass_number"] for r in history] == [1, 2, None]
    assert history[1]["state"]["timeline"] == ["later"]
    assert history[2]["state"]["timeline"] == ["earlier", "later"]
    assert history[2]["edits"][0]["evidence"] == [
        {"artifact_position": 0, "json_pointer": "/text"}
    ]
    save_json(tmp_path / "result.json", result)
    saved = json.loads((tmp_path / "result.json").read_text())
    restored = restore_case(saved["extracted_cases" if keep else "filtered_cases"][0], Output)
    assert restored.execution["revisions"] == history


@pytest.mark.parametrize("failure", ["timeout", "unstructured", "invalid_state"])
async def test_failed_review_never_commits_edits_or_filters(monkeypatch, failure):
    reviewer = Agent(name="review", tools=[query_case_sql, edit_state], output_type=Review)

    async def run(agent, prompt, *, context, **kwargs):
        if agent is not reviewer:
            edit(context, [{"op": "add", "path": "", "value": {
                "extractable": True, "timeline": ["worker"],
            }}])
            response = "done"
        else:
            assert edit(context, [{"op": "replace", "path": "/timeline/0", "value": "draft"}],
                        finish=False)["ok"]
            if failure == "timeout":
                raise TimeoutError("after correction")
            if failure == "invalid_state":
                result = edit(context, [{"op": "remove", "path": "/timeline"}], finish=False)
                assert result["patch_applied"] and not result["ok"]
                response = Review(keep=True, reason="invalid draft")
            else:
                response = "not structured"
        return SimpleNamespace(new_items=[], final_output=response)

    def should_keep(record):
        pytest.fail("filter must not run on failed review")

    monkeypatch.setattr(sdk.Runner, "run", run)
    result = await run_cases(cases=[case()], **options(
        reviewer_agent=reviewer, should_keep=should_keep,
    ))
    failed = result["failed_cases"][0]
    assert failed["stage"] == "review"
    assert failed["review"] is None
    assert failed["output"] == failed["partial_state"] == {
        "extractable": True, "timeline": ["worker"],
    }
    assert len(failed["execution"]["revisions"]) == 1


async def test_review_retry_starts_clean_and_commits_once(monkeypatch):
    reviewer = Agent(name="review", tools=[query_case_sql, edit_state], output_type=Review)
    calls = []
    drafts = []

    async def run(agent, prompt, *, context, **kwargs):
        calls.append(agent.name)
        if agent is reviewer:
            drafts.append(context)
            assert context.pending_state["timeline"] == ["worker"]
            assert context.pending_edits == [] and not context.pass_finished
            assert context.query("SELECT count(*) AS n FROM state_revisions")["rows"] == [{"n": 1}]
            assert edit(context, [{"op": "add", "path": "/timeline/-", "value": "review"}])["ok"]
            if len(drafts) == 1:
                raise TimeoutError("SDK failed after finishing edits")
            response = Review(keep=True, reason="done")
        else:
            edit(context, [{"op": "add", "path": "", "value": {
                "extractable": True, "timeline": ["worker"],
            }}])
            response = "done"
        return SimpleNamespace(new_items=[], final_output=response)

    monkeypatch.setattr(sdk.Runner, "run", run)
    monkeypatch.setattr("raft.extraction.runner._retry_delay", lambda *args: 0)
    result = await run_cases(cases=[case()], **options(reviewer_agent=reviewer, retries=1))
    assert not result["failed_cases"], result
    record = result["extracted_cases"][0]
    assert calls == ["scan", "review", "review"]
    assert record.output.timeline == ["worker", "review"]
    assert [r["stage"] for r in record.execution["revisions"]] == ["worker", "reviewer"]
    for draft in drafts:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            draft.connection.execute("SELECT 1")
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            draft._writer.execute("SELECT 1")


async def test_cancellation_during_review_closes_both_readers_and_writer(monkeypatch):
    reviewer = Agent(name="review", tools=[query_case_sql, edit_state], output_type=Review)
    entered = asyncio.Event()
    contexts = []

    async def run(agent, prompt, *, context, **kwargs):
        contexts.append(context)
        if agent is reviewer:
            entered.set()
            await asyncio.Event().wait()
        edit(context, [{"op": "add", "path": "", "value": {
            "extractable": True, "timeline": [],
        }}])
        return SimpleNamespace(new_items=[], final_output="done")

    monkeypatch.setattr(sdk.Runner, "run", run)
    task = asyncio.create_task(run_cases(cases=[case()], **options(reviewer_agent=reviewer)))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(contexts) == 2
    for context in contexts:
        for connection in (context.connection, context._writer):
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                connection.execute("SELECT 1")
