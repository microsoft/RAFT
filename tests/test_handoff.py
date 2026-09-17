"""Worker-only, append-only handoff records with one draft note per pass."""

import asyncio
import json
import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from agent_helpers import run_cases
from agents import Agent
from pydantic import RootModel, ValidationError
from test_extraction import case, edit, options
from test_review import Review

from raft import ExtractedCase, LocalRetriever, embed_cases
from raft.defaults import CaseExtraction, format_case, state_to_text
from raft.extraction import _agent as sdk
from raft.extraction.context import _build_case_context
from raft.extraction.handoff import apply_handoff_note, validate_handoff_notes
from raft.extraction.state import apply_edit
from raft.tools import edit_state, query_case_sql, write_handoff_note


def note_record(note, number=1, start=0, end=1):
    return {
        "pass_number": number,
        "artifact_range": None if start is None else {
            "start_position": start, "end_position_exclusive": end,
        },
        "note": note,
    }


def write(context, note):
    return apply_handoff_note(context=context, note=note)


@contextmanager
def note_context(model=RootModel[dict]):
    context = _build_case_context(
        {"id": "case", "meta": {}, "items": [{"text": "first"}, {"text": "second"}]},
        id_field="id", metadata_field="meta", artifacts_field="items",
        artifact_sort_field=None, final_output_type=model,
    )
    context.begin_pass(
        {}, pass_number=1, is_final_batch=False,
        artifact_range={"start_position": 0, "end_position_exclusive": 1},
    )
    try:
        yield context
    finally:
        context.close()


def test_repeated_calls_replace_only_the_current_pass_note_and_metadata_is_automatic():
    with note_context() as context:
        result = write(context, "Initial reminder.")
        assert result == {
            "ok": True, "pass_number": 1,
            "artifact_range": {"start_position": 0, "end_position_exclusive": 1},
        }
        assert write(context, "Revised reminder.")["ok"]
        assert context.pending_handoff_notes == [note_record("Revised reminder.")]
        assert context.pending_state == {} and context.pending_edits == []
        assert not context.pass_finished and context.revisions() == []
        result["artifact_range"]["start_position"] = 999
        assert context.pending_handoff_notes[0]["artifact_range"]["start_position"] == 0


@pytest.mark.parametrize("value", [None, ["not a string"], {}, 1, False, "", " \n\t"])
def test_invalid_note_is_rejected_without_losing_existing_draft(value):
    with note_context() as context:
        assert write(context, "Keep this.")["ok"]
        result = write(context, value)
        assert not result["ok"] and result["error"]
        assert context.pending_handoff_notes == [note_record("Keep this.")]


def test_prior_records_are_immutable_and_full_history_is_carried_forward():
    with note_context() as context:
        assert write(context, "First pass.")["ok"]
        context.commit_revision({"value": "first"}, pass_number=1)
        history = context.pending_handoff_notes
        context.begin_pass(
            {"value": "first"}, handoff_notes=history, pass_number=2,
            artifact_range={"start_position": 1, "end_position_exclusive": 2},
        )
        history[0]["note"] = "Mutated caller copy"
        assert write(context, "Second pass.")["ok"]
        assert write(context, "Final second-pass note.")["ok"]
        expected = [note_record("First pass."), note_record("Final second-pass note.", 2, 1, 2)]
        assert context.pending_handoff_notes == expected
        detached = context.pending_handoff_notes
        detached[0]["note"] = "Attempt to erase old note"
        detached[1]["artifact_range"]["start_position"] = 999
        assert context.pending_handoff_notes == expected
        context.commit_revision({"value": "second"}, pass_number=2)
        revisions = context.revisions()
        assert revisions[0]["handoff_notes"] == expected[:1]
        assert revisions[1]["handoff_notes"] == expected


def test_no_note_is_optional_and_empty_artifact_range_is_explicit():
    with note_context() as context:
        assert context.pending_handoff_notes == []
        context.commit_revision({}, pass_number=1)
        context.begin_pass({}, pass_number=2, artifact_range=None)
        result = write(context, "Repaired the state; no new evidence was supplied.")
        assert result["artifact_range"] is None
        assert context.pending_handoff_notes == [
            note_record("Repaired the state; no new evidence was supplied.", 2, None)
        ]


def test_note_does_not_require_case_state_to_be_valid():
    with note_context(CaseExtraction) as context:
        assert context.pending_state == {}
        assert write(context, "Inspect later evidence.")["ok"]
        assert context.pending_state == {} and not context.pass_finished


def test_finished_pass_and_missing_pass_context_reject_writes():
    with note_context() as context:
        assert write(context, "Committed reminder.")["ok"]
        assert apply_edit(context=context, patch_json="[]", finish_pass=True)["pass_finished"]
        assert write(context, "Too late.") == {
            "ok": False, "error": "This pass has already been finished.",
        }
        context.begin_pass({})
        assert write(context, "No pass metadata.") == {
            "ok": False, "error": "No active worker pass is configured.",
        }


def test_reviewer_reads_complete_history_but_cannot_write_even_if_tool_is_attached():
    with note_context() as context:
        write(context, "Investigate this.")
        context.commit_revision({}, pass_number=1)
        reviewer = context.for_review({})
        try:
            assert reviewer.pending_handoff_notes == [note_record("Investigate this.")]
            assert write(reviewer, "Cannot change history.") == {
                "ok": False, "error": "Handoff notes are read-only in this context.",
            }
            copy = reviewer.pending_handoff_notes
            copy.clear()
            assert reviewer.pending_handoff_notes == [note_record("Investigate this.")]
            row = reviewer.query("SELECT handoff_notes_json FROM state_revisions")["rows"][0]
            assert json.loads(row["handoff_notes_json"]) == [note_record("Investigate this.")]
            assert "error" in context.query("SELECT handoff_notes_json FROM state_revisions")
        finally:
            reviewer.close()


def test_state_and_note_snapshot_commit_atomically():
    with note_context() as context:
        write(context, "Committed.")
        context.commit_revision({"value": "original"}, pass_number=1)
        committed = context.pending_handoff_notes
        context.begin_pass({"value": "original"}, handoff_notes=committed, pass_number=2)
        context._writer.execute(
            "CREATE TRIGGER reject_revision BEFORE INSERT ON state_revisions "
            "BEGIN SELECT RAISE(ABORT, 'simulated commit failure'); END"
        )
        write(context, "Uncommitted.")
        with pytest.raises(sqlite3.IntegrityError, match="simulated commit failure"):
            context.commit_revision({"value": "draft"}, pass_number=2)
        assert len(context.revisions()) == 1
        assert context.revisions()[0]["state"] == {"value": "original"}
        assert context.revisions()[0]["handoff_notes"] == [note_record("Committed.")]


def test_edit_note_is_separate_patch_audit_metadata():
    with note_context() as context:
        write(context, "Check the version next.")
        assert apply_edit(
            context=context, patch_json='[{"op":"add","path":"/finding","value":"observed"}]',
            edit_note="Recorded artifact evidence.",
            evidence=[{"artifact_position": 0, "json_pointer": "/text"}],
            finish_pass=True,
        )["ok"]
        context.commit_revision(context.pending_state, pass_number=1)
        revision = context.revisions()[0]
        assert revision["edits"][0]["edit_note"] == "Recorded artifact evidence."
        assert revision["handoff_notes"] == [note_record("Check the version next.")]
        assert revision["state"] == {"finding": "observed"}


def test_tool_takes_only_note_and_does_not_accept_model_supplied_provenance():
    schema = write_handoff_note.params_json_schema
    assert set(schema["properties"]) == {"note"}
    assert schema["properties"]["note"]["type"] == "string"
    assert schema["additionalProperties"] is False
    assert not any(role in write_handoff_note.description.lower() for role in ("worker", "reviewer"))
    assert "Repeated calls" in write_handoff_note.description
    assert "metadata" not in schema["properties"]
    assert "edit_note" in edit_state.params_json_schema["properties"]


@pytest.mark.parametrize("records", [
    ["legacy text is not runtime pass metadata"],
    [note_record("invalid", 0)],
    [note_record("wrong range", 1, 1, 1)],
    [note_record("a"), note_record("b")],
    [note_record("later", 2), note_record("earlier", 1)],
])
def test_runtime_record_validation_rejects_invalid_or_duplicate_provenance(records):
    with pytest.raises((ValueError, ValidationError)):
        validate_handoff_notes(records)


@pytest.mark.asyncio
async def test_workers_receive_entire_history_and_sorted_source_positions(monkeypatch):
    async def run(agent, prompt, *, context, **kwargs):
        payload = json.loads(prompt.split("Pass context:\n")[1])
        number = payload["pass_number"]
        expected = [] if number == 1 else [note_record(f"{context.case_id}: first")]
        assert payload["handoff_notes"] == context.pending_handoff_notes == expected
        assert "handoff_notes" not in payload["target_output_schema"]["properties"]
        item = payload["batch"]["items"][0]
        assert item["position"] == number - 1
        assert item["original_position"] == (1 if number == 1 else 0)
        response = write(context, f"{context.case_id}: {'first' if number == 1 else 'second'}")
        assert response["pass_number"] == number
        assert response["artifact_range"] == {
            "start_position": number - 1, "end_position_exclusive": number,
        }
        await asyncio.sleep(0)
        assert edit(context, [{"op": "add", "path": "", "value": {
            "extractable": True, "timeline": [context.case_id],
        }}])["ok"]
        return SimpleNamespace(new_items=[], final_output="done")

    monkeypatch.setattr(sdk.Runner, "run", run)
    result = await run_cases(cases=[case("a"), case("b")], **options(
        batch_budget={"unit": "chars", "limit": 31}, artifact_sort_field="order", concurrency=2,
    ))
    assert not result["failed_cases"], result["failed_cases"]
    for record in result["extracted_cases"]:
        assert record.execution["handoff_notes"] == [
            note_record(f"{record.id}: first"), note_record(f"{record.id}: second", 2, 1, 2),
        ]
        assert "handoff_notes" not in record.output.model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_finish", [True, False])
async def test_failed_worker_attempt_does_not_duplicate_or_erase_notes(monkeypatch, after_finish):
    failed = False
    calls = []

    async def run(agent, prompt, *, context, **kwargs):
        nonlocal failed
        payload = json.loads(prompt.split("Pass context:\n")[1])
        number = payload["pass_number"]
        calls.append(number)
        assert payload["handoff_notes"] == ([] if number == 1 else [note_record("Committed.")])
        write(context, "Committed." if number == 1 else "Recovered." if failed else "Discard.")
        assert edit(context, [{"op": "add", "path": "", "value": {
            "extractable": True, "timeline": [str(number)],
        }}], finish=(number == 1 or failed or after_finish))["ok"]
        if number == 2 and not failed:
            failed = True
            raise TimeoutError("Retry after draft edits")
        return SimpleNamespace(new_items=[], final_output="done")

    monkeypatch.setattr(sdk.Runner, "run", run)
    monkeypatch.setattr("raft.extraction.runner._retry_delay", lambda *args: 0)
    result = await run_cases(cases=[case()], **options(
        batch_budget={"unit": "chars", "limit": 31}, retries=1,
    ))
    assert not result["failed_cases"]
    record = result["extracted_cases"][0]
    assert calls == [1, 2, 2]
    assert record.execution["handoff_notes"] == [
        note_record("Committed."), note_record("Recovered.", 2, 1, 2),
    ]
    assert len(record.execution["revisions"]) == 2
    assert "Discard." not in json.dumps(record.execution["revisions"])


@pytest.mark.asyncio
async def test_terminal_worker_failure_returns_only_successful_pass_records(monkeypatch):
    async def run(agent, prompt, *, context, **kwargs):
        payload = json.loads(prompt.split("Pass context:\n")[1])
        if payload["pass_number"] == 1:
            write(context, "Committed.")
            edit(context, [{"op": "add", "path": "", "value": {
                "extractable": True, "timeline": ["committed"],
            }}])
            return SimpleNamespace(new_items=[], final_output="done")
        write(context, "Discard.")
        raise ValueError("terminal failure")

    monkeypatch.setattr(sdk.Runner, "run", run)
    result = await run_cases(cases=[case()], **options(batch_budget={"unit": "chars", "limit": 31}))
    failure = result["failed_cases"][0]
    assert failure["execution"]["handoff_notes"] == [note_record("Committed.")]
    assert failure["partial_state"]["timeline"] == ["committed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("keep", [True, False])
async def test_review_changes_state_without_changing_notes_or_adding_note_only_revision(monkeypatch, keep):
    reviewer = Agent(name="review", tools=[query_case_sql, edit_state, write_handoff_note],
                     output_type=Review)

    async def run(agent, prompt, *, context, **kwargs):
        if agent is reviewer:
            payload = json.loads(prompt.split("Review context:\n")[1])
            assert payload["handoff_notes"] == [note_record("Verify this.", 1, 0, 2)]
            assert not write(context, "Not permitted.")["ok"]
            # Mutating a detached view cannot change the committed history.
            context.pending_handoff_notes.clear()
            response = Review(keep=keep, reason="Verified.")
        else:
            write(context, "Verify this.")
            edit(context, [{"op": "add", "path": "", "value": {
                "extractable": True, "timeline": ["evidence"],
            }}])
            response = "done"
        return SimpleNamespace(new_items=[], final_output=response)

    monkeypatch.setattr(sdk.Runner, "run", run)
    result = await run_cases(cases=[case()], **options(
        reviewer_agent=reviewer, should_keep=lambda record: record.review.keep,
    ))
    assert not result["failed_cases"]
    record = result["extracted_cases" if keep else "filtered_cases"][0]
    assert record.execution["handoff_notes"] == [note_record("Verify this.", 1, 0, 2)]
    assert len(record.execution["revisions"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "invalid_state", "unstructured"])
async def test_failed_review_preserves_notes_and_worker_output(monkeypatch, failure):
    reviewer = Agent(name="review", tools=[query_case_sql, edit_state], output_type=Review)

    async def run(agent, prompt, *, context, **kwargs):
        if agent is reviewer:
            assert context.pending_handoff_notes == [note_record("Worker reminder.", 1, 0, 2)]
            if failure == "timeout":
                raise TimeoutError("review failed")
            if failure == "invalid_state":
                assert not edit(context, [{"op": "remove", "path": "/timeline"}], finish=False)["ok"]
            response = "bad" if failure == "unstructured" else Review(keep=True, reason="done")
        else:
            write(context, "Worker reminder.")
            edit(context, [{"op": "add", "path": "", "value": {
                "extractable": True, "timeline": ["worker"],
            }}])
            response = "done"
        return SimpleNamespace(new_items=[], final_output=response)

    monkeypatch.setattr(sdk.Runner, "run", run)
    result = await run_cases(cases=[case()], **options(reviewer_agent=reviewer))
    failure = result["failed_cases"][0]
    assert failure["execution"]["handoff_notes"] == [note_record("Worker reminder.", 1, 0, 2)]
    assert len(failure["execution"]["revisions"]) == 1


@pytest.mark.asyncio
async def test_review_retry_reads_same_full_history_without_repeating_worker(monkeypatch):
    reviewer = Agent(name="review", tools=[query_case_sql, edit_state], output_type=Review)
    calls = []

    async def run(agent, prompt, *, context, **kwargs):
        calls.append(agent.name)
        if agent is reviewer:
            assert context.pending_handoff_notes == [note_record("Verify.", 1, 0, 2)]
            if calls.count("review") == 1:
                raise TimeoutError("retry")
            response = Review(keep=True, reason="done")
        else:
            write(context, "Verify.")
            edit(context, [{"op": "add", "path": "", "value": {
                "extractable": True, "timeline": ["worker"],
            }}])
            response = "done"
        return SimpleNamespace(new_items=[], final_output=response)

    monkeypatch.setattr(sdk.Runner, "run", run)
    monkeypatch.setattr("raft.extraction.runner._retry_delay", lambda *args: 0)
    result = await run_cases(cases=[case()], **options(reviewer_agent=reviewer, retries=1))
    assert not result["failed_cases"] and calls == ["scan", "review", "review"]
    assert len(result["extracted_cases"][0].execution["handoff_notes"]) == 1


@pytest.mark.asyncio
async def test_empty_repair_pass_has_null_artifact_range(monkeypatch):
    async def run(agent, prompt, *, context, **kwargs):
        payload = json.loads(prompt.split("Pass context:\n")[1])
        if payload["pass_number"] == 1:
            write(context, "Original pass.")
            # Custom editors cannot bypass runner validation; force its repair pass.
            context.pending_state = {"timeline": []}
            context.pass_finished = True
        else:
            assert payload["batch"]["items"] == [] and payload["validation_error"]
            assert write(context, "Repaired schema only.")["artifact_range"] is None
            edit(context, [{"op": "add", "path": "/extractable", "value": True}])
        return SimpleNamespace(new_items=[], final_output="done")

    monkeypatch.setattr(sdk.Runner, "run", run)
    result = await run_cases(cases=[case()], **options())
    assert not result["failed_cases"]
    assert result["extracted_cases"][0].execution["handoff_notes"] == [
        note_record("Original pass.", 1, 0, 2), note_record("Repaired schema only.", 2, None),
    ]


@pytest.mark.asyncio
async def test_note_history_persists_without_entering_retrieval_text(tmp_path):
    from test_local_pipeline import Embeddings, Worker, pipeline, raw

    class WithNotes(Worker):
        async def run(self, agent, prompt, *, context, **kwargs):
            write(context, "PRIVATE-REMINDER")
            return await super().run(agent, prompt, context=context, **kwargs)

    p = pipeline(tmp_path, worker=WithNotes(), backend=Embeddings())
    await p.index([raw("case", ["technical evidence"])])
    result = (await pipeline(tmp_path).retrieve(["evidence"]))["results"][0]
    record = result["candidates"][0]["case"]
    assert record.execution["handoff_notes"] == [note_record("PRIVATE-REMINDER", 1, None)]
    assert record.execution["revisions"][0]["handoff_notes"] == record.execution["handoff_notes"]
    assert "PRIVATE-REMINDER" not in result["formatted_context"]


@pytest.mark.asyncio
async def test_default_state_embeddings_ignore_execution_notes():
    from test_local_pipeline import Embeddings

    state = CaseExtraction(
        entities=[], timeline=["Supported technical evidence. " * 10],
        root_cause=None, resolution_steps=None,
    )
    record = ExtractedCase(id="a", metadata={}, output=state,
                           execution={"handoff_notes": [note_record("PRIVATE-REMINDER")]})
    backend = Embeddings()
    result = await embed_cases(cases=[record], backend=backend, state_to_text=state_to_text)
    assert backend.calls == [[state.timeline[0]]]
    retriever = LocalRetriever.from_embeddings(result["embedded_cases"])
    hit = (await retriever.retrieve(["evidence"], backend=backend))["results"][0]["candidates"][0]
    assert "PRIVATE-REMINDER" not in format_case(hit)


@pytest.mark.asyncio
async def test_root_model_and_skipped_note_passes_need_no_reserved_output_fields(monkeypatch):
    async def run(agent, prompt, *, context, **kwargs):
        payload = json.loads(prompt.split("Pass context:\n")[1])
        if payload["pass_number"] == 2:
            assert payload["handoff_notes"] == []
            write(context, "Only the second pass needed a note.")
        edit(context, [{"op": "add", "path": "", "value": ["root item"]}])
        return SimpleNamespace(new_items=[], final_output="done")

    monkeypatch.setattr(sdk.Runner, "run", run)
    result = await run_cases(cases=[case()], **options(
        output_type=RootModel[list[str]], batch_budget={"unit": "chars", "limit": 31},
    ))
    record = result["extracted_cases"][0]
    assert record.output.root == ["root item"]
    assert record.execution["handoff_notes"] == [
        note_record("Only the second pass needed a note.", 2, 1, 2)
    ]


@pytest.mark.asyncio
async def test_cancelled_pass_does_not_commit_its_note(monkeypatch):
    started = asyncio.Event()
    contexts = []

    async def run(agent, prompt, *, context, **kwargs):
        assert context.pending_handoff_notes == []
        contexts.append(context)
        write(context, "Uncommitted.")
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(sdk.Runner, "run", run)
    task = asyncio.create_task(run_cases(cases=[case()], **options()))
    await asyncio.wait_for(started.wait(), timeout=5)
    assert contexts[0].revisions() == []
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        contexts[0].connection.execute("SELECT 1")


@pytest.mark.asyncio
async def test_legacy_snapshots_keep_original_notes_without_inventing_pass_metadata(tmp_path):
    from test_local_pipeline import Embeddings, pipeline

    from raft.storage import save_json

    state = {
        "entities": [], "timeline": [{"narrative": "Supported case evidence. " * 10}],
        "root_cause": None, "resolution_steps": None, "handoff_notes": ["LEGACY-REMINDER"],
    }
    save_json(tmp_path / "extraction.json", {
        "extracted_cases": [{"id": "legacy", "metadata": {}, "output": state}]
    })
    p = pipeline(tmp_path, backend=Embeddings())
    p.output_type = CaseExtraction
    p.extraction["output_type"] = CaseExtraction
    p.embedding["state_to_text"] = state_to_text
    p.embedding.pop("should_embed")
    assert (await p.index())["summary"]["stored_embedded"] == 1
    result = (await p.retrieve(["evidence"]))["results"][0]
    assert result["candidates"][0]["case"].execution["handoff_notes"] == ["LEGACY-REMINDER"]
    assert "LEGACY-REMINDER" not in result["formatted_context"]
