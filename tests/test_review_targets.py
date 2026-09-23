"""Role-scoped tool editing, dual-target completion, and transactional review."""

import asyncio
import json
import sqlite3
from ast import literal_eval
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
from agents import Agent, RunConfig
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from test_execution import ScriptedModel, call, message

from raft import LocalPipeline, run_cases
from raft.defaults import CaseReview
from raft.extraction.context import _build_case_context
from raft.extraction.state import apply_edit
from raft.tools import edit_state, query_case_sql


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


@contextmanager
def drafts(*, reviewer=True, case_type=Case, review_type=CaseReview):
    worker = _build_case_context(
        {"id": "case", "metadata": {}, "artifacts": [{"text": "Source evidence"}]},
        id_field="id", metadata_field="metadata", artifacts_field="artifacts",
        artifact_sort_field=None, final_output_type=case_type,
    )
    worker.begin_pass({"text": "worker"}, pass_number=1)
    context = worker.for_review(
        worker.pending_state, review_output_type=review_type,
    ) if reviewer else worker
    try:
        yield context
    finally:
        if context is not worker:
            context.close()
        worker.close()


def patch(context, operations, *, target="case", finish=False):
    return apply_edit(
        context=context, target=target, patch_json=json.dumps(operations), finish_pass=finish,
    )


def set_review(context, value=None, *, finish=False):
    return patch(context, [{
        "op": "add", "path": "",
        "value": {"extractable": True, "non_extractable_reasoning": None}
        if value is None else value,
    }], target="review", finish=finish)


@pytest.mark.parametrize("target", ["review", "execution", "", None])
def test_worker_rejects_unavailable_targets_without_applying_anything(target):
    with drafts(reviewer=False) as context:
        # Even accidental schema attachment cannot grant a worker review access.
        context.review_output_type = CaseReview
        before = deepcopy((context.pending_state, context.pending_review, context.pending_edits))
        result = patch(context, [{"op": "add", "path": "", "value": {}}], target=target, finish=True)
        assert not result["ok"] and result["target"] == target
        assert "patch_applied" not in result
        assert (context.pending_state, context.pending_review, context.pending_edits) == before
        assert not context.pass_finished


def test_review_registration_error_is_explicit_and_does_not_mutate_drafts():
    with drafts(review_type=None) as context:
        for target in ("case", "review"):
            result = patch(context, [], target=target, finish=True)
            assert not result["ok"] and "review_output_type" in result["error"]
        assert context.pending_edits == [] and context.pending_review == {}


def test_review_can_be_built_incrementally_with_targeted_feedback():
    with drafts() as context:
        result = patch(context, [{"op": "add", "path": "/extractable", "value": False}],
                       target="review")
        assert result["ok"] and result["patch_applied"] and not result["state_valid"]
        assert result["target"] == "review"
        assert result["validation_errors"][0]["target"] == "review"
        assert result["validation_errors"][0]["loc"] == ["non_extractable_reasoning"]
        assert not context.pass_finished
        assert context.pending_state == {"text": "worker"}
        repaired = patch(context, [{
            "op": "add", "path": "/non_extractable_reasoning", "value": "Only scheduling.",
        }], target="review", finish=True)
        assert repaired["ok"] and repaired["pass_finished"]
        assert [edit["target"] for edit in context.pending_edits] == ["review", "review"]


@pytest.mark.parametrize("target", ["case", "review"])
def test_finish_validates_both_targets_and_locks_both(target):
    with drafts() as context:
        invalid_case = patch(context, [{"op": "remove", "path": "/text"}])
        assert not invalid_case["ok"] and invalid_case["patch_applied"]
        result = patch(context, [], target=target, finish=True)
        assert not result["ok"] and not result["state_valid"] and not context.pass_finished
        assert {error["target"] for error in result["validation_errors"]} == {"case", "review"}
        assert set_review(context)["ok"]
        # Finishing the review target must still reject an invalid case draft.
        result = patch(context, [], target="review", finish=True)
        assert {error["target"] for error in result["validation_errors"]} == {"case"}
        assert not result["pass_finished"]
        assert patch(context, [{"op": "add", "path": "/text", "value": "corrected"}])["ok"]
        assert patch(context, [], target=target, finish=True)["pass_finished"]
        before = deepcopy((context.pending_state, context.pending_review, context.pending_edits))
        for selected in ("case", "review"):
            denied = patch(context, [], target=selected)
            assert not denied["ok"] and "already been finished" in denied["error"]
        assert (context.pending_state, context.pending_review, context.pending_edits) == before


@pytest.mark.parametrize("op", ["copy", "move"])
def test_patch_cannot_read_from_the_other_target_and_is_atomic(op):
    with drafts() as context:
        result = patch(context, [
            {"op": "add", "path": "/extractable", "value": True},
            {"op": op, "from": "/text", "path": "/non_extractable_reasoning"},
        ], target="review")
        assert not result["ok"] and "patch_applied" not in result
        assert context.pending_review == {} and context.pending_edits == []
        assert context.pending_state == {"text": "worker"}
        assert not patch(context, [{"op": "remove", "path": "/review"}])["ok"]


def test_review_model_validator_feedback_is_json_safe_and_repairable():
    with drafts() as context:
        result = set_review(context, {"extractable": False, "non_extractable_reasoning": None},
                            finish=True)
        assert not result["ok"] and result["patch_applied"]
        error = result["validation_errors"][0]
        assert error["target"] == "review" and error["type"] == "value_error"
        json.dumps(result, allow_nan=False)
        assert patch(context, [{
            "op": "replace", "path": "/non_extractable_reasoning", "value": "No technical content.",
        }], target="review", finish=True)["ok"]


def test_both_validators_receive_copies_and_run_once_per_finish():
    validations = []

    class MutatesCase(Case):
        @model_validator(mode="before")
        @classmethod
        def normalize(cls, value):
            validations.append("case")
            value["text"] += " normalized"
            return value

    class MutatesReview(BaseModel):
        labels: list[str]

        @model_validator(mode="before")
        @classmethod
        def normalize(cls, value):
            validations.append("review")
            value["labels"].append("normalized")
            return value

    with drafts(case_type=MutatesCase, review_type=MutatesReview) as context:
        assert set_review(context, {"labels": ["observed"]}, finish=True)["ok"]
        assert validations == ["case", "review"]
        assert context.pending_state == {"text": "worker"}
        assert context.pending_review == {"labels": ["observed"]}
        assert context.pending_edits[0]["patch"][0]["value"] == {"labels": ["observed"]}


def test_custom_aliased_and_root_review_models():
    class Aliased(BaseModel):
        keep: bool = Field(alias="retain")

    with drafts(review_type=Aliased) as context:
        assert set_review(context, {"keep": True}, finish=True)["ok"]
    with drafts(review_type=RootModel[list[int]]) as context:
        result = set_review(context, ["bad"], finish=True)
        assert result["validation_errors"][0]["loc"] == [0]
        assert patch(context, [{"op": "replace", "path": "/0", "value": 1}],
                     target="review", finish=True)["ok"]
    with drafts(review_type=RootModel[None]) as context:
        assert patch(context, [{"op": "add", "path": "", "value": None}],
                     target="review", finish=True)["ok"]


def settings(worker_model=None, review_model=None, **overrides):
    return {
        "cases": [{"id": "case", "metadata": {}, "artifacts": [{"text": "evidence"}]}],
        "worker_agent": Agent(name="worker", tools=[query_case_sql, edit_state], model=worker_model),
        "reviewer_agent": Agent(name="reviewer", tools=[query_case_sql, edit_state], model=review_model),
        "output_type": Case, "review_output_type": CaseReview,
        "id_field": "id", "metadata_field": "metadata", "artifacts_field": "artifacts",
        "run_config": RunConfig(tracing_disabled=True), "retries": 0, "rpm": 1000,
        **overrides,
    }


def editing_call(target, value, *, finish=False, identifier="edit"):
    return call("edit_state", {
        "target": target,
        "patch_json": json.dumps([{"op": "add", "path": "", "value": value}]),
        "finish_pass": finish,
    }, identifier)


async def test_real_sdk_review_repairs_feedback_without_native_output_and_persists(tmp_path):
    from test_local_pipeline import Embeddings

    observed = []

    class ReviewModel(ScriptedModel):
        async def get_response(self, *args, **kwargs):
            assert kwargs["output_schema"] is None
            results = [item for item in kwargs["input"]
                       if item.get("type") == "function_call_output"]
            if results:
                observed.append(literal_eval(results[-1]["output"]))
            return await super().get_response(*args, **kwargs)

    worker = ScriptedModel([
        [editing_call("case", {"text": "worker"}, finish=True)], [message("Worker complete.")],
    ])
    review = ReviewModel([
        [editing_call("case", {"text": "corrected"})],
        [editing_call("review", {"extractable": False}, finish=True, identifier="assessment")],
        [call("edit_state", {
            "target": "review",
            "patch_json": '[{"op":"add","path":"/non_extractable_reasoning","value":"No reusable insight."}]',
            "finish_pass": True,
        }, "repair")],
        [message("Finished; this final text is not JSON.")],
    ])
    config = settings(worker, review)
    raw = config.pop("cases")
    pipeline = LocalPipeline(
        tmp_path, extraction=config,
        embedding={"backend": Embeddings(), "state_to_text": lambda output: [output.text]},
        bm25=False,
    )
    result = await pipeline.index(raw)
    assert not result["extraction"]["failed_cases"], result
    record = result["extraction"]["extracted_cases"][0]
    assert record.output.text == "corrected"
    assert record.review == CaseReview(
        extractable=False, non_extractable_reasoning="No reusable insight.",
    )
    assert [(item["target"], item["ok"]) for item in observed] == [
        ("case", True), ("review", False), ("review", True),
    ]
    revisions = record.execution["revisions"]
    assert [r["stage"] for r in revisions] == ["worker", "reviewer"]
    assert revisions[-1]["review"] == record.review.model_dump()
    assert [edit["target"] for edit in revisions[-1]["edits"]] == ["case", "review", "review"]
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    saved = next(iter(catalog["cases"].values()))["case"]
    assert saved["review"] == record.review.model_dump()
    assert saved["execution"]["revisions"] == revisions
    config["worker_agent"].model = ScriptedModel([])
    config["reviewer_agent"].model = ScriptedModel([])
    reopened = LocalPipeline(
        tmp_path, extraction=config,
        embedding={"backend": Embeddings(), "state_to_text": lambda output: [output.text]},
        bm25=False,
    )
    assert (await reopened.index(raw))["skipped_ids"] == ["case"]
    hit = (await reopened.retrieve(["corrected"]))["results"][0]["candidates"][0]
    assert hit["case"].review == saved["review"]


@pytest.mark.parametrize("failure", ["unfinished", "invalid_review", "invalid_case", "timeout", "commit"])
async def test_failed_review_rolls_back_both_drafts(monkeypatch, failure):
    seen = []

    async def run(agent, prompt, *, context, **kwargs):
        seen.append(context)
        if context.stage == "worker":
            assert patch(context, [{"op": "replace", "path": "", "value": {"text": "worker"}}],
                         finish=True)["ok"]
        else:
            assert patch(context, [{"op": "replace", "path": "/text", "value": "discard"}])["ok"]
            assert set_review(context)["ok"]
            if failure == "invalid_review":
                assert not patch(context, [{"op": "remove", "path": "/extractable"}],
                                 target="review", finish=True)["ok"]
            elif failure == "invalid_case":
                assert not patch(context, [{"op": "remove", "path": "/text"}], finish=True)["ok"]
            elif failure in ("timeout", "commit"):
                assert patch(context, [], target="review", finish=True)["ok"]
                if failure == "timeout":
                    raise TimeoutError("Failure after both drafts finished")
                context._writer.execute(
                    "CREATE TRIGGER reject_review BEFORE INSERT ON state_revisions "
                    "WHEN NEW.stage = 'reviewer' BEGIN SELECT RAISE(ABORT, 'commit failed'); END"
                )
        # Returning a structured value cannot bypass tool-based completion.
        return SimpleNamespace(new_items=[], final_output={"extractable": True})

    def should_keep(record):
        pytest.fail("Filtering must not run after a failed review")

    monkeypatch.setattr("raft.extraction._agent.Runner.run", run)
    result = await run_cases(**settings(should_keep=should_keep))
    assert not result["extracted_cases"] and not result["filtered_cases"]
    failed = result["failed_cases"][0]
    assert failed["output"] == failed["partial_state"] == {"text": "worker"}
    assert failed["review"] is None
    assert [r["stage"] for r in failed["execution"]["revisions"]] == ["worker"]
    assert failed["execution"]["tool_calls"][-1]["stage"] == "review"
    for context in seen:
        with pytest.raises(sqlite3.ProgrammingError):
            context.connection.execute("SELECT 1")


@pytest.mark.parametrize("failure", ["unfinished", "timeout"])
async def test_retry_restores_worker_state_and_empty_assessment(monkeypatch, failure):
    calls = []

    async def run(agent, prompt, *, context, **kwargs):
        calls.append(context.stage)
        payload = json.loads(prompt.split("\n", 1)[1])
        if context.stage == "worker":
            assert "review" not in payload and "review_output_schema" not in payload
            assert patch(context, [{"op": "add", "path": "", "value": {"text": "worker"}}],
                         finish=True)["ok"]
        else:
            assert payload["review"] == context.pending_review == {}
            assert payload["review_output_schema"] == CaseReview.model_json_schema()
            assert payload["output"] == context.pending_state == {"text": "worker"}
            assert context.pending_edits == []
            assert patch(context, [{"op": "replace", "path": "/text", "value": "reviewed"}])["ok"]
            assert set_review(context, {
                "extractable": False, "non_extractable_reasoning": "Reviewed.",
            })["ok"]
            if calls.count("reviewer") == 1:
                if failure == "timeout":
                    assert patch(context, [], target="review", finish=True)["ok"]
                    raise TimeoutError("Retry after finishing")
            else:
                assert patch(context, [], target="case", finish=True)["ok"]
        return SimpleNamespace(new_items=[], final_output="done")

    monkeypatch.setattr("raft.extraction._agent.Runner.run", run)
    monkeypatch.setattr("raft.extraction.runner._retry_delay", lambda *_: 0)
    result = await run_cases(**settings(retries=1))
    assert not result["failed_cases"], result
    assert calls == ["worker", "reviewer", "reviewer"]
    record = result["extracted_cases"][0]
    assert record.output.text == "reviewed"
    assert record.review.extractable is False
    assert [r["stage"] for r in record.execution["revisions"]] == ["worker", "reviewer"]


@pytest.mark.parametrize("invalid", [None, {}, str, CaseReview(extractable=True, non_extractable_reasoning=None)])
async def test_review_schema_is_checked_before_agents_run(invalid):
    with pytest.raises(ValueError, match="review_output_type must be a Pydantic model class"):
        await run_cases(**settings(cases=[], review_output_type=invalid))


async def test_review_schema_is_required():
    config = settings(cases=[])
    config.pop("review_output_type")
    with pytest.raises(TypeError, match="review_output_type"):
        await run_cases(**config)


async def test_concurrent_review_drafts_and_audits_are_case_local(monkeypatch):
    async def run(agent, prompt, *, context, **kwargs):
        identifier = context.case_id
        if context.stage == "worker":
            assert patch(context, [{
                "op": "add", "path": "", "value": {"text": identifier},
            }], finish=True)["ok"]
        else:
            assert patch(context, [{
                "op": "replace", "path": "/text", "value": f"reviewed-{identifier}",
            }])["ok"]
            assert set_review(context, {
                "extractable": False, "non_extractable_reasoning": identifier,
            })["ok"]
            await asyncio.sleep(0)
            assert context.pending_review["non_extractable_reasoning"] == identifier
            assert patch(context, [], target="review", finish=True)["ok"]
        return SimpleNamespace(new_items=[], final_output="done")

    monkeypatch.setattr("raft.extraction._agent.Runner.run", run)
    result = await run_cases(**settings(
        cases=[{"id": str(i), "metadata": {}, "artifacts": []} for i in range(3)],
        concurrency=3,
    ))
    assert not result["failed_cases"], result
    for record in result["extracted_cases"]:
        assert record.output.text == f"reviewed-{record.id}"
        assert record.review.non_extractable_reasoning == record.id
        revision = record.execution["revisions"][-1]
        assert revision["state"]["text"] == f"reviewed-{record.id}"
        assert revision["review"]["non_extractable_reasoning"] == record.id
