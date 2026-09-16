"""Advisory worker validation and strict final/reviewer edit feedback."""

import json
from contextlib import contextmanager
from copy import deepcopy
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

from raft.extraction.context import _build_case_context
from raft.extraction.state import apply_edit
from raft.tools import edit_state


def test_sdk_tool_description_is_role_neutral_and_includes_output_contract():
    description = edit_state.description
    assert not any(role in description.lower() for role in ("worker", "reviewer", "persona"))
    for field in (
        "ok", "error", "patch_applied", "state_valid", "validation_errors",
        "operations_applied", "pass_finished", "is_final_batch", "loc", "msg",
    ):
        assert field in description
    assert set(edit_state.params_json_schema["properties"]) == {
        "patch_json", "finish_pass", "edit_note", "evidence",
    }


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    count: int = Field(ge=0, strict=True)


@contextmanager
def draft(model=Output, *, final=False, reviewer=False, state=None):
    context = _build_case_context(
        {"id": "case", "metadata": {}, "artifacts": [{"body": "Source evidence"}]},
        id_field="id", metadata_field="metadata", artifacts_field="artifacts",
        artifact_sort_field=None, final_output_type=model,
    )
    context.begin_pass({} if state is None else state, is_final_batch=final)
    review_context = context.for_review(context.pending_state) if reviewer else None
    try:
        yield review_context or context
    finally:
        if review_context is not None:
            review_context.close()
        context.close()


def edit(context, operations, *, finish=False):
    return apply_edit(context=context, patch_json=json.dumps(operations), finish_pass=finish)


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("finish", [False, True])
def test_worker_feedback_is_advisory_except_for_final_finish(final, finish):
    operations = [{"op": "add", "path": "/title", "value": "Initial observation"}]
    with draft(final=final) as context:
        result = edit(context, operations, finish=finish)
        blocked = final and finish
        assert result["ok"] is (not blocked)
        assert result["patch_applied"] is True
        assert result["state_valid"] is False
        assert result["pass_finished"] is (finish and not blocked)
        assert result["is_final_batch"] is final
        assert context.pass_finished is result["pass_finished"]
        assert result["operations_applied"] == 1
        assert [(e["loc"], e["type"]) for e in result["validation_errors"]] == [
            (["count"], "missing")
        ]
        assert ("error" in result) is blocked
        assert context.pending_state == {"title": "Initial observation"}
        assert context.pending_edits[0]["patch"] == operations
        assert context.revisions() == []
        json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("finish", [False, True])
def test_valid_worker_edits_report_validity_and_allow_finishing(final, finish):
    with draft(final=final) as context:
        result = edit(context, [
            {"op": "add", "path": "", "value": {"title": "Confirmed finding", "count": 1}}
        ], finish=finish)
        assert result == {
            "ok": True, "patch_applied": True, "state_valid": True, "validation_errors": [],
            "operations_applied": 1, "pass_finished": finish, "is_final_batch": final,
        }


@pytest.mark.parametrize("state,error_type,location", [
    ({"title": "Report", "count": "wrong"}, "int_type", ["count"]),
    ({"title": "Report", "count": -1}, "greater_than_equal", ["count"]),
    ({"title": "Report", "count": 0, "typo": True}, "extra_forbidden", ["typo"]),
])
def test_intermediate_feedback_includes_types_constraints_and_extra_fields(state, error_type, location):
    with draft() as context:
        result = edit(context, [{"op": "add", "path": "", "value": state}])
        assert result["ok"] and not result["state_valid"]
        assert any(e["type"] == error_type and e["loc"] == location
                   for e in result["validation_errors"])
        assert context.pending_state == state


def test_invalid_final_finish_can_be_repaired_without_replaying_the_first_patch():
    with draft(final=True) as context:
        result = edit(context, [{"op": "add", "path": "/title", "value": "Report"}], finish=True)
        assert not result["ok"] and result["patch_applied"]
        assert not result["state_valid"] and not context.pass_finished
        repaired = edit(context, [{"op": "add", "path": "/count", "value": 1}], finish=True)
        assert repaired["ok"] and repaired["state_valid"] and repaired["pass_finished"]
        assert repaired["validation_errors"] == []
        assert context.pending_state == {"title": "Report", "count": 1}
        assert len(context.pending_edits) == 2
        before = deepcopy(context.pending_state), deepcopy(context.pending_edits)
        locked = edit(context, [{"op": "replace", "path": "/count", "value": 2}])
        assert not locked["ok"] and "already been finished" in locked["error"]
        assert (context.pending_state, context.pending_edits) == before


@pytest.mark.parametrize("reviewer", [False, True])
def test_empty_patch_validates_current_draft_and_blocks_invalid_final_finish(reviewer):
    with draft(final=True, reviewer=reviewer) as context:
        result = edit(context, [], finish=True)
        assert not result["ok"] and not result["state_valid"]
        assert result["operations_applied"] == 0 and result["patch_applied"]
        assert not context.pass_finished
        assert len(result["validation_errors"]) == 2


@pytest.mark.parametrize("finish", [False, True])
def test_reviewer_invalid_edit_stays_strict_and_private_until_repaired(finish):
    with draft(reviewer=True, state={"title": "Report", "count": 1}) as context:
        result = edit(context, [{"op": "remove", "path": "/count"}], finish=finish)
        assert not result["ok"] and not result["state_valid"] and result["patch_applied"]
        assert not context.pass_finished
        assert context.pending_state == {"title": "Report"}
        assert context.revisions() == []
        repaired = edit(context, [{"op": "add", "path": "/count", "value": 2}], finish=finish)
        assert repaired["ok"] and repaired["state_valid"]
        assert context.pass_finished is finish


def test_validation_runs_once_after_whole_patch_not_after_each_operation():
    validations = []

    class Counted(Output):
        @model_validator(mode="after")
        def record(self):
            validations.append(self.count)
            return self

    with draft(Counted, state={"title": "Report", "count": 1}) as context:
        result = edit(context, [
            {"op": "remove", "path": "/count"},
            {"op": "add", "path": "/count", "value": 2},
        ])
        assert result["ok"] and result["state_valid"]
        assert validations == [2]
        assert result["operations_applied"] == 2


@pytest.mark.parametrize("patch", [
    "not JSON", "{}",
    '[{"op":"add","path":"/title","value":"discarded"},{"op":"remove","path":"/missing"}]',
])
def test_invalid_patch_is_atomic_and_never_reaches_model_validation(patch):
    class MustNotValidate(Output):
        @model_validator(mode="before")
        @classmethod
        def fail_if_called(cls, value):
            pytest.fail("Invalid patches must not trigger schema validation")

    with draft(MustNotValidate) as context:
        result = apply_edit(context=context, patch_json=patch)
        assert not result["ok"]
        assert "patch_applied" not in result
        assert context.pending_state == {} and context.pending_edits == []


def test_custom_validator_errors_are_json_serializable_and_preserve_field_feedback():
    class Validated(Output):
        @field_validator("title")
        @classmethod
        def check_title(cls, value):
            if value == "bad":
                raise ValueError("Use a supported title")
            return value

    with draft(Validated) as context:
        result = edit(context, [
            {"op": "add", "path": "", "value": {"title": "bad", "count": 1}}
        ])
        assert result["ok"] and not result["state_valid"]
        errors = json.loads(json.dumps(result))["validation_errors"]
        assert errors[0]["loc"] == ["title"]
        assert errors[0]["type"] == "value_error"
        assert errors[0]["ctx"]["error"] == "Use a supported title"
        assert "Use a supported title" in errors[0]["msg"]


def test_validation_cannot_mutate_draft_or_add_defaults_to_the_edit_history():
    class Mutates(BaseModel):
        values: list[str]
        defaulted: int = 7

        @model_validator(mode="before")
        @classmethod
        def normalize(cls, value):
            value["values"].append("validator-only")
            return value

    operations = [{"op": "add", "path": "", "value": {"values": ["original"]}}]
    with draft(Mutates) as context:
        result = edit(context, operations)
        assert result["ok"] and result["state_valid"]
        assert context.pending_state == {"values": ["original"]}
        assert context.pending_edits[0]["patch"] == operations
        assert edit(context, [])["state_valid"]
        assert context.pending_state == {"values": ["original"]}


def test_root_model_feedback_and_repair():
    with draft(RootModel[list[int]], final=True) as context:
        result = edit(context, [{"op": "add", "path": "", "value": ["bad"]}])
        assert result["ok"] and not result["state_valid"]
        assert result["validation_errors"][0]["loc"] == [0]
        repaired = edit(context, [{"op": "replace", "path": "/0", "value": 5}], finish=True)
        assert repaired["state_valid"] and repaired["pass_finished"]
        assert context.pending_state == [5]


def test_alias_field_names_match_runner_validation():
    class Aliased(BaseModel):
        title: str = Field(alias="subject")

    with draft(Aliased, final=True) as context:
        result = edit(context, [{"op": "add", "path": "/title", "value": "Report"}], finish=True)
        assert result["ok"] and result["state_valid"]


def test_nested_discriminated_union_is_reported_without_treating_missing_fields_as_valid():
    class Fixed(BaseModel):
        kind: Literal["fixed"]
        resolution: str

    class Open(BaseModel):
        kind: Literal["open"]
        question: str

    class Issue(BaseModel):
        status: Annotated[Fixed | Open, Field(discriminator="kind")]

    with draft(Issue) as context:
        result = edit(context, [{"op": "add", "path": "/status", "value": {"kind": "fixed"}}])
        assert result["ok"] and not result["state_valid"]
        assert result["validation_errors"][0]["loc"] == ["status", "fixed", "resolution"]
