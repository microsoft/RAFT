"""Live draft reads, role permissions, and SDK edit/read interactions."""

import json
from ast import literal_eval
from copy import deepcopy

import pytest
from agents import Agent
from agents.tool_context import ToolContext
from pydantic import RootModel, model_validator
from test_execution import ScriptedModel, call, message
from test_review_targets import Case, drafts, editing_call, patch, set_review, settings

from raft import run_cases
from raft.defaults import REVIEWER_INSTRUCTIONS, WORKER_INSTRUCTIONS, CaseReview
from raft.extraction.state import read_draft
from raft.tools import read_state


def test_tool_schema_and_description_cover_live_read_contract():
    assert set(read_state.params_json_schema["properties"]) == {"target", "json_pointer"}
    assert read_state.params_json_schema["properties"]["target"]["enum"] == ["case", "review"]
    for phrase in (
        "live", "incomplete or invalid", "after completion", "RFC 6901",
        "detached value", "ok=false", "without a value", "not truncated",
    ):
        assert phrase in read_state.description
    assert "read_state" in WORKER_INSTRUCTIONS and "read_state" in REVIEWER_INSTRUCTIONS


@pytest.mark.parametrize("pointer,expected", [
    ("", {"items": [{"a/b~c": [None, False, 3.5, "text"]}], "": "empty", "~1": "escape"}),
    ("/items", [{"a/b~c": [None, False, 3.5, "text"]}]),
    ("/items/0/a~1b~0c", [None, False, 3.5, "text"]),
    ("/items/0/a~1b~0c/0", None),
    ("/items/0/a~1b~0c/1", False),
    ("/items/0/a~1b~0c/2", 3.5),
    ("/items/0/a~1b~0c/3", "text"),
    ("/", "empty"),
    ("/~01", "escape"),
])
def test_read_whole_draft_subtrees_and_values_with_pointer_escaping(pointer, expected):
    with drafts(reviewer=False) as context:
        context.pending_state = {
            "items": [{"a/b~c": [None, False, 3.5, "text"]}],
            "": "empty", "~1": "escape",
        }
        before = deepcopy(context.pending_state)
        result = read_draft(context=context, json_pointer=pointer)
        assert result == {
            "ok": True, "target": "case", "json_pointer": pointer, "value": expected,
        }
        assert context.pending_state == before
        assert context.pending_edits == [] and not context.pass_finished


@pytest.mark.parametrize("pointer", [
    "items", "#/items", "/~", "/~2", "/items/01", "/items/-1", "/items/+0",
    "/items/-", "/items/1", "/items/ 0", "/items/0/missing", "/absent",
    "/items/0/text/0", "/items/0/null/x", "/items/0/bool/x", None, 0,
])
def test_invalid_and_missing_paths_return_errors_without_mutation(pointer):
    with drafts(reviewer=False) as context:
        context.pending_state = {"items": [{"text": "hello", "null": None, "bool": True}]}
        before = deepcopy(context.pending_state)
        result = read_draft(context=context, json_pointer=pointer)
        assert not result["ok"] and result["error"]
        assert result["json_pointer"] == pointer and "value" not in result
        assert context.pending_state == before and context.pending_edits == []


def test_numeric_and_dash_object_keys_are_not_treated_as_array_indices():
    with drafts(reviewer=False) as context:
        context.pending_state = {"01": "leading zero", "-": "dash", "a%2Fb": "literal"}
        assert read_draft(context=context, json_pointer="/01")["value"] == "leading zero"
        assert read_draft(context=context, json_pointer="/-")["value"] == "dash"
        assert read_draft(context=context, json_pointer="/a%2Fb")["value"] == "literal"


@pytest.mark.parametrize("value", [None, False, 0, "plain", ["first", "second"], {}])
def test_root_model_and_scalar_drafts_need_no_case_specific_fields(value):
    with drafts(reviewer=False, case_type=RootModel[object]) as context:
        context.pending_state = deepcopy(value)
        assert read_draft(context=context)["value"] == value


@pytest.mark.parametrize("target", ["review", "execution", "metadata", None, ""])
def test_workers_cannot_read_review_or_other_context_objects(target):
    with drafts(reviewer=False) as context:
        context.review_output_type = CaseReview
        context.pending_review = {"private": "review data"}
        result = read_draft(context=context, target=target)
        assert not result["ok"] and result["error"]
        assert "value" not in result
        assert "review data" not in json.dumps(result)
        assert read_draft(context=context)["value"] == {"text": "worker"}
        assert not read_draft(context=context, json_pointer="/pending_review")["ok"]


def test_read_review_requires_configured_target():
    with drafts(review_type=None) as context:
        result = read_draft(context=context, target="review")
        assert not result["ok"] and "review_output_type" in result["error"]
        assert "value" not in result
        assert read_draft(context=context, target="case")["ok"]


def test_live_reads_include_invalid_applied_drafts_and_do_not_return_committed_state():
    with drafts() as context:
        context.commit_revision({"text": "saved"}, pass_number=None, review={})
        revision_before = context.revisions()
        result = patch(context, [{"op": "replace", "path": "/text", "value": 123}])
        assert not result["ok"] and result["patch_applied"]
        assert read_draft(context=context)["value"] == {"text": 123}
        assert read_draft(context=context, target="review")["value"] == {}
        review_result = set_review(context, {"extractable": False}, finish=True)
        assert not review_result["ok"] and review_result["patch_applied"]
        assert read_draft(context=context, target="review")["value"] == {"extractable": False}
        edits_before = deepcopy(context.pending_edits)
        for target in ("case", "review"):
            read_draft(context=context, target=target)
        assert context.pending_edits == edits_before
        assert context.revisions() == revision_before
        assert not context.pass_finished


def test_reads_do_not_invoke_validators_or_add_defaults():
    class MustNotValidate(Case):
        defaulted: int = 42

        @model_validator(mode="before")
        @classmethod
        def fail(cls, value):
            pytest.fail("read_state must not validate")

    with drafts(case_type=MustNotValidate, review_type=MustNotValidate) as context:
        context.pending_state = {"text": 1}
        context.pending_review = {}
        assert read_draft(context=context)["value"] == {"text": 1}
        assert read_draft(context=context, target="review")["value"] == {}


@pytest.mark.parametrize("target", ["case", "review"])
@pytest.mark.parametrize("pointer", ["", "/items"])
def test_returned_values_are_detached_from_drafts_and_edit_history(target, pointer):
    with drafts(review_type=RootModel[dict]) as context:
        value = {"items": [{"text": "original"}]}
        patch(context, [{"op": "add", "path": "", "value": value}], target=target)
        before = deepcopy((context.pending_state, context.pending_review, context.pending_edits))
        result = read_draft(context=context, target=target, json_pointer=pointer)
        items = result["value"]["items"] if pointer == "" else result["value"]
        items[0]["text"] = "caller mutation"
        items.append({"text": "caller append"})
        assert (context.pending_state, context.pending_review, context.pending_edits) == before


def test_completed_drafts_stay_readable_while_edits_are_locked():
    with drafts() as context:
        assert set_review(context, finish=True)["pass_finished"]
        for target in ("case", "review"):
            assert read_draft(context=context, target=target)["ok"]
            assert not patch(context, [], target=target)["ok"]
        assert context.pass_finished


async def test_sdk_tool_returns_missing_path_error_instead_of_null():
    with drafts(reviewer=False) as context:
        arguments = json.dumps({"target": "case", "json_pointer": "/missing"})
        result = await read_state.on_invoke_tool(
            ToolContext(context=context, tool_name="read_state", tool_call_id="read",
                        tool_arguments=arguments),
            arguments,
        )
        assert result["ok"] is False and "value" not in result


async def test_sdk_agents_can_read_live_drafts_and_keep_initial_state_inputs():
    received = {}
    feedback = {"worker": [], "reviewer": []}

    class InspectingModel(ScriptedModel):
        def __init__(self, role, steps):
            super().__init__(steps)
            self.role = role

        async def get_response(self, *args, **kwargs):
            inputs = kwargs["input"]
            if len(inputs) == 1:
                received[self.role] = json.loads(inputs[0]["content"].split("\n", 1)[1])
            outputs = [item for item in inputs if item.get("type") == "function_call_output"]
            if outputs:
                feedback[self.role].append(literal_eval(outputs[-1]["output"]))
            return await super().get_response(*args, **kwargs)

    def read_call(target="case", pointer="", identifier="read"):
        return call("read_state", {"target": target, "json_pointer": pointer}, identifier)

    worker_model = InspectingModel("worker", [
        [read_call(identifier="initial-worker")],
        [read_call("review", identifier="forbidden")],
        [editing_call("case", {"text": "worker"}, identifier="worker-edit")],
        [read_call(pointer="/text", identifier="worker-after-edit")],
        [call("edit_state", {"patch_json": "[]", "finish_pass": True}, "worker-finish")],
        [read_call(identifier="worker-after-finish")],
        [message("Worker complete.")],
    ])
    reviewer_model = InspectingModel("reviewer", [
        [read_call("review", identifier="initial-review")],
        [editing_call("review", {"extractable": False}, finish=True, identifier="invalid")],
        [read_call("review", identifier="read-invalid")],
        [editing_call("case", {"text": "corrected"}, identifier="correct-case")],
        [read_call(pointer="/text", identifier="read-correction")],
        [editing_call("review", {"extractable": True, "non_extractable_reasoning": None},
                      finish=True, identifier="review-finish")],
        [read_call("review", identifier="review-after-finish")],
        [message("Review complete.")],
    ])
    config = settings(worker_model, reviewer_model)
    for role in ("worker_agent", "reviewer_agent"):
        assert isinstance(config[role], Agent)
        config[role].tools.append(read_state)
    result = await run_cases(**config)
    assert not result["failed_cases"], result
    assert received["worker"]["current_state"] == {}
    assert received["worker"]["target_output_schema"] == Case.model_json_schema()
    assert "review" not in received["worker"]
    assert received["reviewer"]["output"] == {"text": "worker"}
    assert received["reviewer"]["review"] == {}
    assert received["reviewer"]["review_output_schema"] == CaseReview.model_json_schema()
    assert feedback["worker"][0]["value"] == {}
    assert feedback["worker"][1]["ok"] is False
    assert feedback["worker"][3]["value"] == "worker"
    assert feedback["worker"][-1]["value"] == {"text": "worker"}
    assert feedback["reviewer"][0]["value"] == {}
    assert feedback["reviewer"][1]["state_valid"] is False
    assert feedback["reviewer"][2]["value"] == {"extractable": False}
    assert feedback["reviewer"][4]["value"] == "corrected"
    assert feedback["reviewer"][-1]["value"] == {
        "extractable": True, "non_extractable_reasoning": None,
    }
    record = result["extracted_cases"][0]
    assert record.output.text == "corrected" and record.review.extractable
    revisions = record.execution["revisions"]
    assert len(revisions) == 2
    assert [len(revision["edits"]) for revision in revisions] == [2, 3]
    for invocation in record.execution["tool_calls"]:
        for round_ in invocation["rounds"]:
            for tool_call in round_["calls"]:
                if tool_call["name"] == "edit_state":
                    output = tool_call["output"]
                    assert not {"value", "state", "review", "current_state"} & output.keys()
                if tool_call["name"] == "read_state":
                    assert "ok" in tool_call["output"]
