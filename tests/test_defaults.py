"""Offline regression coverage for the legacy v4 defaults in the pass workflow."""

import ast
import inspect
import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest
from agents import Agent
from agents.agent_output import AgentOutputSchema
from pydantic import ValidationError

import raft.defaults as defaults
from raft import run_cases
from raft.cases import restore_case
from raft.defaults import (
    REVIEWER_INSTRUCTIONS,
    WORKER_INSTRUCTIONS,
    CaseExtraction,
    CaseReview,
    case_to_text,
    state_to_text,
)
from raft.extraction import _agent as sdk
from raft.extraction.handoff import apply_handoff_note
from raft.extraction.state import apply_edit
from raft.tools import edit_state, query_case_sql, write_handoff_note

INITIAL_NARRATIVE = (
    "Windows Server 2019 reports 0xC004F074 during activation on the customer's "
    "new server; the cause is not yet confirmed. The support engineer requested "
    "the activation log to investigate whether the configured activation host "
    "was reachable. No log results or completed connectivity checks were recorded "
    "at this stage, so the proposed connectivity explanation remained unconfirmed."
)
FINAL_NARRATIVE = (
    "Windows Server 2019 activation with 0xC004F074 remains unresolved after the "
    "customer stopped responding. Support proposed checking the configured "
    "activation host, but the record contains neither the requested logs nor "
    "evidence that the check was performed. The case was closed administratively; "
    "closure does not confirm the proposed explanation or establish a successful fix."
)


def extraction(**changes):
    return CaseExtraction(
        **{
            "entities": ["0xC004F074", "Windows Server 2019"],
            "timeline": [INITIAL_NARRATIVE, FINAL_NARRATIVE],
            "root_cause": None,
            "resolution_steps": None,
            **changes,
        }
    )


def test_default_exports():
    assert set(defaults.__all__) == {
        "CaseExtraction",
        "CaseReview",
        "WORKER_INSTRUCTIONS",
        "REVIEWER_INSTRUCTIONS",
        "state_to_text",
        "case_to_text",
        "format_case",
    }
    from raft.defaults import extraction as models
    from raft.defaults import prompts, text

    for name in ("CaseExtraction", "CaseReview"):
        assert getattr(defaults, name) is getattr(models, name)
    assert defaults.WORKER_INSTRUCTIONS is prompts.WORKER_INSTRUCTIONS
    assert defaults.REVIEWER_INSTRUCTIONS is prompts.REVIEWER_INSTRUCTIONS
    assert defaults.state_to_text is text.state_to_text
    assert defaults.case_to_text is text.case_to_text
    assert defaults.format_case is text.format_case


def test_default_retrieval_formatter_supports_arbitrary_root_state_and_json_metadata():
    from datetime import datetime, timezone

    from pydantic import RootModel

    from raft import ExtractedCase

    case = ExtractedCase(
        id=123,
        metadata={"created": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        output=RootModel[list[str]](["initial observation", "later finding"]),
        review={"assessment": "private"},
        execution={"usage": {"requests": 10}},
    )
    hit = {
        "id": case.id, "case": case, "item_index": 1, "entry_id": "entry-1",
        "score": 1.0, "cosine_similarity": 1.0, "bm25_score": None, "source": "direct",
    }
    assert json.loads(defaults.format_case(hit)) == {
        "id": 123,
        "metadata": {"created": "2026-01-01T00:00:00Z"},
        "output": ["initial observation", "later finding"],
        "item_index": 1,
    }


def test_schema_preserves_legacy_v4_fields_without_delta_protocol():
    schema = CaseExtraction.model_json_schema()
    assert set(schema["properties"]) == {
        "entities", "timeline", "root_cause", "resolution_steps"
    }
    assert set(schema["required"]) == {
        "entities", "timeline", "root_cause", "resolution_steps"
    }
    assert schema["additionalProperties"] is False
    assert "$defs" not in schema
    for field, maximum in (("entities", 120), ("timeline", 4800)):
        array = schema["properties"][field]
        assert array["type"] == "array"
        assert "minItems" not in array
        if field == "entities":
            assert array["maxItems"] == 25
        else:
            assert "maxItems" not in array
        assert array["items"] == {
            "type": "string", "maxLength": maximum,
        }
    for field, limit in (("root_cause", 4800), ("resolution_steps", 4800)):
        assert {"type": "string", "maxLength": limit} in schema["properties"][field]["anyOf"]
        assert {"type": "null"} in schema["properties"][field]["anyOf"]
        assert "default" not in schema["properties"][field]
    review = CaseReview.model_json_schema()
    assert set(review["properties"]) == {"extractable", "non_extractable_reasoning"}
    assert set(review["required"]) == set(review["properties"])
    assert review["additionalProperties"] is False


@pytest.mark.parametrize("field", ["entities", "timeline", "root_cause", "resolution_steps"])
def test_core_case_fields_must_be_explicit(field):
    data = extraction().model_dump()
    del data[field]
    with pytest.raises(ValidationError) as error:
        CaseExtraction.model_validate(data)
    assert any(e["type"] == "missing" and e["loc"] == (field,) for e in error.value.errors())


@pytest.mark.parametrize(
    "model, data",
    [
        (CaseExtraction, {"verdict": {"extractable": True}}),
        (CaseExtraction, {"is_terminal": True}),
        (CaseExtraction, {"coverage": {}}),
        (CaseExtraction, {"node_deltas": []}),
        (CaseExtraction, {"state_snapshot": "obsolete"}),
        (CaseReview, {"extractable": True, "non_extractable_reasoning": None, "timeline": []}),
    ],
)
def test_obsolete_or_misplaced_fields_are_not_silently_dropped(model, data):
    if model is CaseExtraction:
        data = {**extraction().model_dump(), **data}
    with pytest.raises(ValidationError) as error:
        model.model_validate(data)
    assert any(e["type"] == "extra_forbidden" for e in error.value.errors())


def test_identifier_text_is_preserved_and_entity_count_is_capped():
    name = r"HKLM\SYSTEM\CurrentControlSet\Services\Dfs"
    assert extraction(entities=[name]).entities == [name]
    assert extraction(entities=["x" * 120]).entities == ["x" * 120]
    assert len(extraction(entities=[f"ERROR_{i}" for i in range(25)]).entities) == 25
    with pytest.raises(ValidationError) as error:
        extraction(entities=[f"ERROR_{i}" for i in range(26)])
    assert error.value.errors()[0]["loc"] == ("entities",)
    assert error.value.errors()[0]["type"] == "too_long"


@pytest.mark.parametrize("field", ["entities", "timeline"])
@pytest.mark.parametrize("value", ["", " ", "x", " short text\n"])
def test_string_items_have_no_minimum_or_whitespace_checks(field, value):
    assert getattr(extraction(**{field: [value]}), field) == [value]


@pytest.mark.parametrize("field,value", [
    ("entities", {"name": "0xC004F074"}),
    ("timeline", {"narrative": INITIAL_NARRATIVE}),
    ("entities", 123),
    ("timeline", None),
    ("entities", True),
])
def test_default_arrays_require_strings_not_wrapped_objects(field, value):
    with pytest.raises(ValidationError) as error:
        extraction(**{field: [value]})
    assert error.value.errors()[0]["loc"] == (field, 0)
    assert error.value.errors()[0]["type"] == "string_type"


@pytest.mark.parametrize("field,value", [
    ("entities", "x" * 121),
    ("timeline", "x" * 4801),
])
def test_string_constraints_apply_to_each_item_with_indexed_errors(field, value):
    valid = extraction().model_dump()[field][0]
    with pytest.raises(ValidationError) as error:
        extraction(**{field: [valid, value]})
    assert error.value.errors()[0]["loc"] == (field, 1)


def test_string_lists_serialize_flat_and_preserve_text_verbatim():
    from raft import ExtractedCase

    entity = " 0xC004F074 "
    narrative = f" {INITIAL_NARRATIVE}\n"
    state = extraction(entities=[entity], timeline=[narrative])
    assert json.loads(state.model_dump_json()) == {
        "entities": [entity], "timeline": [narrative],
        "root_cause": None, "resolution_steps": None,
    }
    assert state_to_text(state) == [narrative]
    assert case_to_text(state) == narrative
    case = ExtractedCase(id="flat", metadata={}, output=state)
    formatted = json.loads(defaults.format_case({"case": case, "item_index": 0}))
    assert formatted["output"] == state.model_dump()
    assert formatted["item_index"] == 0
    assert "execution" not in formatted
    assert len(extraction(timeline=["x" * 4800, "y" * 4800]).timeline) == 2


@pytest.mark.parametrize("field, limit", [("root_cause", 4800), ("resolution_steps", 4800)])
def test_conclusion_limits_and_explicit_nulls(field, limit):
    assert getattr(extraction(**{field: None}), field) is None
    assert getattr(extraction(**{field: "x" * limit}), field) == "x" * limit
    assert getattr(extraction(**{field: ""}), field) == ""
    assert getattr(extraction(**{field: " \n\t"}), field) == " \n\t"
    with pytest.raises(ValidationError):
        extraction(**{field: "x" * (limit + 1)})


def test_case_output_does_not_own_handoff_notes():
    data = {"entities": [], "timeline": [], "root_cause": None, "resolution_steps": None}
    assert CaseExtraction(**data).model_dump() == data
    with pytest.raises(ValidationError, match="handoff_notes"):
        CaseExtraction(**data, handoff_notes=["Await logs."])


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"extractable": True},
        {"non_extractable_reasoning": None},
        {"extractable": False, "non_extractable_reasoning": None},
        {"extractable": False, "non_extractable_reasoning": ""},
        {"extractable": False, "non_extractable_reasoning": " \n\t"},
        {"extractable": True, "non_extractable_reasoning": "has useful guidance"},
        {"extractable": True, "non_extractable_reasoning": ""},
        {"extractable": "false", "non_extractable_reasoning": "empty"},
        {"extractable": 1, "non_extractable_reasoning": None},
    ],
)
def test_review_rejects_missing_inconsistent_or_coerced_assessments(data):
    with pytest.raises(ValidationError):
        CaseReview.model_validate(data)


@pytest.mark.parametrize("keep", [True, False])
def test_review_schema_works_as_strict_sdk_output(keep):
    schema = AgentOutputSchema(CaseReview)
    data = {
        "extractable": keep,
        "non_extractable_reasoning": None if keep else "The only artifact is an empty test event.",
    }
    review = schema.validate_json(json.dumps(data))
    assert isinstance(review, CaseReview)
    assert review.model_dump() == data
    assert schema.json_schema()["additionalProperties"] is False


def test_text_preparation_embeds_narratives_in_order_without_handoff_or_conclusions():
    state = extraction(root_cause="Confirmed cause.", resolution_steps="Accepted fix.")
    snapshot = state.model_dump()
    texts = state_to_text(state)
    assert texts == [INITIAL_NARRATIVE, FINAL_NARRATIVE]
    texts.reverse()
    assert state.model_dump() == snapshot
    assert state_to_text(state) == [INITIAL_NARRATIVE, FINAL_NARRATIVE]
    assert case_to_text(state) == "Confirmed cause.\nAccepted fix."


@pytest.mark.parametrize(
    "root, resolution, expected",
    [
        ("Confirmed cause.", None, "Confirmed cause."),
        (None, "Accepted fix.", "Accepted fix."),
        (None, None, FINAL_NARRATIVE),
    ],
)
def test_case_text_prefers_supported_conclusions_then_final_narrative(root, resolution, expected):
    assert case_to_text(extraction(root_cause=root, resolution_steps=resolution)) == expected


def test_empty_case_does_not_invent_retrieval_text():
    state = extraction(entities=[], timeline=[])
    assert state_to_text(state) == []
    assert case_to_text(state) == ""


def test_only_two_standalone_prompt_literals_without_shared_assembly():
    from raft.defaults import prompts

    module = ast.parse(inspect.getsource(prompts))
    assignments = [node for node in module.body if isinstance(node, ast.Assign)]
    assert len(module.body) == 3  # Module docstring and two independently editable prompts.
    assert [node.targets[0].id for node in assignments] == [
        "WORKER_INSTRUCTIONS", "REVIEWER_INSTRUCTIONS",
    ]
    assert all(isinstance(node.value, ast.Constant) for node in assignments)
    assert [node.value.value for node in assignments] == [
        WORKER_INSTRUCTIONS, REVIEWER_INSTRUCTIONS,
    ]


@pytest.mark.parametrize("instructions,sections,word_limit", [
    (WORKER_INSTRUCTIONS, [
        "overall_objective", "input", "run_task", "tools", "timeline", "other_fields",
    ], 1200),
    (REVIEWER_INSTRUCTIONS, [
        "overall_objective", "input", "run_task", "tools", "assessment",
    ], 550),
], ids=["worker", "reviewer"])
def test_prompts_have_role_focused_structure_and_moderate_length(instructions, sections, word_limit):
    document = ET.fromstring(f"<prompt>{instructions}</prompt>")
    assert document.text.strip().startswith("You are ")
    assert [section.tag for section in document] == sections
    assert all(section.text.strip() for section in document)
    assert all(not (section.tail or "").strip() for section in document)
    assert len(instructions.split()) <= word_limit


@pytest.mark.parametrize("instructions,objective_tag,task_tag,role_scope", [
    (WORKER_INSTRUCTIONS, "overall_objective", "run_task", [
        "supplied artifact batch", "current_state",
        "earlier handoff_notes", "Continue the existing record", "finish this pass",
        "batch.is_last=false", "batch.is_last=true",
        "including when this pass contains the whole case",
        "using both prior context and this batch",
    ]),
    (REVIEWER_INSTRUCTIONS, "overall_objective", "run_task", [
        "Review the supplied output as a whole", "correct it through edit_state",
        "return a separate CaseReview assessment",
    ]),
], ids=["worker", "reviewer"])
def test_overall_objective_is_separate_from_current_role_assignment(
    instructions, objective_tag, task_tag, role_scope,
):
    document = ET.fromstring(f"<prompt>{instructions}</prompt>")
    objective = " ".join(document.findtext(objective_tag).split())
    for concept in ("workflow", "timeline", "technical identifiers", "root-cause", "resolution"):
        assert concept in objective
    task = " ".join(document.findtext(task_tag).split())
    for phrase in role_scope:
        assert phrase in task


@pytest.mark.parametrize(
    "instructions", [WORKER_INSTRUCTIONS, REVIEWER_INSTRUCTIONS], ids=["worker", "reviewer"],
)
def test_both_roles_retain_grounding_without_duplicating_tool_reference(instructions):
    text = " ".join(instructions.lower().split())
    for phrase in (
        "verbatim", "unresolved theories",
        "evidence", "rfi/guidance", "root_cause", "resolution_steps",
        "recorded outcome", "proposed", "handoff",
    ):
        assert phrase in text
    for reference_detail in (
        "artifacts(position", "state_revisions(revision_id", "operations_applied",
        "query_result_too_large", "rfc 6901", "rfc 6902",
    ):
        assert reference_detail not in text
    for obsolete in ("interactiongraph", "nodedelta", '"node_deltas"', "is_terminal=true"):
        assert obsolete not in text


def test_worker_prioritizes_state_transitions_over_batch_boundaries():
    timeline = ET.fromstring(f"<prompt>{WORKER_INSTRUCTIONS}</prompt>").findtext("timeline")
    text = " ".join(timeline.split())
    for phrase in (
        "When starting a timeline", "earliest meaningful stage in the supplied artifacts",
        "Otherwise, continue the existing timeline", "new theory enters investigation",
        "theory is confirmed, or ruled out", "scope, or problem framing changes",
        "finding that changes understanding", "unresolved closure",
        "Continue or revise an existing entry", "purpose across batch boundaries",
        "consecutive, related artifacts", "primary embedding and retrieval content",
        "zero, one, or several entries", "guided by meaningful transitions", "same ongoing check",
        "plain string containing a self-contained prose paragraph", "independently embedded",
        "without neighboring entries", "record confirmation at the stage where it occurred",
        "Naturally explain", "available in the artifacts",
    ):
        assert phrase in text
    assert "The first entry captures the initial customer report and framing" not in text


def test_worker_preserves_current_pass_protocol():
    text = " ".join(WORKER_INSTRUCTIONS.split())
    for phrase in (
        "target_output_schema", "current_state", "metadata", "every artifact",
        "whole source artifacts", "artifact_json",
        "previously committed source coverage, advanced by the runner",
        "handoff_notes", "1-based pass_number", "zero-based artifact_range",
        "exclusive end", "Earlier notes remain read-only",
        "important working context that the case output does not capture",
        "Keep notes focused on what the next pass needs",
        "edit_note", "evidence", "write_handoff_note",
        "finish_pass=true in your last edit", "empty patch",
        "batch.is_last",
        "pass_finished=true", "plain-text confirmation",
        "The updated state is the extraction result",
    ):
        assert phrase.lower() in text.lower()


def test_worker_task_uses_requested_actions_without_validation_tutorials():
    document = ET.fromstring(f"<prompt>{WORKER_INSTRUCTIONS}</prompt>")
    assert "worker agent" in document.text
    assert "You are working in a sequential workflow" in document.findtext("overall_objective")
    text = " ".join(document.findtext("run_task").lower().split())
    for phrase in (
        "actions taken", "findings made", "changes in issue understanding",
        "shifts in the resolution approach", "finalize a coherent, complete state",
        "finish this pass",
    ):
        assert phrase in text
    assert "validation" not in text
    assert "validation" not in document.findtext("tools").lower()
    for removed in ("not proof", "do not stop early", "never invent facts to satisfy validation"):
        assert removed not in WORKER_INSTRUCTIONS.lower()


def test_worker_timeline_covers_recorded_investigation_and_customer_actions():
    text = " ".join(ET.fromstring(
        f"<prompt>{WORKER_INSTRUCTIONS}</prompt>"
    ).findtext("timeline").lower().split())
    for phrase in (
        "activities recorded in its artifacts", "investigation", "reproduction",
        "environment/configuration checks", "diagnostic queries", "evidence gathering",
        "requested or collected logs", "customer data", "hypothesis testing",
        "confirmed, ruled out, or left it unresolved", "customer guidance",
        "instructions", "sent to the customer", "responses and reported results",
        "resolution work", "patches", "workarounds", "follow-up verification",
        "proposed or requested, performed, or followed by an observed result",
    ):
        assert phrase in text


def test_worker_run_task_merges_pass_workflow_without_editor_notes():
    document = ET.fromstring(f"<prompt>{WORKER_INSTRUCTIONS}</prompt>")
    assert document.find("workflow") is None
    task = document.findtext("run_task")
    assert all(f"\n{step}. " in task for step in range(1, 6))
    assert "[" not in WORKER_INSTRUCTIONS.replace("[]", "")


def test_entities_prioritize_case_defining_identifiers_over_quantity():
    fields = ET.fromstring(f"<prompt>{WORKER_INSTRUCTIONS}</prompt>").findtext("other_fields")
    entities = " ".join(fields.split("- entities:", 1)[1].split("- root_cause:", 1)[0].split())
    assert "only distinct, important technical identifiers" in entities
    assert "essential to understanding or matching the case" in entities
    assert "Omit incidental mentions" in entities
    assert "significance over quantity" in entities
    assert "about ten" not in entities
    review = " ".join(REVIEWER_INSTRUCTIONS.split())
    assert "important, verbatim entities central to the issue" in review
    assert "remove incidental mentions" in review


def test_conclusion_guidance_distinguishes_early_drafts_from_final_accounts():
    document = ET.fromstring(f"<prompt>{WORKER_INSTRUCTIONS}</prompt>")
    fields = " ".join(document.findtext("other_fields").lower().split())
    for phrase in (
        "in early passes, root_cause and resolution_steps may remain null",
        "later worker agents can use the timeline as context when writing these summaries",
        "when batch.is_last=true, populate both fields",
        "using the accumulated state and this batch",
        "for unresolved cases, describe the available understanding",
        "brief symptom or intermediate reasoning context",
        "focus on the actual fix, mitigation, workaround, or solution",
        "include concise intermediate actions only when they clarify the resolution path",
        "proposed, performed, and successful",
    ):
        assert phrase in fields
    review = " ".join(REVIEWER_INSTRUCTIONS.lower().split())
    assert "complete both root_cause and resolution_steps" in review
    assert "focus on the actual fix, mitigation, workaround, or solution" in review
    assert "keep intermediate actions concise and relevant" in review
    assert "proposed actions from performed steps and confirmed results" in review


def test_output_fields_omit_descriptions_without_losing_constraints():
    from examples.custom_extraction import SupportCase

    for model in (CaseExtraction, CaseReview):
        assert all(field.description is None for field in model.model_fields.values())
        for properties in model.model_json_schema()["properties"].values():
            assert "description" not in properties
    for model in (CaseExtraction, SupportCase):
        for field in ("timeline", "root_cause", "resolution_steps"):
            assert model.model_fields[field].description is None
            assert "description" not in model.model_json_schema()["properties"][field]


def test_final_reviewer_retains_useful_guidance_and_separates_assessment():
    text = " ".join(REVIEWER_INSTRUCTIONS.split())
    for phrase in (
        "reviewer agent in a sequential support-case extraction workflow",
        "completed extraction after all worker passes",
        "worker_final_revision", "targeted source evidence or relevant state_revisions",
        "recover omitted information", "Notes are read-only",
        "finish_pass=false", "final assessment completes the review",
        "Return only the CaseReview fields", "partial troubleshooting, proposed fixes",
        "informational/advisory guidance", "successful resolution is not required",
        "content, not labels", "non_extractable_reasoning: null when extractable=true",
        "no usable technical insight", "set extractable=false",
        "nonblank, evidence-based paragraph",
        "Keep the corrected case record even when your assessment is negative",
    ):
        assert phrase in text


def test_reviewer_focuses_on_review_instead_of_repeating_worker_rules():
    document = ET.fromstring(f"<prompt>{REVIEWER_INSTRUCTIONS}</prompt>")
    task = " ".join(document.findtext("run_task").split())
    for phrase in (
        "Each string", "customer instructions", "unresolved theories",
        "Merge repeated stages", "restore missing details",
        "knowledge appropriate to that stage", "suspected causes and unknowns explicit",
    ):
        assert phrase in task
    for removed in ("not proof", "400-1500", "batch.is_last", "validation", "write_handoff_note"):
        assert removed not in REVIEWER_INSTRUCTIONS


@pytest.mark.asyncio
@pytest.mark.parametrize("keep", [True, False])
async def test_defaults_complete_worker_passes_then_review_without_losing_state(monkeypatch, keep):
    worker = Agent(
        name="worker", instructions=WORKER_INSTRUCTIONS,
        tools=[query_case_sql, edit_state, write_handoff_note],
    )
    reviewer = Agent(
        name="reviewer",
        instructions=REVIEWER_INSTRUCTIONS,
        tools=[query_case_sql, edit_state],
        output_type=CaseReview,
    )
    artifacts = [{"body": "Request for activation guidance."}, {"body": "No further data supplied."}]
    metadata = {"category": "RFI", "environment": "test"}
    final_root = "The customer requested activation guidance; no underlying defect was established."
    final_resolution = "No guidance or resolving action was recorded in the supplied artifacts."
    seen = []
    review_data = {
        "extractable": keep,
        "non_extractable_reasoning": (
            None if keep else "The request received no reusable guidance or technical findings."
        ),
    }

    async def run(agent, prompt, *, context, **kwargs):
        seen.append(agent.name)
        if agent is worker:
            payload = json.loads(prompt.split("Pass context:\n")[1])
            assert payload["metadata"] == metadata
            assert payload["target_output_schema"] == CaseExtraction.model_json_schema()
            assert payload["coverage"]["covered_count"] == len(seen) - 1
            assert "error" in context.query("SELECT count(*) FROM state_revisions")
            assert len(payload["batch"]["items"]) == 1
            item = payload["batch"]["items"][0]
            assert set(item) == {
                "position", "original_position", "start_char", "end_char_exclusive",
                "total_chars", "artifact_json",
            }
            assert item["start_char"] == 0
            assert item["end_char_exclusive"] == item["total_chars"]
            assert json.loads(item["artifact_json"]) == artifacts[len(seen) - 1]
            state = payload["current_state"] or {
                "entities": [], "timeline": [], "root_cause": None,
                "resolution_steps": None,
            }
            assert apply_handoff_note(context=context, note=f"Processed artifact {item['position']}.")["ok"]
            assert state["root_cause"] is None
            assert state["resolution_steps"] is None
            if payload["batch"]["is_last"]:
                # Deliberately unsupported claim for the reviewer to correct.
                state["root_cause"] = "Unverified activation-host failure."
                state["resolution_steps"] = final_resolution
            edited = apply_edit(
                context=context,
                patch_json=json.dumps([{"op": "add", "path": "", "value": state}]),
                finish_pass=True,
            )
            assert edited["ok"] and edited["pass_finished"]
            result = "Pass complete."
        else:
            assert seen == ["worker", "worker", "reviewer"]
            payload = json.loads(prompt.split("Review context:\n")[1])
            assert payload["coverage"]["complete"]
            assert payload["worker_final_revision"] == 2
            assert payload["output"]["root_cause"] == "Unverified activation-host failure."
            assert [n["note"] for n in payload["handoff_notes"]] == [
                "Processed artifact 0.", "Processed artifact 1.",
            ]
            assert context.query("SELECT count(*) AS n FROM state_revisions")["rows"] == [{"n": 2}]
            corrected = apply_edit(
                context=context,
                patch_json=json.dumps([{
                    "op": "replace", "path": "/root_cause", "value": final_root,
                }]),
            )
            assert corrected["ok"] and not corrected["pass_finished"]
            result = CaseReview(**review_data)
        return SimpleNamespace(new_items=[], final_output=result)

    monkeypatch.setattr(sdk.Runner, "run", run)

    def should_keep(record):
        assert seen == ["worker", "worker", "reviewer"]
        assert record.output.root_cause == final_root
        return record.review.extractable

    result = await run_cases(
        cases=[{"id": "guidance", "metadata": metadata, "artifacts": artifacts}],
        id_field="id",
        metadata_field="metadata",
        artifacts_field="artifacts",
        worker_agent=worker,
        reviewer_agent=reviewer,
        output_type=CaseExtraction,
        should_keep=should_keep,
        batch_budget={"unit": "chars", "limit": max(len(json.dumps(a)) for a in artifacts)},
        query_budget={"unit": "chars", "limit": 500},
        rpm=1000,
        retries=0,
    )
    assert result["failed_cases"] == []
    record = result["extracted_cases" if keep else "filtered_cases"][0]
    assert record.metadata == metadata
    assert record.output.entities == []
    assert record.output.timeline == []
    assert record.output.root_cause == final_root
    assert record.output.resolution_steps == final_resolution
    assert "handoff_notes" not in record.output.model_dump()
    assert [n["note"] for n in record.execution["handoff_notes"]] == [
        "Processed artifact 0.", "Processed artifact 1.",
    ]
    assert record.review.model_dump() == review_data
    revisions = record.execution["revisions"]
    assert [revision["stage"] for revision in revisions] == ["worker", "worker", "reviewer"]
    assert revisions[0]["state"]["root_cause"] is None
    assert revisions[0]["state"]["resolution_steps"] is None
    assert revisions[-2]["state"]["root_cause"] == "Unverified activation-host failure."
    assert revisions[-1]["state"]["root_cause"] == final_root
    assert revisions[-1]["state"]["resolution_steps"] == final_resolution
    saved = record.model_dump(mode="json")
    restored = restore_case(saved, CaseExtraction)
    assert restored.model_dump(mode="json") == saved
