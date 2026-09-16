"""Offline regression coverage for the legacy v4 defaults in the pass workflow."""

import json
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
    Entity,
    TimelineEntry,
    case_to_text,
    state_to_text,
)
from raft.extraction import _agent as sdk
from raft.extraction.state import apply_edit
from raft.tools import edit_state, query_case_sql

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
            "entities": [{"name": "0xC004F074"}, {"name": "Windows Server 2019"}],
            "timeline": [
                {"narrative": INITIAL_NARRATIVE},
                {"narrative": FINAL_NARRATIVE},
            ],
            "root_cause": None,
            "resolution_steps": None,
            "handoff_notes": ["Check whether activation logs were ever supplied."],
            **changes,
        }
    )


def test_default_exports():
    assert set(defaults.__all__) == {
        "Entity",
        "TimelineEntry",
        "CaseExtraction",
        "CaseReview",
        "WORKER_INSTRUCTIONS",
        "REVIEWER_INSTRUCTIONS",
        "state_to_text",
        "case_to_text",
    }
    from raft.defaults import extraction as models
    from raft.defaults import prompts, text

    for name in ("Entity", "TimelineEntry", "CaseExtraction", "CaseReview"):
        assert getattr(defaults, name) is getattr(models, name)
    assert defaults.WORKER_INSTRUCTIONS is prompts.WORKER_INSTRUCTIONS
    assert defaults.REVIEWER_INSTRUCTIONS is prompts.REVIEWER_INSTRUCTIONS
    assert defaults.state_to_text is text.state_to_text
    assert defaults.case_to_text is text.case_to_text


def test_schema_preserves_legacy_v4_fields_without_delta_protocol():
    schema = CaseExtraction.model_json_schema()
    assert set(schema["properties"]) == {
        "entities", "timeline", "root_cause", "resolution_steps", "handoff_notes"
    }
    assert set(schema["required"]) == {
        "entities", "timeline", "root_cause", "resolution_steps"
    }
    assert schema["additionalProperties"] is False
    entity = schema["$defs"]["Entity"]
    assert set(entity["properties"]) == {"name"}
    assert entity["properties"]["name"]["maxLength"] == 120
    assert entity["additionalProperties"] is False
    entry = schema["$defs"]["TimelineEntry"]
    assert set(entry["properties"]) == {"narrative"}
    assert entry["properties"]["narrative"]["minLength"] == 200
    assert entry["additionalProperties"] is False
    for field, limit in (("root_cause", 800), ("resolution_steps", 1200)):
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
        (TimelineEntry, {"narrative": INITIAL_NARRATIVE, "hypotheses": []}),
        (Entity, {"name": "0xC004F074", "entity_name": "obsolete"}),
        (CaseReview, {"extractable": True, "non_extractable_reasoning": None, "timeline": []}),
    ],
)
def test_obsolete_or_misplaced_fields_are_not_silently_dropped(model, data):
    if model is CaseExtraction:
        data = {**extraction().model_dump(), **data}
    with pytest.raises(ValidationError) as error:
        model.model_validate(data)
    assert any(e["type"] == "extra_forbidden" for e in error.value.errors())


@pytest.mark.parametrize("value", ["", " ", "x" * 121])
def test_entity_name_validation(value):
    with pytest.raises(ValidationError):
        Entity(name=value)


def test_identifier_text_is_preserved_and_entity_count_is_guidance():
    name = r"HKLM\SYSTEM\CurrentControlSet\Services\Dfs"
    assert Entity(name=name).name == name
    assert Entity(name="x" * 120).name == "x" * 120
    assert len(extraction(entities=[{"name": f"ERROR_{i}"} for i in range(11)]).entities) == 11


@pytest.mark.parametrize("value", ["", "x" * 199, " " * 200, " " + "x" * 199])
def test_narrative_minimum_cannot_be_bypassed_with_empty_padding(value):
    with pytest.raises(ValidationError):
        TimelineEntry(narrative=value)


def test_narrative_minimum_is_hard_but_target_length_is_guidance():
    assert TimelineEntry(narrative="x" * 200).narrative == "x" * 200
    assert len(TimelineEntry(narrative="x" * 1600).narrative) == 1600


@pytest.mark.parametrize("field, limit", [("root_cause", 800), ("resolution_steps", 1200)])
def test_conclusion_limits_and_explicit_nulls(field, limit):
    assert getattr(extraction(**{field: None}), field) is None
    assert getattr(extraction(**{field: "x" * limit}), field) == "x" * limit
    for invalid in ("x" * (limit + 1), "", " \n\t"):
        with pytest.raises(ValidationError):
            extraction(**{field: invalid})


def test_optional_handoff_lists_are_independent_without_defaulting_case_facts():
    data = {"entities": [], "timeline": [], "root_cause": None, "resolution_steps": None}
    first, second = CaseExtraction(**data), CaseExtraction(**data)
    first.handoff_notes.append("Await logs.")
    assert second.handoff_notes == []
    assert second.model_dump() == {**data, "handoff_notes": []}


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


@pytest.mark.parametrize("instructions", [WORKER_INSTRUCTIONS, REVIEWER_INSTRUCTIONS])
def test_both_roles_receive_legacy_domain_and_grounding_guidance(instructions):
    for phrase in (
        "MATERIAL changes",
        "initial customer report",
        "2-8 entries",
        "not a quota",
        "Worker batch boundaries",
        "FIRST SENTENCE",
        "front-load verbatim identifiers",
        "embedded directly and independently",
        "without neighboring entries",
        "200 characters",
        "400-1500 characters",
        "specific commands",
        "ruled out, or confirmed",
        "evidence",
        "unresolved theories",
        "not a hard count limit",
        "120 characters",
        "800 characters",
        "1200",
        "RFI or",
    ):
        assert phrase in instructions
    for phrase in (
        "Never invent",
        "closure alone does not prove",
        "Use null",
        "not per-delta signals",
        "Correct or replace conclusions",
        "Do not use blank strings",
        "SQL returns complete results or an error, never partial results.",
        "query_result_too_large",
        "LIMIT/OFFSET",
    ):
        assert phrase in instructions
    for obsolete in ("InteractionGraph", "NodeDelta", '"node_deltas"', "is_terminal=true"):
        assert obsolete not in instructions


def test_worker_preserves_current_pass_protocol():
    for phrase in (
        "target_output_schema",
        "current_state",
        "metadata",
        "Process every item",
        "start_char",
        "end_char_exclusive",
        "total_chars",
        "artifact_json",
        "never splits",
        "or truncates",
        "current batch is not",
        "runner advances coverage automatically",
        "does not skip future batches or advance coverage",
        "Do not decide eligibility",
        "Workers cannot read revision history",
        "handoff_notes",
        "RFC 6902",
        "RFC 6901",
        "finish_pass=true in the last edit_state call",
        "patch_json='[]'",
        "batch.is_last=true",
        "repair pass, not new evidence",
        "pass_finished=true",
        "brief plain-text confirmation",
        "not your final text",
    ):
        assert phrase in WORKER_INSTRUCTIONS


def test_final_reviewer_retains_useful_guidance_and_separates_assessment():
    for phrase in (
        "state_revisions(revision_id, stage, pass_number, state_json, edits_json)",
        "worker_final_revision",
        "including deleted information",
        "not independently verified",
        "private draft",
        "finish_pass may remain false",
        "no empty edit is required",
        "further edits are locked",
        "failed review attempts leave the completed worker state unchanged",
        "RFI/guidance qualifies",
        "proposed-but-unconfirmed resolutions",
        "NOT required",
        "not sufficient reasons to reject",
        "extractable=false ONLY for pure noise",
        "non_extractable_reasoning=null",
        "Do not clear entities, timeline, conclusions",
        "separate review assessment",
        "do not duplicate the case",
    ):
        assert phrase in REVIEWER_INSTRUCTIONS


@pytest.mark.asyncio
@pytest.mark.parametrize("keep", [True, False])
async def test_defaults_complete_worker_passes_then_review_without_losing_state(monkeypatch, keep):
    worker = Agent(name="worker", instructions=WORKER_INSTRUCTIONS, tools=[query_case_sql, edit_state])
    reviewer = Agent(
        name="reviewer",
        instructions=REVIEWER_INSTRUCTIONS,
        tools=[query_case_sql, edit_state],
        output_type=CaseReview,
    )
    artifacts = [{"body": "Request for activation guidance."}, {"body": "No further data supplied."}]
    metadata = {"category": "RFI", "environment": "test"}
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
                "resolution_steps": None, "handoff_notes": [],
            }
            state["handoff_notes"].append(f"Processed artifact {item['position']}.")
            # Deliberately unconfirmed conclusion for the final reviewer to remove.
            state["root_cause"] = "Unverified activation-host failure."
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
            assert context.query("SELECT count(*) AS n FROM state_revisions")["rows"] == [{"n": 2}]
            corrected = apply_edit(
                context=context,
                patch_json='[{"op":"replace","path":"/root_cause","value":null}]',
            )
            assert corrected["ok"] and not corrected["pass_finished"]
            result = CaseReview(**review_data)
        return SimpleNamespace(new_items=[], final_output=result)

    monkeypatch.setattr(sdk.Runner, "run", run)

    def should_keep(record):
        assert seen == ["worker", "worker", "reviewer"]
        assert record.output.root_cause is None
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
        max_batch_chars=max(len(json.dumps(a)) for a in artifacts),
        max_query_chars=500,
        rpm=1000,
        retries=0,
    )
    assert result["failed_cases"] == []
    record = result["extracted_cases" if keep else "filtered_cases"][0]
    assert record.metadata == metadata
    assert record.output.entities == []
    assert record.output.timeline == []
    assert record.output.root_cause is None
    assert record.output.resolution_steps is None
    assert record.output.handoff_notes == ["Processed artifact 0.", "Processed artifact 1."]
    assert record.review.model_dump() == review_data
    revisions = record.execution["revisions"]
    assert [revision["stage"] for revision in revisions] == ["worker", "worker", "reviewer"]
    assert revisions[-2]["state"]["root_cause"] == "Unverified activation-host failure."
    assert revisions[-1]["state"]["root_cause"] is None
    saved = record.model_dump(mode="json")
    restored = restore_case(saved, CaseExtraction)
    assert restored.model_dump(mode="json") == saved
