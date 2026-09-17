import json
import subprocess
import sys
from copy import deepcopy

import pytest
from pydantic import BaseModel, ConfigDict, RootModel, ValidationError

from raft.cases import ExtractedCase, load_cases, restore_case
from raft.defaults.extraction import CaseExtraction, CaseReview


def output_data():
    return {
        "entities": ["connection_error"],
        "timeline": [],
        "root_cause": None,
        "resolution_steps": None,
    }


def legacy_case(notes):
    legacy_output = {**output_data(), "entities": [{"name": "connection_error"}]}
    initial_state = {**legacy_output, "handoff_notes": []}
    output = {**legacy_output, "handoff_notes": notes}
    return {
        "id": "legacy",
        "metadata": {"product": "example", "nested": {"tags": ["original"]}},
        "output": output,
        "review": {
            "extractable": False,
            "non_extractable_reasoning": "No confirmed reusable technical insight.",
            "audit": {"reviewer": "legacy"},
        },
        "execution": {
            "usage": {"requests": 3, "tokens": {"input": 42}},
            "tool_calls": [{"name": "edit_state", "arguments": {"note": "old"}}],
            "elapsed_seconds": 1.5,
            "passes": 2,
            "attempts": 3,
            "custom": {"status": "retained"},
            "revisions": [
                {
                    "state": initial_state,
                    "patch": [{"op": "add", "path": "", "value": deepcopy(initial_state)}],
                    "note": "Initial legacy state.",
                },
                {
                    "state": deepcopy(output),
                    "patch": [
                        {"op": "test", "path": "/handoff_notes", "value": []},
                        {"op": "replace", "path": "/handoff_notes", "value": deepcopy(notes)},
                    ],
                    "note": "Legacy audit note, not a handoff note.",
                },
            ],
        },
    }


@pytest.mark.parametrize("notes", [[], ["Check the logs.", "Follow up after restart."]])
def test_legacy_notes_migrate_without_mutating_or_rewriting_history(notes):
    raw = legacy_case(notes)
    before = deepcopy(raw)
    revisions_json = json.dumps(raw["execution"]["revisions"])

    restored = restore_case(raw, CaseExtraction)

    assert type(restored.output) is CaseExtraction
    assert restored.output.model_dump(mode="json") == output_data()
    assert restored.execution == {**before["execution"], "handoff_notes": notes}
    assert restored.metadata == before["metadata"]
    assert restored.review == before["review"]
    assert json.dumps(restored.execution["revisions"]) == revisions_json
    assert raw == before
    assert restored.execution is not raw["execution"]
    assert restored.execution["handoff_notes"] is not raw["output"]["handoff_notes"]
    restored.execution["handoff_notes"].append("A later note.")
    assert raw == before


@pytest.mark.parametrize(
    "patch",
    [
        [{"op": "add", "path": "/handoff_notes/-", "value": "Appended note."}],
        [{"op": "remove", "path": "/handoff_notes/0"}],
        [{"op": "copy", "from": "/handoff_notes/0", "path": "/handoff_notes/-"}],
        [{"op": "move", "from": "/handoff_notes/0", "path": "/handoff_notes/1"}],
        [{"op": "replace", "path": "", "value": {**output_data(), "handoff_notes": []}}],
    ],
)
def test_legacy_rfc_patch_paths_and_root_values_are_preserved_verbatim(patch):
    raw = legacy_case(["Existing note.", "Another note."])
    raw["execution"]["revisions"][-1]["patch"] = patch
    before = deepcopy(raw)

    restored = restore_case(raw, CaseExtraction)

    assert restored.execution["revisions"] == before["execution"]["revisions"]
    assert raw == before


@pytest.mark.parametrize("notes", [[], ["Preserve this note."]])
def test_legacy_notes_without_execution_keep_default_execution_fields(notes):
    raw = {"id": 1, "metadata": {}, "output": {**output_data(), "handoff_notes": notes}}
    before = deepcopy(raw)

    restored = restore_case(raw, CaseExtraction)

    assert restored.execution == {
        "usage": {},
        "tool_calls": [],
        "elapsed_seconds": 0.0,
        "passes": 0,
        "attempts": 0,
        "handoff_notes": notes,
        "revisions": [],
    }
    assert raw == before


@pytest.mark.parametrize("notes", [[], ["The same note."]])
def test_equal_legacy_and_execution_notes_can_coexist(notes):
    raw = legacy_case(notes)
    raw["execution"]["handoff_notes"] = deepcopy(notes)
    before = deepcopy(raw)

    restored = restore_case(raw, CaseExtraction)

    assert restored.execution == before["execution"]
    assert restored.execution["handoff_notes"] is not raw["execution"]["handoff_notes"]
    assert "handoff_notes" not in restored.output.model_dump()
    assert raw == before


@pytest.mark.parametrize(
    ("legacy_notes", "execution_notes"),
    [
        ([], ["Newer note."]),
        (["Older note."], []),
        (["Older note."], ["Newer note."]),
        (["First.", "Second."], ["Second.", "First."]),
        (["A note."], "A note."),
        ([], None),
    ],
)
def test_conflicting_notes_fail_without_mutation(legacy_notes, execution_notes):
    raw = legacy_case(legacy_notes)
    raw["execution"]["handoff_notes"] = execution_notes
    before = deepcopy(raw)

    with pytest.raises(
        ValueError, match=r"Conflicting output\.handoff_notes and execution\.handoff_notes"
    ):
        restore_case(raw, CaseExtraction)

    assert raw == before


@pytest.mark.parametrize(
    "notes",
    [None, False, 1, "not a list", {}, (), ("note",), [1], [None], [b"note"], ["valid", False]],
)
def test_malformed_legacy_notes_are_not_coerced(notes):
    raw = legacy_case(notes)
    before = deepcopy(raw)

    with pytest.raises(ValueError, match=r"Legacy output\.handoff_notes must be a list of strings"):
        restore_case(raw, CaseExtraction)

    assert raw == before


def test_unknown_output_extra_is_still_rejected():
    raw = legacy_case(["A valid note."])
    raw["output"]["unexpected"] = "must not be discarded"
    before = deepcopy(raw)

    with pytest.raises(ValidationError) as error:
        restore_case(raw, CaseExtraction)

    assert any(
        item["loc"] == ("unexpected",) and item["type"] == "extra_forbidden"
        for item in error.value.errors()
    )
    assert raw == before


class CustomOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    handoff_notes: list[int]


class CustomCaseExtraction(CaseExtraction):
    handoff_notes: list[int]


@pytest.mark.parametrize("output_type", [CustomOutput, CustomCaseExtraction])
def test_custom_domain_fields_including_default_subclasses_are_untouched(output_type):
    output = output_data() if output_type is CustomCaseExtraction else {}
    raw = {
        "id": "custom",
        "metadata": {},
        "output": {**output, "handoff_notes": [17]},
        "execution": {"handoff_notes": ["Separate package-owned note."]},
    }
    before = deepcopy(raw)

    restored = restore_case(raw, output_type)

    assert type(restored.output) is output_type
    assert restored.output.handoff_notes == [17]
    assert restored.execution == before["execution"]
    assert raw == before


def test_root_model_domain_field_is_untouched():
    output_type = RootModel[dict[str, list[int]]]
    raw = {
        "id": "root",
        "metadata": {},
        "output": {"handoff_notes": [23]},
        "execution": {"handoff_notes": ["A separate note."]},
    }
    before = deepcopy(raw)

    restored = restore_case(raw, output_type)

    assert type(restored.output) is output_type
    assert restored.output.root == before["output"]
    assert restored.execution == before["execution"]
    assert raw == before


@pytest.mark.parametrize("output_type", [CaseExtraction, CustomOutput, CustomCaseExtraction])
def test_live_case_and_live_output_identity_is_preserved(output_type):
    values = {} if output_type is CustomOutput else output_data()
    if output_type is not CaseExtraction:
        values["handoff_notes"] = [31]
    output = output_type.model_validate(values)
    review = CaseReview(extractable=True, non_extractable_reasoning=None)
    raw = {
        "id": "live",
        "metadata": {"source": "memory"},
        "output": output,
        "review": review,
        "execution": {"handoff_notes": ["Package-owned."]},
    }

    restored = restore_case(raw, output_type)

    assert restored.output is output
    assert restored.review is review
    assert restored.execution == raw["execution"]
    assert restore_case(restored, output_type) is restored
    assert restore_case(restored) is restored
    assert restore_case(raw).output is output


@pytest.mark.parametrize("wrapped", [False, True])
def test_load_cases_migrates_both_saved_formats(tmp_path, wrapped):
    raw = legacy_case(["Loaded legacy note."])
    payload = {"extracted_cases": [raw], "errors": []} if wrapped else [raw]
    path = tmp_path / "saved.json"
    original_json = json.dumps(payload)
    path.write_text(original_json, encoding="utf-8")

    cases = load_cases(path, output_type=CaseExtraction)

    assert len(cases) == 1
    assert cases[0].execution["handoff_notes"] == ["Loaded legacy note."]
    assert cases[0].execution["revisions"] == raw["execution"]["revisions"]
    assert cases[0].output.model_dump() == output_data()
    assert path.read_text(encoding="utf-8") == original_json


def test_roundtrip_migration_does_not_repeat_or_mutate_history():
    raw = legacy_case(["Migrate exactly once."])
    before = deepcopy(raw)
    first = restore_case(raw, CaseExtraction)
    serialized = json.loads(first.model_dump_json())
    serialized_before = deepcopy(serialized)

    second = restore_case(serialized, CaseExtraction)
    third = restore_case(json.loads(second.model_dump_json()), CaseExtraction)

    assert first == second == third
    assert second.execution["handoff_notes"] == ["Migrate exactly once."]
    assert second.execution["revisions"] == before["execution"]["revisions"]
    assert "handoff_notes" not in serialized["output"]
    assert serialized == serialized_before
    assert raw == before


def test_new_serialized_records_keep_separate_revision_notes_and_edit_notes():
    raw = {
        "id": "new",
        "metadata": {"source": "new"},
        "output": output_data(),
        "execution": {
            "handoff_notes": ["Execution note."],
            "revisions": [
                {
                    "state": output_data(),
                    "patch": [{"op": "replace", "path": "/root_cause", "value": None}],
                    "handoff_notes": ["Snapshot note."],
                    "edit_note": "New audit field.",
                }
            ],
        },
    }
    before = deepcopy(raw)

    restored = restore_case(raw, CaseExtraction)

    assert restored.execution == before["execution"]
    assert restored.model_dump(mode="json")["output"] == output_data()
    assert raw == before


def test_manually_created_cases_have_independent_default_handoff_notes():
    first = ExtractedCase(id="first", metadata={}, output=CaseExtraction(**output_data()))
    second = ExtractedCase(id="second", metadata={}, output=CaseExtraction(**output_data()))

    assert first.execution["handoff_notes"] == []
    assert second.execution["handoff_notes"] == []
    first.execution["handoff_notes"].append("First record only.")
    assert second.execution["handoff_notes"] == []


def test_core_import_does_not_load_openai_sdk():
    script = (
        "import sys; import raft.cases; "
        "assert 'openai' not in sys.modules; "
        "assert 'agents' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
