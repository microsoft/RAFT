"""Restore older default string wrappers without changing their audit history."""

import json
from copy import deepcopy

import pytest
from pydantic import BaseModel, ValidationError

from raft.cases import load_cases, restore_case
from raft.defaults import CaseExtraction, case_to_text, state_to_text

NARRATIVE = (
    "AUTH_CERT_EXPIRED was returned by the login service during authentication. "
    "The engineer inspected the configured certificate and found that its validity "
    "period had expired. After rotating the certificate, the customer repeated the "
    "login request and reported that authentication succeeded."
)


def legacy_case():
    output = {
        "entities": [{"name": "AUTH_CERT_EXPIRED"}, {"name": "login-service"}],
        "timeline": [{"narrative": NARRATIVE}],
        "root_cause": None,
        "resolution_steps": None,
    }
    return {
        "id": "old-case",
        "metadata": {"source": "saved"},
        "output": output,
        "execution": {
            "handoff_notes": [],
            "revisions": [{
                "state": deepcopy(output),
                "patch": [{
                    "op": "replace", "path": "/timeline/0/narrative", "value": NARRATIVE,
                }],
            }],
        },
    }


@pytest.mark.parametrize("mixed", [False, True])
def test_legacy_wrappers_restore_as_strings_without_mutating_history(mixed):
    raw = legacy_case()
    if mixed:
        raw["output"]["entities"][1] = "login-service"
        raw["output"]["timeline"].append(f" {NARRATIVE}\n")
    before = deepcopy(raw)
    restored = restore_case(raw, CaseExtraction)

    assert restored.output.entities == ["AUTH_CERT_EXPIRED", "login-service"]
    expected = [NARRATIVE, f" {NARRATIVE}\n"] if mixed else [NARRATIVE]
    assert restored.output.timeline == expected
    assert state_to_text(restored.output) == expected
    assert case_to_text(restored.output) == expected[-1]
    assert restored.execution == before["execution"]
    assert raw == before
    saved = json.loads(restored.model_dump_json())
    assert saved["output"]["timeline"] == expected
    assert restore_case(saved, CaseExtraction) == restored
    restored.output.entities.append("another-identifier")
    restored.output.timeline.append(NARRATIVE)
    assert raw == before


def test_wrapper_and_handoff_migrations_apply_together(tmp_path):
    raw = legacy_case()
    raw["output"]["handoff_notes"] = ["Check the follow-up."]
    raw["execution"].pop("handoff_notes")
    path = tmp_path / "extraction.json"
    content = json.dumps({"extracted_cases": [raw]})
    path.write_text(content, encoding="utf-8")

    restored = load_cases(path, output_type=CaseExtraction)[0]

    assert restored.output.timeline == [NARRATIVE]
    assert restored.output.entities == ["AUTH_CERT_EXPIRED", "login-service"]
    assert restored.execution["handoff_notes"] == ["Check the follow-up."]
    assert restored.execution["revisions"] == raw["execution"]["revisions"]
    assert "handoff_notes" not in restored.output.model_dump()
    assert path.read_text(encoding="utf-8") == content


@pytest.mark.parametrize("field,item", [
    ("entities", {"name": "AUTH_CERT_EXPIRED", "kind": "error"}),
    ("entities", {"label": "AUTH_CERT_EXPIRED"}),
    ("entities", {"name": 42}),
    ("entities", {"name": "x" * 121}),
    ("timeline", {"narrative": NARRATIVE, "evidence": []}),
    ("timeline", {"text": NARRATIVE}),
    ("timeline", {"narrative": "x" * 4801}),
])
def test_migration_preserves_errors_instead_of_dropping_fields_or_invalid_text(field, item):
    raw = legacy_case()
    raw["output"][field] = [item]
    before = deepcopy(raw)

    with pytest.raises(ValidationError) as error:
        restore_case(raw, CaseExtraction)

    assert error.value.errors()[0]["loc"] == (field, 0)
    assert raw == before


class CustomEntity(BaseModel):
    name: str


class CustomEntry(BaseModel):
    narrative: str


class CustomOutput(BaseModel):
    entities: list[CustomEntity]
    timeline: list[CustomEntry]
    root_cause: str | None
    resolution_steps: str | None


class CustomDefault(CaseExtraction):
    entities: list[CustomEntity]
    timeline: list[CustomEntry]


@pytest.mark.parametrize("output_type", [CustomOutput, CustomDefault])
def test_custom_models_and_default_subclasses_keep_their_own_shapes(output_type):
    raw = legacy_case()
    before = deepcopy(raw)

    restored = restore_case(raw, output_type)

    assert type(restored.output) is output_type
    assert restored.output.entities[0].name == "AUTH_CERT_EXPIRED"
    assert restored.output.timeline[0].narrative == NARRATIVE
    assert restored.output.model_dump() == before["output"]
    assert raw == before
