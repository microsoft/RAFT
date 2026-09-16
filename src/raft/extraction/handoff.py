"""One case-local handoff record per writing pass, with runner-owned provenance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

if TYPE_CHECKING:
    from .context import CaseContext


class ArtifactRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_position: int = Field(ge=0, strict=True)
    end_position_exclusive: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_position_exclusive <= self.start_position:
            raise ValueError("artifact_range must be a nonempty ordered range")
        return self


class HandoffNote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pass_number: int = Field(ge=1, strict=True)
    artifact_range: ArtifactRange | None
    note: str = Field(strict=True)

    @field_validator("note")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("note must be a nonblank string")
        return value


def validate_handoff_notes(notes: list[dict[str, Any]]) -> tuple[HandoffNote, ...]:
    """Keep committed records immutable and preserve their pass order."""
    if not isinstance(notes, list):
        raise ValueError("handoff_notes must be a list of records")
    records = tuple(HandoffNote.model_validate(note) for note in notes)
    if any(left.pass_number >= right.pass_number for left, right in zip(records, records[1:])):
        raise ValueError("handoff note pass numbers must be strictly increasing")
    return records


def apply_handoff_note(*, context: CaseContext, note: str) -> dict[str, Any]:
    if context.stage != "worker":
        return {"ok": False, "error": "Handoff notes are read-only in this context."}
    if context.pass_finished:
        return {"ok": False, "error": "This pass has already been finished."}
    if context._pass_number is None:
        return {"ok": False, "error": "No active worker pass is configured."}
    try:
        record = HandoffNote(
            pass_number=context._pass_number,
            artifact_range=context._artifact_range,
            note=note,
        )
    except ValidationError as exc:
        return {"ok": False, "error": str(exc)}
    context._pending_handoff_note = record
    return {
        "ok": True,
        "pass_number": record.pass_number,
        "artifact_range": record.artifact_range.model_dump() if record.artifact_range else None,
    }
