from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Literal

from jsonpath import JSONPatchError
from jsonpath import patch as json_patch
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .context import CaseContext


class EvidenceReference(BaseModel):
    """A source location supporting an edit, independent of the output schema."""

    model_config = ConfigDict(extra="forbid")

    artifact_position: int = Field(ge=0, strict=True)
    json_pointer: str = Field(default="", description="RFC 6901 pointer; empty means whole artifact")


def _resolve_pointer(value: Any, pointer: str) -> Any:
    if not isinstance(pointer, str) or (
        pointer and (not pointer.startswith("/") or re.search(r"~(?![01])", pointer))
    ):
        raise ValueError("json_pointer must be an RFC 6901 JSON pointer")
    if not pointer:
        return value
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token):
                raise ValueError(f"Invalid array index in json_pointer: {pointer}")
            index = int(token)
            if index >= len(value):
                raise ValueError(f"json_pointer does not resolve: {pointer}")
            value = value[index]
        elif isinstance(value, dict) and token in value:
            value = value[token]
        else:
            raise ValueError(f"json_pointer does not resolve: {pointer}")
    return value


def _evidence_metadata(context: CaseContext, evidence: list[EvidenceReference]) -> list[dict]:
    references = []
    for value in evidence:
        reference = EvidenceReference.model_validate(value)
        row = context.connection.execute(
            "SELECT artifact_json FROM artifacts WHERE position = ?", (reference.artifact_position,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown artifact_position: {reference.artifact_position}")
        _resolve_pointer(json.loads(row[0]), reference.json_pointer)
        references.append(reference.model_dump(mode="json"))
    return references


def read_draft(
    *,
    context: CaseContext,
    target: Literal["case", "review"] = "case",
    json_pointer: str = "",
) -> dict[str, Any]:
    response = {"target": target, "json_pointer": json_pointer}
    if target not in ("case", "review"):
        return {**response, "ok": False, "error": "target must be 'case' or 'review'."}
    if target == "review" and context.stage != "reviewer":
        return {
            **response, "ok": False,
            "error": "The review target is only available during review.",
        }
    if target == "review" and context.review_output_type is None:
        return {
            **response, "ok": False,
            "error": "Configure review_output_type before reading a review.",
        }
    draft = context.pending_state if target == "case" else context.pending_review
    try:
        value = _resolve_pointer(draft, json_pointer)
    except ValueError as exc:
        return {**response, "ok": False, "error": str(exc)}
    return {**response, "ok": True, "value": deepcopy(value)}


def apply_edit(
    *,
    context: CaseContext,
    patch_json: str,
    finish_pass: bool = False,
    edit_note: str | None = None,
    evidence: list[EvidenceReference] | None = None,
    target: Literal["case", "review"] = "case",
) -> dict[str, Any]:
    if target not in ("case", "review"):
        return {"ok": False, "target": target, "error": "target must be 'case' or 'review'."}
    if target == "review" and context.stage != "reviewer":
        return {
            "ok": False, "target": target,
            "error": "The review target is only available during review.",
        }
    if context.stage == "reviewer" and context.review_output_type is None and (
        target == "review" or finish_pass
    ):
        return {
            "ok": False, "target": target,
            "error": "Configure review_output_type before editing or finishing a review.",
        }
    if context.pass_finished:
        return {"ok": False, "target": target, "error": "This pass has already been finished."}

    try:
        references = _evidence_metadata(context, evidence or [])
        operations = json.loads(patch_json)
        if not isinstance(operations, list):
            raise ValueError("patch_json must encode a JSON array")
        # The patch engine can mutate an operation's value when later operations
        # edit that newly added object. Keep the original operations for history.
        draft = context.pending_state if target == "case" else context.pending_review
        updated_state = json_patch.patched(deepcopy(operations), draft)
    except (
        json.JSONDecodeError,
        JSONPatchError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
    ) as exc:
        return {"ok": False, "target": target, "error": str(exc)}

    if target == "case":
        context.pending_state = updated_state
    else:
        context.pending_review = updated_state
    context.pending_edits.append({
        "target": target, "patch": operations, "edit_note": edit_note, "evidence": references
    })

    validation_errors = []
    models = {"case": context.final_output_type}
    if context.stage == "reviewer" and context.review_output_type is not None:
        models["review"] = context.review_output_type
    targets = ("case", "review") if context.stage == "reviewer" and finish_pass else (target,)
    for checked_target in targets:
        draft = context.pending_state if checked_target == "case" else context.pending_review
        try:
            # Validation must not mutate either editable draft.
            models[checked_target].model_validate(deepcopy(draft), by_name=True)
        except ValidationError as exc:
            # Custom validator errors can contain exception objects in their context.
            validation_errors.extend(
                {**error, "target": checked_target}
                for error in json.loads(exc.json(include_url=False))
            )

    blocked = bool(validation_errors) and (
        (context.stage == "reviewer" and target == "case")
        or (finish_pass and (context.stage == "reviewer" or context.is_final_batch))
    )
    if finish_pass and not blocked:
        context.pass_finished = True

    result = {
        "ok": not blocked,
        "target": target,
        "patch_applied": True,
        "state_valid": not validation_errors,
        "validation_errors": validation_errors,
        "operations_applied": len(operations),
        "pass_finished": context.pass_finished,
        "is_final_batch": context.is_final_batch,
    }
    if blocked:
        result["error"] = "Draft validation failed; repair the targets listed in validation_errors."
    return result
