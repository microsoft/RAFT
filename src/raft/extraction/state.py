from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from jsonpath import JSONPatchError
from jsonpath import patch as json_patch
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .context import CaseContext


class EvidenceReference(BaseModel):
    """A source location supporting an edit, independent of the output schema."""

    model_config = ConfigDict(extra="forbid")

    artifact_position: int = Field(ge=0, strict=True)
    json_pointer: str = Field(default="", description="RFC 6901 pointer; empty means whole artifact")


def _evidence_metadata(context: CaseContext, evidence: list[EvidenceReference]) -> list[dict]:
    references = []
    for value in evidence:
        reference = EvidenceReference.model_validate(value)
        pointer = reference.json_pointer
        if pointer and (not pointer.startswith("/") or re.search(r"~(?![01])", pointer)):
            raise ValueError("json_pointer must be an RFC 6901 JSON pointer")
        row = context.connection.execute(
            "SELECT artifact_json FROM artifacts WHERE position = ?", (reference.artifact_position,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown artifact_position: {reference.artifact_position}")
        if pointer:
            current = json.loads(row[0])
            for token in pointer[1:].split("/"):
                token = token.replace("~1", "/").replace("~0", "~")
                if isinstance(current, list):
                    if not re.fullmatch(r"0|[1-9][0-9]*", token):
                        raise ValueError(f"Invalid array index in json_pointer: {pointer}")
                    current = current[int(token)]
                elif isinstance(current, dict):
                    current = current[token]
                else:
                    raise ValueError(f"json_pointer does not resolve: {pointer}")
        references.append(reference.model_dump(mode="json"))
    return references


def apply_edit(
    *,
    context: CaseContext,
    patch_json: str,
    finish_pass: bool = False,
    note: str | None = None,
    evidence: list[EvidenceReference] | None = None,
) -> dict[str, Any]:
    if context.pass_finished:
        return {"ok": False, "error": "This pass has already been finished."}

    try:
        references = _evidence_metadata(context, evidence or [])
        operations = json.loads(patch_json)
        if not isinstance(operations, list):
            raise ValueError("patch_json must encode a JSON array")
        # The patch engine can mutate an operation's value when later operations
        # edit that newly added object. Keep the original operations for history.
        updated_state = json_patch.patched(deepcopy(operations), context.pending_state)
    except (
        json.JSONDecodeError,
        JSONPatchError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
    ) as exc:
        return {"ok": False, "error": str(exc)}

    context.pending_state = updated_state
    context.pending_edits.append({"patch": operations, "note": note, "evidence": references})

    if context.stage == "reviewer" or (finish_pass and context.is_final_batch):
        try:
            context.final_output_type.model_validate(updated_state, by_name=True)
        except ValidationError as exc:
            return {
                "ok": False,
                "patch_applied": True,
                "error": "Final state validation failed.",
                "validation_errors": exc.errors(include_url=False),
            }

    if finish_pass:
        context.pass_finished = True

    return {
        "ok": True,
        "operations_applied": len(operations),
        "pass_finished": context.pass_finished,
        "is_final_batch": context.is_final_batch,
    }
