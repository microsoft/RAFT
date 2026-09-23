"""Prebuilt case tools for caller-owned OpenAI Agents SDK agents."""

from __future__ import annotations

from typing import Any, Literal

from agents import RunContextWrapper, function_tool

from raft.extraction.context import CaseContext
from raft.extraction.handoff import apply_handoff_note
from raft.extraction.state import EvidenceReference, apply_edit, read_draft


@function_tool
async def query_case_sql(
    ctx: RunContextWrapper[CaseContext],
    query: str,
) -> dict[str, Any]:
    """Run one read-only SQLite query against the current case.

    Tables:
    - artifacts(position, original_position, sort_value, char_count, artifact_json)
    - state_revisions(revision_id, stage, pass_number, state_json, edits_json,
      handoff_notes_json, review_json):
      reviewer-only committed history. Workers cannot read this table.
      state_json is a case snapshot; edits_json contains {target, patch, edit_note, evidence}.
      review_json is the committed assessment, or SQL NULL for a worker revision.
      handoff_notes_json contains the separately committed working notes for that revision.
      List revision IDs first, then select specific JSON fields or edits as needed.
      stage is worker/reviewer; reviewer pass_number is NULL. Failed drafts are absent.

    Use SQLite JSON functions such as json_extract to inspect JSON columns.
    Results must fit query_budget as a complete serialized JSON response,
    measured in its configured characters or tokens.
    On query_result_too_large, select fewer columns, filter or paginate with
    ORDER BY and LIMIT/OFFSET, or use substr() for large fields (1-based offsets).
    No partial results or shortened fields are returned.

    Args:
        query: A single read-only SELECT or WITH query.
    """
    return ctx.context.query(query)


@function_tool
async def read_state(
    ctx: RunContextWrapper[CaseContext],
    target: Literal["case", "review"] = "case",
    json_pointer: str = "",
) -> dict[str, Any]:
    """Read the live case or review draft, or a selected value within it.

    Use after edits to inspect the current draft, including incomplete or invalid
    values. Reads do not validate, edit, finish, or commit anything and remain
    available after completion. The case target is always readable; review
    requires a review context.

    Paths use RFC 6901 JSON Pointer: "" selects the entire target, "/timeline"
    selects a field, and "/timeline/0" selects its first item. Escape literal ~
    and / in keys as ~0 and ~1. Array indices are zero-based; "-" cannot be read.

    Returns ok, target, json_pointer, and a detached value on success, including
    null when that is the stored value. An unavailable target, invalid pointer,
    or missing path returns ok=false and error without a value. Returned values
    are complete, not truncated. Request a specific path to limit response size.
    To observe an edit's result, call this after that edit has completed.

    Args:
        target: Draft to read: case (default) or review (only in a review context).
        json_pointer: Path within the selected draft; empty reads the entire draft.
    """
    return read_draft(context=ctx.context, target=target, json_pointer=json_pointer)


@function_tool
async def edit_state(
    ctx: RunContextWrapper[CaseContext],
    patch_json: str,
    finish_pass: bool = False,
    edit_note: str | None = None,
    evidence: list[EvidenceReference] | None = None,
    target: Literal["case", "review"] = "case",
) -> dict[str, Any]:
    """Apply an RFC 6902 JSON Patch to the selected case or review draft.

    Apply all patch operations atomically, then validate the resulting draft
    against its configured output model. Paths, including the root path "", are
    relative to the selected target; copy/move cannot access the other draft.
    The case target is always available; review requires a review context.
    Unavailable targets, malformed patches, invalid paths, or invalid evidence
    return ok=false and error without changing drafts.
    Create parent objects and arrays before adding nested fields or appending items.

    Incomplete drafts may be accepted with ok=true and state_valid=false.
    Validation errors block completion of the final batch; contexts requiring
    strict case validity also return ok=false for invalid case edits. Assessment
    drafts may be built incrementally. Applied patches remain in the draft after
    validation errors: repair the updated draft rather than replaying the patch.

    Set finish_pass=true to request completion of the current invocation.
    During review this validates BOTH case and review, regardless of target,
    and locks both drafts only when valid. Otherwise completion remains open
    for repair. An empty patch checks or finishes unchanged drafts. Accepted
    completion returns pass_finished=true; subsequent edits to either target
    are rejected. The runner commits completed drafts and their edit history.

    The response includes ok (request accepted, not necessarily valid state) and
    target, plus error when rejected.
    After a patch is applied, it also includes:
    - patch_applied: True even if output-model validation blocks the request.
    - state_valid: Whether all drafts checked by this call satisfy their schemas.
    - validation_errors: Empty when valid; otherwise field paths (loc), error
      types, messages (msg), offending input, target, and optional validation context.
    - operations_applied: Number of operations in the applied patch array.
    - pass_finished: Whether completion was accepted and further edits are locked.
    - is_final_batch: Whether the current batch is the last source batch.

    Args:
        patch_json: A JSON-encoded RFC 6902 patch array.
        finish_pass: Request completion of the current pass, subject to validation.
        edit_note: Optional audit explanation for this patch, not a handoff reminder.
        evidence: Optional supporting artifact positions and RFC 6901 JSON pointers.
            Empty json_pointer refers to the whole artifact. Locations are checked;
            a valid reference does not establish that the evidence supports the change.
        target: Draft to edit: case (default) or review (only in a review context).
    """
    return apply_edit(
        context=ctx.context,
        patch_json=patch_json,
        finish_pass=finish_pass,
        edit_note=edit_note,
        evidence=evidence,
        target=target,
    )


@function_tool
async def write_handoff_note(
    ctx: RunContextWrapper[CaseContext],
    note: str,
) -> dict[str, Any]:
    """Write one handoff note for the current pass without modifying case state.

    Supply a concise nonblank string with findings, unresolved questions, or checks
    to carry forward. Repeated calls replace only this pass's draft note, not older
    records. Earlier notes are immutable; add corrections or resolutions explicitly
    rather than repeating them. Notes are working context, not verified evidence.

    The runner attaches pass_number and artifact_range automatically. Positions are
    zero-based; end_position_exclusive is excluded. The range describes artifacts
    supplied to this pass, not every source read or proof supporting the note.
    A pass with no new artifacts has artifact_range=null.

    A successful pass appends at most one record with its state commit; failed
    attempts append nothing. This tool does not finish a pass or advance coverage.
    Calls in read-only contexts, after completion, or with invalid input return
    ok=false and error without changing notes. On success, returns ok=true,
    pass_number, and artifact_range. Notes are validated separately from case state.

    Args:
        note: One handoff message for this pass; previous committed records remain unchanged.
    """
    return apply_handoff_note(context=ctx.context, note=note)
