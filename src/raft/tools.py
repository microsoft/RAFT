"""Prebuilt case tools for caller-owned OpenAI Agents SDK agents."""

from __future__ import annotations

from typing import Any

from agents import RunContextWrapper, function_tool

from raft.extraction.context import CaseContext
from raft.extraction.handoff import apply_handoff_note
from raft.extraction.state import EvidenceReference, apply_edit


@function_tool
async def query_case_sql(
    ctx: RunContextWrapper[CaseContext],
    query: str,
) -> dict[str, Any]:
    """Run one read-only SQLite query against the current case.

    Tables:
    - artifacts(position, original_position, sort_value, char_count, artifact_json)
    - state_revisions(revision_id, stage, pass_number, state_json, edits_json, handoff_notes_json):
      reviewer-only committed history. Workers cannot read this table.
      state_json is a snapshot; edits_json is an array of {patch, edit_note, evidence}.
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
async def edit_state(
    ctx: RunContextWrapper[CaseContext],
    patch_json: str,
    finish_pass: bool = False,
    edit_note: str | None = None,
    evidence: list[EvidenceReference] | None = None,
) -> dict[str, Any]:
    """Apply an RFC 6902 JSON Patch to the current draft state.

    Apply all patch operations atomically, then validate the resulting draft
    against the configured output model. Malformed patches, invalid paths, or
    invalid evidence references return ok=false and error without changing state.
    Create parent objects and arrays before adding nested fields or appending items.

    Incomplete drafts may be accepted with ok=true and state_valid=false.
    Validation errors block completion of the final batch; contexts requiring
    validity on every edit also return ok=false for invalid drafts. The patch
    remains applied in either case: repair the updated draft rather than replaying
    the original patch.

    Set finish_pass=true to request completion of the current pass. Accepted
    completion returns pass_finished=true and locks further edits; blocked
    completion leaves the pass open for repair. An empty patch array checks or
    finishes the unchanged draft. Finishing a pass does not necessarily finish the
    case. The runner controls coverage, advancement, and committing the draft
    and edit history.

    The response includes ok (request accepted, not necessarily valid state) and
    error when rejected, including attempts to edit an already-finished pass.
    After a patch is applied, it also includes:
    - patch_applied: True even if output-model validation blocks the request.
    - state_valid: Whether the draft satisfies the configured output model.
    - validation_errors: Empty when valid; otherwise field paths (loc), error
      types, messages (msg), offending input, and optional validation context.
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

    """
    return apply_edit(
        context=ctx.context,
        patch_json=patch_json,
        finish_pass=finish_pass,
        edit_note=edit_note,
        evidence=evidence,
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
