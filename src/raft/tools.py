"""Prebuilt case tools for caller-owned OpenAI Agents SDK agents."""

from __future__ import annotations

from typing import Any

from agents import RunContextWrapper, function_tool

from raft.extraction.context import CaseContext
from raft.extraction.state import EvidenceReference, apply_edit


@function_tool
async def query_case_sql(
    ctx: RunContextWrapper[CaseContext],
    query: str,
) -> dict[str, Any]:
    """Run one read-only SQLite query against the current case.

    Tables:
    - artifacts(position, original_position, sort_value, char_count, artifact_json)
    - state_revisions(revision_id, stage, pass_number, state_json, edits_json):
      reviewer-only committed history. Workers cannot read this table.
      state_json is a snapshot; edits_json is an array of {patch, note, evidence}.
      List revision IDs first, then select specific JSON fields or edits as needed.
      stage is worker/reviewer; reviewer pass_number is NULL. Failed drafts are absent.

    Use SQLite JSON functions such as json_extract to inspect JSON columns. Results must fit max_query_chars as a complete serialized JSON response.
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
    note: str | None = None,
    evidence: list[EvidenceReference] | None = None,
) -> dict[str, Any]:
    """Apply an RFC 6902 JSON Patch to pass-local extraction state.

    Call this as often as needed. Once all supplied batch content has been processed,
    set finish_pass to true. The runner owns coverage and selects the next batch.
    Final Pydantic validation is required on the last batch. Use an empty patch
    array when only finishing the pass.

    Reviewers edit a private draft of the completed output. Each edit reports
    schema errors for repair. Returning the structured review completes review;
    finish_pass can remain false, and no empty edit is needed for unchanged output.
    A true finish_pass locks further edits. Edits commit only after SDK success.
    Applied patches and optional source metadata are saved in revision history.

    Args:
        patch_json: A JSON-encoded RFC 6902 patch array.
        finish_pass: Whether this call completes the current outer-loop pass.
        note: Optional brief explanation of the change.
        evidence: Optional supporting artifact positions and RFC 6901 JSON pointers.
            Empty json_pointer refers to the whole artifact. Locations are checked,
            but the reviewer must assess whether the evidence supports the change.
    """
    return apply_edit(
        context=ctx.context,
        patch_json=patch_json,
        finish_pass=finish_pass,
        note=note,
        evidence=evidence,
    )
