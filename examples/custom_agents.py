"""Wire custom schema, prompts, text views, models, and tools into one pipeline.

Pass caller-owned SDK Model objects and an embedding backend (see the Jira
notebook or model_routing.py for client setup). Nothing runs on import.
Use a fresh output directory after changing schemas, prompts, or text views.

After pipeline.index(cases), inspect extraction/embedding failures as in the
notebook. To use the matching retrieval formatter:

    from examples.custom_text import format_case

    retrieved = await pipeline.retrieve(
        ["Login fails with AUTH_CERT_EXPIRED."],
        format_case=format_case,
        context_budget={"unit": "chars", "limit": 12_000},
    )
    result = retrieved["results"][0]
    if result["error"]:
        raise RuntimeError(result["error"])
    print(result["formatted_context"])
"""

from collections.abc import Callable
from pathlib import Path

from agents import Agent, ModelSettings, RunConfig, function_tool
from agents.models.interface import Model

from examples.custom_extraction import REVIEWER_PROMPT, WORKER_PROMPT, SupportCase
from examples.custom_text import case_to_text, state_to_text
from raft import LocalPipeline
from raft.defaults import CaseReview
from raft.embedding.backend import EmbeddingBackend
from raft.extraction.context import CaseContext
from raft.tools import edit_state, query_case_sql, write_handoff_note

ERROR_CATALOG = {
    "AUTH_CERT_EXPIRED": (
        "Example reference: check the configured authentication certificate's "
        "validity period and the system clock before considering certificate rotation."
    ),
}


@function_tool
def lookup_error(error_code: str) -> dict[str, str]:
    """Look up an exact code in a tiny, synthetic error catalog.

    Replace the local catalog with your authorized internal search service.
    A match provides reference guidance, not evidence that an action occurred.
    Unknown codes return status=not_found, not a guessed explanation.
    """
    if error_code not in ERROR_CATALOG:
        return {"status": "not_found", "error_code": error_code}
    return {"status": "found", "error_code": error_code, "guidance": ERROR_CATALOG[error_code]}


def build_pipeline(
    output_dir: str | Path,
    *,
    worker_model: Model,
    reviewer_model: Model,
    embedding_backend: EmbeddingBackend,
    with_graph: bool = False,
    run_config: RunConfig | Callable[[Agent[CaseContext], CaseContext], RunConfig] | None = None,
) -> LocalPipeline:
    """Configure separate agents while preserving RAFT's required tool contracts."""
    worker = Agent[CaseContext](
        name="Support worker",
        model=worker_model,
        model_settings=ModelSettings(store=False),
        instructions=WORKER_PROMPT,
        tools=[query_case_sql, edit_state, write_handoff_note, lookup_error],
    )
    reviewer = Agent[CaseContext](
        name="Support reviewer",
        model=reviewer_model,
        model_settings=ModelSettings(store=False),
        instructions=REVIEWER_PROMPT,
        tools=[query_case_sql, edit_state, lookup_error],
        output_type=CaseReview,
    )
    return LocalPipeline(
        output_dir,
        extraction={
            "worker_agent": worker,
            "reviewer_agent": reviewer,
            "output_type": SupportCase,
            "id_field": "id",
            "metadata_field": "metadata",
            "artifacts_field": "artifacts",
            "should_keep": lambda case: case.review.extractable,
            "batch_budget": {"unit": "chars", "limit": 64_000},
            "query_budget": {"unit": "chars", "limit": 12_000},
            "run_config": run_config,
        },
        embedding={"backend": embedding_backend, "state_to_text": state_to_text},
        graph={
            "backend": embedding_backend,
            "case_to_text": case_to_text,
            "top_k": 5,
        } if with_graph else None,
    )
