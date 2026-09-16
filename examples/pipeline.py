"""Small live demo: extraction, state embeddings, and an optional case graph.

Run from the repo root: python examples/pipeline.py
Loads the repository .env. API keys never enter output files.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import voyageai
from agents import Agent, OpenAIResponsesModel, function_tool, set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI

from raft import LocalPipeline
from raft.defaults import (
    REVIEWER_INSTRUCTIONS,
    WORKER_INSTRUCTIONS,
    CaseExtraction,
    CaseReview,
    case_to_text,
    state_to_text,
)
from raft.embedding.voyage import VoyageEmbeddings
from raft.tools import edit_state, query_case_sql


@function_tool
def lookup_error(code: str) -> str:
    """Consult a tiny example error catalog. Real callers can provide their own tools."""
    return {
        "AUTH-401": "Authentication failed; investigate credentials or certificate validity."
    }.get(code, "Unknown code; rely on the case evidence.")


TOOL_INSTRUCTIONS = (
    "\nWhen AUTH-401 appears, call lookup_error once to interpret it, "
    "but ground your conclusions in the case evidence."
)


CASES = [
    {
        "ticket_number": "SHORT-1",
        "metadata": {"product": "portal"},
        "artifacts": [
            {"sequence": 1, "text": "Login fails with AUTH-401."},
            {
                "sequence": 2,
                "text": "SSO certificate expired. Renewed it; customer confirmed login works.",
            },
        ],
    },
    {
        "ticket_number": "LONG-1",
        "metadata": {"product": "upload"},
        "artifacts": [
            {"sequence": i, "text": text + " Diagnostic detail: " + "upload queue unchanged; " * 24}
            for i, text in enumerate(
                [
                    "Customer says uploads stall at 99%.",
                    "Support initially suspects network latency.",
                    "Network tests are normal; network hypothesis discarded.",
                    "A stale upload lock is found in the worker logs.",
                    "Support clears the stale lock and restarts the worker.",
                    "Customer confirms new uploads complete successfully.",
                ]
            )
        ],
    },
    {
        "ticket_number": "RFI-1",
        "metadata": {"product": "general"},
        "artifacts": [
            {"sequence": 1, "text": "Please send your product brochure and pricing sheet."}
        ],
    },
]


async def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    load_dotenv(repo / ".env")
    set_tracing_disabled(True)
    async with AsyncOpenAI(max_retries=0) as client:
        model = OpenAIResponsesModel(model="gpt-5.2", openai_client=client)
        # Agent definitions and any business-specific tools belong to the application.
        worker = Agent(
            name="Case worker",
            model=model,
            tools=[query_case_sql, edit_state, lookup_error],
            instructions=WORKER_INSTRUCTIONS + TOOL_INSTRUCTIONS,
        )
        pipeline = LocalPipeline(
            extraction={
                "output_type": CaseExtraction,
                "worker_agent": worker,
                "reviewer_agent": Agent(
                    name="Case reviewer",
                    model=worker.model,
                    tools=[query_case_sql, edit_state],
                    instructions=REVIEWER_INSTRUCTIONS,
                    output_type=CaseReview,
                ),
                "should_keep": lambda case: case.review.extractable,
                "id_field": "ticket_number",
                "artifacts_field": "artifacts",
                "metadata_field": "metadata",
                "artifact_sort_field": "sequence",
                "max_batch_chars": 1800,
                "max_query_chars": 1800,
                "concurrency": 2,
                "agent_concurrency": 1,
                "timeout": 180,
                "retries": 1,
                "rpm": 60,
            },
            embedding={
                "backend": VoyageEmbeddings(client=voyageai.AsyncClient(max_retries=0), model="voyage-4"),
                "state_to_text": state_to_text,
                "batch_size": 64,
                "concurrency": 2,
                "rpm": 60,
            },
            # Omit this dictionary (or set graph=None) to skip case graph construction.
            graph={
                "backend": VoyageEmbeddings(client=voyageai.AsyncClient(max_retries=0), model="voyage-4"),
                "case_to_text": case_to_text,
                "batch_size": 64,
                "top_k": 10,
                "concurrency": 2,
                "rpm": 60,
            },
            output_dir=repo / "outputs" / "demo",
        )
        result = await pipeline.index(CASES)
        retrieval = await pipeline.retrieve(
            ["SSO login fails", "Uploads stall"], top_k=2, batch_size=64,
        )
        print("Query embedding usage:", retrieval["embedding_usage"])
        for match in retrieval["results"]:
            print(match["query"], [hit["id"] for hit in match["candidates"]], match["error"])
            if result["graph"] is not None and not result["graph"]["failed_cases"]:
                seed_ids = [hit["id"] for hit in match["candidates"][:1]]
                neighbor_groups = await pipeline.graph_expansion(seed_ids, per_case_top_k=2)
                for group in neighbor_groups:
                    print("Graph neighbors for", group["seed_id"], group["neighbors"])
                # Groups stay separate; the caller chooses how to combine and budget cases.
    print(
        json.dumps(
            {
                "extraction": result["extraction"]["summary"],
                "embedding": result["embedding"]["summary"],
                "stored": result["summary"],
                "graph": result["graph"]["summary"],
                "cases": [
                    {
                        "id": case.id,
                        "passes": case.execution["passes"],
                    }
                    for case in result["extraction"]["extracted_cases"]
                ],
                "failures": result["extraction"]["failed_cases"]
                + result["embedding"]["failed_cases"]
                + result["graph"]["failed_cases"],
            },
            indent=2,
            default=lambda value: value.model_dump(mode="json"),
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
