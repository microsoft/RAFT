"""Storage-free indexing demo. Makes API calls; writes no result files.

Run from the repo root: python examples/index.py
Reuses the demo inputs/tools and loads .env without printing credentials.
"""

import asyncio
from pathlib import Path

import voyageai
from agents import Agent, OpenAIResponsesModel, set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pipeline import CASES, TOOL_INSTRUCTIONS, lookup_error

from raft import embed_cases, run_cases
from raft.defaults import (
    REVIEWER_INSTRUCTIONS,
    WORKER_INSTRUCTIONS,
    CaseExtraction,
    CaseReview,
    state_to_text,
)
from raft.embedding.voyage import VoyageEmbeddings
from raft.tools import edit_state, query_case_sql


async def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    set_tracing_disabled(True)
    async with AsyncOpenAI(max_retries=0) as client:
        worker = Agent(
            name="Case worker",
            model=OpenAIResponsesModel(model="gpt-5.2", openai_client=client),
            tools=[query_case_sql, edit_state, lookup_error],
            instructions=WORKER_INSTRUCTIONS + TOOL_INSTRUCTIONS,
        )
        extraction = await run_cases(
            cases=CASES,
            **{
                "worker_agent": worker,
                "reviewer_agent": Agent(
                    name="Case reviewer",
                    model=worker.model,
                    tools=[query_case_sql, edit_state],
                    instructions=REVIEWER_INSTRUCTIONS,
                    output_type=CaseReview,
                ),
                "should_keep": lambda case: case.review.extractable,
                "output_type": CaseExtraction,
                "id_field": "ticket_number",
                "metadata_field": "metadata",
                "artifacts_field": "artifacts",
                "artifact_sort_field": "sequence",
                "max_batch_chars": 1800,
                "concurrency": 2,
                "agent_concurrency": 1,
                "rpm": 60,
            },
        )
        embedding = await embed_cases(
            cases=extraction["extracted_cases"],
            **{
                "backend": VoyageEmbeddings(client=voyageai.AsyncClient(max_retries=0), model="voyage-4"),
                "state_to_text": state_to_text,
                "concurrency": 2,
                "rpm": 60,
            },
        )
    print("Extraction:", extraction["summary"])
    print("Embedding:", embedding["summary"])
    for item in embedding["embedded_cases"]:
        case = item["case"]  # Live ExtractedCase, not a serialized dictionary.
        print(case.id, case.metadata, case.execution["usage"], len(item["embeddings"]))


if __name__ == "__main__":
    asyncio.run(main())
