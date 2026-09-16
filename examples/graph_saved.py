"""Build a case graph from saved extraction, without extracting again.

Makes embedding API requests. Loads .env, never writes its contents.
"""

import asyncio
import json
from pathlib import Path

import voyageai
from dotenv import load_dotenv

from raft import build_case_graph, load_cases
from raft.defaults import CaseExtraction, case_to_text
from raft.embedding.voyage import VoyageEmbeddings
from raft.storage import save_graph


async def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    load_dotenv(repo / ".env")
    cases = load_cases(repo / "outputs/demo/extraction.json", output_type=CaseExtraction)
    result = await build_case_graph(
        cases=cases,
        backend=VoyageEmbeddings(client=voyageai.AsyncClient(max_retries=0), model="voyage-4"),
        case_to_text=case_to_text,
        batch_size=64,
        top_k=10,
        # Optionally filter the original case records:
        # neighbor_filter=lambda source, candidate: (
        #     source.metadata.get("product") == candidate.metadata.get("product")
        # ),
    )
    await asyncio.to_thread(save_graph, result, repo / "outputs/demo/case_graph")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
