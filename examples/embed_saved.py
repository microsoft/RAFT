"""Run the embedding stage independently against a saved extraction result."""

import asyncio
from pathlib import Path

import voyageai
from dotenv import load_dotenv

from raft import embed_cases, load_cases
from raft.defaults import CaseExtraction, state_to_text
from raft.embedding.voyage import VoyageEmbeddings
from raft.storage import save_embeddings


async def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    load_dotenv(repo / ".env")
    cases = load_cases(repo / "outputs/demo/extraction.json", output_type=CaseExtraction)
    result = await embed_cases(
        cases=cases,
        backend=VoyageEmbeddings(client=voyageai.AsyncClient(max_retries=0), model="voyage-4"),
        state_to_text=state_to_text,
    )
    await asyncio.to_thread(
        save_embeddings, result, output_path=repo / "outputs/demo/reembedded.jsonl"
    )
    print(result["summary"])


if __name__ == "__main__":
    asyncio.run(main())
