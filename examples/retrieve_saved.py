"""Query a saved local snapshot. Makes query-embedding API calls only.

Requires explicitly saved extraction.json and embeddings.jsonl in outputs/demo.
Optional saved graph edges are loaded and expanded separately below.
"""

import asyncio
from pathlib import Path

import voyageai
from dotenv import load_dotenv

from raft import LocalRetriever
from raft.defaults import CaseExtraction
from raft.embedding.voyage import VoyageEmbeddings
from raft.graph import build_adjacency, expand_neighbors
from raft.storage import load_jsonl


async def main():
    repo = Path(__file__).resolve().parents[1]
    load_dotenv(repo / ".env")
    retriever = await LocalRetriever.load(repo / "outputs/demo", output_type=CaseExtraction)
    edges_path = repo / "outputs/demo/case_graph/edges.jsonl"
    adjacency = build_adjacency(load_jsonl(edges_path)) if edges_path.exists() else None
    report = await retriever.retrieve(
        ["Login fails with AUTH-401", "Uploads stall at 99%"],
        backend=VoyageEmbeddings(client=voyageai.AsyncClient(max_retries=0), model="voyage-4"),
        top_k=3,
        max_chars=20_000,
        batch_size=64,
    )
    print("Embedding usage:", report["embedding_usage"], "Requests:", report["embedding_requests"])
    for result in report["results"]:
        print(result["query"], result["error"])
        for hit in result["candidates"]:
            case = hit["case"]
            print(case.id, case.metadata, hit["item_index"])
            print(case.output)
        if adjacency is not None:
            seed_ids = [hit["id"] for hit in result["candidates"][:1]]
            neighbor_groups = expand_neighbors(seed_ids, adjacency, per_case_top_k=2)
            for group in neighbor_groups:
                print("Graph neighbors for", group["seed_id"], group["neighbors"])
            # Caller chooses how to load, combine, filter, and budget neighbor cases.


if __name__ == "__main__":
    asyncio.run(main())
