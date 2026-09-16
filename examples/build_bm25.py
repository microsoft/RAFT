"""Build/query BM25 from saved embeddings, without any API requests."""

from pathlib import Path

from raft.embedding import BM25Index
from raft.storage import load_jsonl


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "outputs/demo"
    records = load_jsonl(output_dir / "embeddings.jsonl")
    index = BM25Index.from_records(records)
    index.save(output_dir / "bm25")
    loaded = BM25Index.load(output_dir / "bm25")
    print(f"Indexed {len(loaded.documents)} states")
    for hit in loaded.search("AUTH-401", k=3):
        print(hit["case_id"], hit["item_index"], hit["score"], hit["text"])


if __name__ == "__main__":
    main()
