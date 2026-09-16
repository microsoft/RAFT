"""Local BM25 indexing over the same state records used for dense embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import bm25s

from raft._json import _id_key
from raft.storage import load_jsonl, save_json, save_jsonl


def _tokenize(texts: list[str]) -> list[list[str]]:
    # Keep single-character terms and stopwords (including negations). No stemming.
    return bm25s.tokenize(
        texts,
        lower=True,
        token_pattern=r"(?u)\b\w+\b",
        stopwords=[],
        return_ids=False,
        show_progress=False,
    )


class BM25Index:
    """A batch-built corpus with stable state IDs for later rank fusion.

    Build from embedding records; vectors are not copied into this index.
    Rebuild when documents change. save/load use a directory, not a database.
    """

    def __init__(self, documents: list[dict[str, Any]], index: bm25s.BM25 | None):
        self.documents = documents
        self._index = index

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]]) -> BM25Index:
        documents = []
        seen = set()
        for record in records:
            identifier, text = record["id"], record["text"]
            if identifier is None or identifier == "" or _id_key(identifier) in seen:
                raise ValueError("BM25 records must have nonempty, unique IDs")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("BM25 records must have nonempty text")
            seen.add(_id_key(identifier))
            documents.append(
                {
                    "id": identifier,
                    "text": text,
                    **{key: record[key] for key in ("case_id", "item_index") if key in record},
                }
            )
        tokens = _tokenize([doc["text"] for doc in documents])
        index = None
        if any(tokens):
            index = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
            index.index(tokens, show_progress=False)
        return cls(documents, index)

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Return up to k matching records with descending positive BM25 scores."""
        if k < 1:
            raise ValueError("k must be at least 1")
        tokens = _tokenize([query])
        if self._index is None or not tokens[0]:
            return []
        positions, scores = self._index.retrieve(
            tokens,
            k=min(k, len(self.documents)),
            show_progress=False,
        )
        return [
            {**self.documents[int(position)], "score": float(score)}
            for position, score in zip(positions[0], scores[0], strict=True)
            if score > 0
        ]

    def scores(self, query: str) -> list[float]:
        """Score every document in corpus order, allowing filtering before ranking."""
        tokens = _tokenize([query])[0]
        if self._index is None or not tokens:
            return [0.0] * len(self.documents)
        return self._index.get_scores(tokens).tolist()

    def save(self, path: str | Path) -> None:
        """Save a snapshot. Do not read/write this directory concurrently."""
        path = Path(path)
        # A failed write must not leave a snapshot that appears complete.
        save_json(path / "raft.json", {"version": 1, "complete": False})
        save_jsonl(path / "documents.jsonl", self.documents)
        if self._index is not None:
            self._index.save(path, show_progress=False)
        save_json(
            path / "raft.json",
            {
                "version": 1,
                "complete": True,
                "indexed": self._index is not None,
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        path = Path(path)
        manifest = json.loads((path / "raft.json").read_text(encoding="utf-8"))
        if manifest.get("version") != 1 or not manifest.get("complete"):
            raise ValueError("Unsupported or incomplete BM25 snapshot")
        documents = load_jsonl(path / "documents.jsonl")
        index = bm25s.BM25.load(path, show_progress=False) if manifest["indexed"] else None
        return cls(documents, index)
