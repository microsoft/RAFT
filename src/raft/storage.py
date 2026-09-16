"""Portable artifacts for independently running subsequent indexing stages."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write(path: str | Path, chunks: Iterable[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Replace only after serialization and writing succeed; no partial output file.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, delete=False
        ) as stream:
            temporary = stream.name
            for chunk in chunks:
                stream.write(chunk)
        os.replace(temporary, target)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def save_json(path: str | Path, value: Any) -> None:
    _write(
        path,
        [
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)
            + "\n"
        ],
    )


def save_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    _write(
        path,
        (
            json.dumps(row, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n"
            for row in rows
        ),
    )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_embeddings(
    result: dict[str, Any],
    *,
    output_path: str | Path | None = None,
    bm25_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Explicitly save a batch snapshot; never merge/upsert an existing corpus.

    Returns BM25 path/document information when requested, otherwise None.
    In async applications call through asyncio.to_thread.
    """
    from .embedding.bm25 import BM25Index

    records = [row for case in result["embedded_cases"] for row in case["embeddings"]]
    if output_path is not None:
        save_jsonl(output_path, records)
    if bm25_path is not None:
        index = BM25Index.from_records(records)
        index.save(bm25_path)
        return {"path": str(bm25_path), "documents": len(index.documents)}
    return None


def save_graph(result: dict[str, Any], output_dir: str | Path) -> None:
    """Save a graph snapshot; save canonical cases separately to retain outputs.

    Files are individually replaced, not a directory transaction. Do not read
    or write the same snapshot concurrently. Use asyncio.to_thread when async.
    """
    from .embedding.bm25 import BM25Index

    directory = Path(output_dir)
    save_jsonl(directory / "nodes.jsonl", ({"id": case.id} for case in result["nodes"]))
    for name in ("embeddings", "neighbors", "edges"):
        save_jsonl(directory / f"{name}.jsonl", result[name])
    BM25Index.from_records(result["embeddings"]).save(directory / "bm25")
    save_json(
        directory / "report.json",
        {
            key: value
            for key, value in result.items()
            if key not in {"nodes", "embeddings", "neighbors", "edges"}
        },
    )
