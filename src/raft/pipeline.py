"""Local pipeline: indexing, stage resumption, and retrieval over a case catalog."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Iterable, Sequence

from ._json import _id_key
from .cases import load_cases, restore_case
from .embedding import BM25Index, embed_cases
from .extraction import run_cases
from .graph import build_adjacency, build_case_graph, expand_neighbors
from .local_store import LocalStore, local_io
from .retrieval import LocalRetriever
from .storage import load_jsonl, save_json


class LocalPipeline:
    """Keep processing independent; persist only at this local orchestration layer.

    Provide normal run_cases/embed_cases options (without cases or output paths).
    The output model is extraction['output_type']; model clients/agents remain
    caller-owned. Existing case IDs are skipped unless index(rewrite=True).
    One writer per directory; methods on one instance are serialized. Query lists
    still use the retriever's normal async concurrency within each retrieve call.
    """

    def __init__(
        self,
        output_dir,
        *,
        extraction: dict[str, Any],
        embedding: dict[str, Any],
        graph: dict[str, Any] | None = None,
        bm25: bool = True,
        show_progress: bool = False,
    ):
        self.extraction = dict(extraction)
        self.embedding = dict(embedding)
        self.graph = None if graph is None else dict(graph)
        if type(show_progress) is not bool:
            raise ValueError("show_progress must be a bool")
        self.show_progress = show_progress
        for config in (self.extraction, self.embedding, self.graph):
            if config is not None:
                config.setdefault("show_progress", show_progress)
        # Validate stage options before any paid processing begins.
        inspect.signature(run_cases).bind(cases=[], **self.extraction)
        if self.extraction["reviewer_agent"] is None:
            raise ValueError("reviewer_agent is required")
        inspect.signature(embed_cases).bind(cases=[], **self.embedding)
        if self.graph is not None:
            inspect.signature(build_case_graph).bind(cases=[], **self.graph)
        self.output_type = self.extraction["output_type"]
        self.output_dir = LocalStore(output_dir).directory
        self.store = LocalStore(output_dir)
        self.bm25 = bm25
        self.embedding_space = self._space(self.embedding["backend"])
        self.graph_key = (
            None
            if graph is None
            else {
                **self._space(graph["backend"]),
                "top_k": graph.get("top_k", 10),
                "rrf_constant": graph.get("rrf_constant", 60),
            }
        )
        self._lock = asyncio.Lock()
        self._retriever = None
        self._retriever_key = None

    @staticmethod
    def _space(backend):
        return {
            "provider": backend.name,
            "model": backend.model,
            "dimensions": getattr(backend, "dimensions", None),
        }

    def _load(self):
        catalog = self.store.read()
        if self.store.path.exists():
            if catalog["embedding_space"] != self.embedding_space:
                raise ValueError(
                    "Use the same embedding model/dimensions for this directory, or a new output_dir"
                )
            return catalog
        catalog["embedding_space"] = self.embedding_space
        # Initialize the catalog from snapshot files without modifying them.
        path = self.output_dir / "extraction.json"
        if path.exists():
            cases = load_cases(path, output_type=self.output_type)
            rows_path = self.output_dir / "embeddings.jsonl"
            rows = load_jsonl(rows_path) if rows_path.exists() else []
            grouped = {}
            for row in rows:
                grouped.setdefault(_id_key(row["case_id"]), []).append(row)
            space = self._space(self.embedding["backend"])
            for case in cases:
                vectors = grouped.get(_id_key(case.id), [])
                compatible = bool(vectors) and all(
                    row["provider"] == space["provider"]
                    and row["model"] == space["model"]
                    and (space["dimensions"] is None or row["dimensions"] == space["dimensions"])
                    for row in vectors
                )
                catalog["cases"][_id_key(case.id)] = {
                    "case": case,
                    "status": "embedded" if compatible else "pending",
                    "embedding": {
                        "embeddings": vectors,
                        "requests": 0,
                        "attempts": 0,
                        "elapsed_seconds": 0.0,
                    }
                    if compatible
                    else None,
                }
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(snapshot, dict):
                for saved in snapshot.get("filtered_cases", []):
                    case = restore_case(saved, self.output_type)
                    catalog["cases"][_id_key(case.id)] = {
                        "case": case,
                        "status": "filtered",
                        "embedding": None,
                    }
        self.store.save(catalog)
        return catalog

    def _active(self, catalog):
        items = []
        for record in catalog["cases"].values():
            if record["status"] == "embedded":
                case = restore_case(record["case"], self.output_type)
                items.append({"id": case.id, "case": case, **record["embedding"]})
        return items

    async def index(self, cases: Sequence[dict[str, Any]] = (), *, rewrite: bool = False):
        """Upsert raw cases by ID, resume pending embedding, and refresh local indexes.

        Existing IDs are skipped even if their supplied content has changed.
        rewrite=True re-extracts/re-embeds only the supplied cases. Call index()
        without inputs to finish saved extraction or retry failed embedding.
        Stage checkpoints are atomic; an interrupted stage may repeat its work.
        Failed replacement extraction leaves the previous successful case intact.
        Once replacement extraction succeeds its old vectors are removed, even
        if replacement embedding fails or should_embed skips the new output.
        """
        async with self._lock:
            with self.store.writer():
                catalog = await local_io(self._load)
                pending, skipped_ids = [], []
                seen = set()
                id_field = self.extraction["id_field"]
                for raw in cases:
                    id = raw.get(id_field) if isinstance(raw, dict) else None
                    key = _id_key(id)
                    if type(id) not in (str, int) or key in seen:
                        raise ValueError("Input cases must have unique string/integer IDs per call")
                    seen.add(key)
                    saved = catalog["cases"].get(key)
                    if not rewrite and saved:
                        skipped_ids.append(id)
                    else:
                        pending.append(raw)
                extracted = {
                    "extracted_cases": [],
                    "failed_cases": [],
                    "summary": {"total": 0, "extracted": 0, "failed": 0},
                }
                if pending:
                    extracted = await run_cases(cases=pending, **self.extraction)
                    for case in extracted["extracted_cases"]:
                        key = _id_key(case.id)
                        catalog["cases"][key] = {
                            "case": case,
                            "status": "pending",
                            "embedding": None,
                        }
                    for case in extracted.get("filtered_cases", []):
                        catalog["cases"][_id_key(case.id)] = {
                            "case": case,
                            "status": "filtered",
                            "embedding": None,
                        }
                    catalog["extraction_failures"] = extracted["failed_cases"]
                    await local_io(self.store.save, catalog)

                to_embed = []
                for record in catalog["cases"].values():
                    if record["status"] in ("pending", "failed"):
                        to_embed.append(restore_case(record["case"], self.output_type))
                        record.update(status="pending", embedding=None)
                        record.pop("error", None)
                embedded = {
                    "embedded_cases": [],
                    "skipped_cases": [],
                    "failed_cases": [],
                    "embedding_usage": {},
                    "embedding_requests": 0,
                    "summary": {"total": 0, "embedded": 0, "skipped": 0, "failed": 0, "items": 0},
                }
                if to_embed:
                    await local_io(self.store.save, catalog)
                    embedded = await embed_cases(cases=to_embed, **self.embedding)
                    for item in embedded["embedded_cases"]:
                        record = catalog["cases"][_id_key(item["id"])]
                        record.update(
                            status="embedded",
                            embedding={k: v for k, v in item.items() if k not in ("case", "id")},
                        )
                    for case in embedded["skipped_cases"]:
                        catalog["cases"][_id_key(case.id)].update(status="skipped")
                    for failure in embedded["failed_cases"]:
                        catalog["cases"][_id_key(failure["id"])].update(
                            status="failed", error=failure
                        )
                    await local_io(LocalRetriever.from_embeddings, self._active(catalog))
                    await local_io(self.store.save, catalog)

                items = self._active(catalog)
                # Validate the combined corpus before publishing derived indexes.
                await local_io(LocalRetriever.from_embeddings, items)
                graph_result = await self._refresh_graph(catalog, items)
                await self._prepare_retriever(catalog, items)
                return {
                    "extraction": extracted,
                    "embedding": embedded,
                    "graph": graph_result,
                    "indexed_cases": embedded["embedded_cases"],
                    "skipped_ids": skipped_ids,
                    "summary": {
                        "received": len(cases),
                        "skipped_existing": len(skipped_ids),
                        "stored_extracted": len(catalog["cases"]),
                        "stored_embedded": len(items),
                        "stored_skipped": sum(
                            r["status"] == "skipped" for r in catalog["cases"].values()
                        ),
                        "stored_filtered": sum(
                            r["status"] == "filtered" for r in catalog["cases"].values()
                        ),
                        "stored_failed": sum(
                            r["status"] == "failed" for r in catalog["cases"].values()
                        ),
                    },
                }

    def _read_graph(self, catalog):
        path = self.store.cache_dir(catalog) / "graph.json"
        if not path.exists():
            return None
        saved = json.loads(path.read_text(encoding="utf-8"))
        if self.graph_key is not None and saved["config"] != self.graph_key:
            return None
        saved["result"]["nodes"] = [
            restore_case(case, self.output_type) for case in saved["result"]["nodes"]
        ]
        return saved["result"]

    async def _refresh_graph(self, catalog, items):
        saved = await local_io(self._read_graph, catalog)
        if self.graph is None or (saved is not None and not saved["failed_cases"]):
            return saved
        # Graph neighborhoods depend on the whole corpus, not just incoming cases.
        result = await build_case_graph(cases=[i["case"] for i in items], **self.graph)
        await local_io(
            save_json,
            self.store.cache_dir(catalog) / "graph.json",
            {"config": self.graph_key, "result": result},
        )
        return result

    async def _prepare_retriever(self, catalog, items):
        key = (catalog["revision"], self.bm25)
        if key == self._retriever_key:
            return self._retriever

        def prepare():
            rows = [row for item in items for row in item["embeddings"]]
            bm25 = None
            if self.bm25:
                path = self.store.cache_dir(catalog) / "bm25"
                if path.exists():
                    try:
                        bm25 = BM25Index.load(path)
                    except (ValueError, OSError, KeyError):
                        pass  # A derived cache is recoverable from the catalog.
                if bm25 is None:
                    bm25 = BM25Index.from_records(rows)
                    bm25.save(path)
            return LocalRetriever.from_embeddings(
                items,
                bm25_index=bm25,
            )

        self._retriever = await local_io(prepare)
        self._retriever_key = key
        return self._retriever

    def _read_graph_adjacency(self) -> dict[str | int, dict[str | int, float]]:
        unavailable = (
            "Graph access requires a completed graph for the current catalog; "
            "configure graph and call index() to build or refresh it"
        )
        if not self.store.path.exists():
            raise ValueError(unavailable)
        catalog = self._load()
        graph = self._read_graph(catalog)
        if graph is None or graph["failed_cases"]:
            raise ValueError(unavailable)
        adjacency = build_adjacency(graph["edges"])
        for case in graph["nodes"]:
            adjacency.setdefault(case.id, {})
        return adjacency

    async def graph_adjacency(self) -> dict[str | int, dict[str | int, float]]:
        """Return a detached adjacency mapping for user-owned graph traversal.

        Maps every saved graph case ID to {neighbor_id: weight}. Isolated nodes
        have empty dictionaries; integer and string IDs remain distinct. Edges
        are undirected and weights are SNN Jaccard scores, including zero weights.
        Both levels are fresh dictionaries; callers may mutate them freely.
        Missing, incomplete, stale, or incompatible graphs require index().
        Makes no model calls and never constructs graphs or retrieval indexes.
        The result is a snapshot; call again to read subsequent index updates.
        """
        async with self._lock:
            with self.store.writer():
                return await local_io(self._read_graph_adjacency)

    async def graph_expansion(
        self,
        case_ids: Sequence[str | int],
        *,
        per_case_top_k: int = 10,
        allowed_ids: Iterable[str | int] | None = None,
    ) -> list[dict[str, Any]]:
        """Read one-hop neighbors from the current completed graph; no model calls.

        Select up to per_case_top_k eligible neighbors for each supplied case,
        excluding all supplied IDs. Return one seed_id/neighbors group per distinct
        supplied ID, in input order. Each neighbor has id and weight; shared
        neighbors retain their separate connections in every selecting group.
        Groups can be empty, and there is no overall result cap.
        allowed_ids restricts candidates before each case's selection; retrieval
        filters do not carry over. Missing or incomplete graphs require index().
        This method never builds a graph or prepares retrieval indexes.
        """

        def expand():
            return expand_neighbors(
                case_ids,
                self._read_graph_adjacency(),
                per_case_top_k=per_case_top_k,
                allowed_ids=allowed_ids,
            )

        async with self._lock:
            with self.store.writer():
                return await local_io(expand)

    async def retrieve(self, queries: list[str], **options):
        """Query the current saved snapshot; never runs extraction or corpus embedding.

        Accepts LocalRetriever.retrieve options except backend, which is taken
        from the pipeline's embedding configuration. A missing BM25 cache can be
        rebuilt locally. Graph expansion is a separate graph_expansion() call.
        Returns results in query order plus embedding_usage and embedding_requests
        for this operation. batch_size controls query texts per embedding request.
        format_case receives a full ranked hit; context_budget caps the joined formatted
        text per query, returned as formatted_context alongside structured candidates.
        """
        options.setdefault("show_progress", self.show_progress)
        inspect.signature(LocalRetriever.retrieve).bind(
            None, queries, backend=self.embedding["backend"], **options
        )
        async with self._lock:
            with self.store.writer():
                catalog = await local_io(self._load)
                retriever = await self._prepare_retriever(catalog, self._active(catalog))
            return await retriever.retrieve(queries, backend=self.embedding["backend"], **options)
