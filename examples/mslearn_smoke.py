"""MS Learn migration experiment; --full uses the original notebook's complete split.

Makes billable calls unless --dry-run. No graph construction or LLM-based evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
from collections import Counter
from contextlib import AsyncExitStack
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agents import Agent, ModelSettings, OpenAIResponsesModel, RunConfig
from dotenv import load_dotenv
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from raft import LocalPipeline
from raft.defaults import (
    REVIEWER_INSTRUCTIONS,
    WORKER_INSTRUCTIONS,
    CaseExtraction,
    CaseReview,
    state_to_text,
)
from raft.embedding.openai import OpenAIEmbeddings
from raft.storage import save_json
from raft.tools import edit_state, query_case_sql

REPO = Path(__file__).resolve().parents[1]
PROGRESSES = (0, 30, 60)


async def azure_retry_headers(response):
    """Let native request retries honor Azure's token-window reset before replaying a pass."""
    if response.status_code != 429 or response.headers.get("retry-after"):
        return
    reset = response.headers.get("x-ratelimit-reset-tokens")
    if reset is not None and reset.isdecimal():
        response.headers["retry-after"] = str(max(1, min(60, int(reset))))
        logging.getLogger(__name__).warning(
            "Azure token quota reached; retrying this HTTP request after %s seconds.",
            response.headers["retry-after"],
        )


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def load_dataset(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("MS Learn data must be a nonempty JSON list")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Every case must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError("Every case must have a unique nonempty string case_id")
        seen.add(case_id)
        reference = case.get("synthetic_information")
        if not isinstance(reference, dict) or not isinstance(reference.get("shared_id"), str):
            raise ValueError(f"{case_id}: missing synthetic_information.shared_id")
        if not reference["shared_id"].strip() or not isinstance(case.get("metadata"), dict):
            raise ValueError(f"{case_id}: invalid shared_id or metadata")
        messages = case.get("conversations")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{case_id}: conversations must be a nonempty list")
        positions = []
        for message in messages:
            if (
                not isinstance(message, dict)
                or type(message.get("idx")) is not int
                or not isinstance(message.get("body", message.get("text")), str)
            ):
                raise ValueError(f"{case_id}: each message needs integer idx and string body/text")
            positions.append(message["idx"])
        if len(set(positions)) != len(positions):
            raise ValueError(f"{case_id}: message idx values must be unique")
    return cases


def select_cases(
    cases: list[dict], *, seed: int, test_count: int, index_count: int
) -> tuple[list[dict], list[dict]]:
    """Seeded, paired holdouts with reserved positives and a bounded distractor corpus."""
    if test_count < 1 or index_count < test_count:
        raise ValueError("Need at least one holdout and index_count >= test_count")
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    counts = Counter(case["synthetic_information"]["shared_id"] for case in cases)
    tests, remaining, shared_ids = [], [], set()
    for case in shuffled:
        shared_id = case["synthetic_information"]["shared_id"]
        if len(tests) < test_count and counts[shared_id] > 1 and shared_id not in shared_ids:
            tests.append(case)
            shared_ids.add(shared_id)
        else:
            remaining.append(case)
    if len(tests) != test_count or len(remaining) < index_count:
        raise ValueError("Not enough paired cases for the requested split")
    indexed, reserved = [], set()
    for case in remaining:
        shared_id = case["synthetic_information"]["shared_id"]
        if shared_id in shared_ids and shared_id not in reserved:
            indexed.append(case)
            reserved.add(shared_id)
    used = {case["case_id"] for case in indexed}
    indexed.extend(case for case in remaining if case["case_id"] not in used)
    return tests, indexed[:index_count]


def select_full_cases(
    cases: list[dict], *, seed: int, test_count: int = 1000
) -> tuple[list[dict], list[dict]]:
    """Original notebook split, including singleton holdouts and every remaining case."""
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    tests, indexed, seen_shared_ids = [], [], set()
    for case in shuffled:
        shared_id = case["synthetic_information"]["shared_id"]
        if len(tests) < test_count and shared_id not in seen_shared_ids:
            tests.append(case)
            seen_shared_ids.add(shared_id)
        else:
            indexed.append(case)
    if not tests or not indexed:
        raise ValueError("The full split needs nonempty test and index sets")
    return tests, indexed


def agent_case(case: dict) -> dict:
    """Do not leak synthetic answers or shared IDs into extraction or retrieval."""
    return {
        "case_id": case["case_id"],
        "metadata": deepcopy(case["metadata"]),
        "conversations": deepcopy(sorted(case["conversations"], key=lambda message: message["idx"])),
    }


def build_query(conversations: list[dict], progress: int) -> str:
    if type(progress) is not int or not 0 <= progress <= 100 or not conversations:
        raise ValueError("A nonempty conversation and integer progress in [0, 100] are required")
    ordered = sorted(conversations, key=lambda message: message["idx"])
    stop = progress * (len(ordered) - 1) // 100 + 1
    parts = []
    for message in ordered[:stop]:
        block = f"From: {message.get('from', 'unknown')}"
        if message.get("subject"):
            block += f"\nSubject: {message['subject']}"
        block += f"\n\n{message.get('body', message.get('text', ''))}"
        parts.append(block)
    return "\n\n---\n\n".join(parts)


def check_extraction(result: dict, expected_count: int, *, full: bool = False) -> dict:
    if result["failed_cases"]:
        failures = [(case["id"], case["error_type"]) for case in result["failed_cases"]]
        raise RuntimeError(f"Extraction failed; inspect extraction.json: {failures}")
    cases = result["extracted_cases"]
    filtered = result["filtered_cases"]
    if len(cases) + len(filtered) != expected_count or (filtered and not full):
        raise RuntimeError("A selected technical case was filtered or not extracted")
    for case in filtered:
        CaseExtraction.model_validate(case.output.model_dump())
        review = CaseReview.model_validate(case.review.model_dump())
        if review.extractable:
            raise RuntimeError(f"{case.id}: eligible case was incorrectly filtered")
    pass_counts = {}
    for case in cases:
        state = CaseExtraction.model_validate(case.output.model_dump())
        review = CaseReview.model_validate(case.review.model_dump())
        if not review.extractable or not state.timeline or (not state.entities and not full):
            raise RuntimeError(f"{case.id}: missing eligible technical extraction")
        revisions = case.execution["revisions"]
        worker_revisions = [revision for revision in revisions if revision["stage"] == "worker"]
        passes = case.execution["passes"]
        if len(worker_revisions) != passes:
            raise RuntimeError(f"{case.id}: committed pass history is incomplete")
        if not any(call.get("stage") == "review" for call in case.execution["tool_calls"]):
            raise RuntimeError(f"{case.id}: final reviewer was not executed")
        if not case.execution["usage"]:
            raise RuntimeError(f"{case.id}: live model usage is missing")
        if state_to_text(state) != [entry.narrative for entry in state.timeline]:
            raise RuntimeError(f"{case.id}: default embedding text does not match narratives")
        pass_counts[case.id] = passes
    if not full and not any(passes > 1 for passes in pass_counts.values()):
        raise RuntimeError("No multi-pass case was exercised; lower --max-batch-chars")
    return pass_counts


def format_case_context(hit: dict) -> str:
    """Map the notebook's terminal-state context to the migrated default fields."""
    case = hit["case"]
    state = case.output
    return json.dumps(
        {
            "case_metadata": case.metadata,
            "root_cause": state.root_cause,
            "resolution_steps": state.resolution_steps,
            "narrative": state.timeline[-1].narrative if state.timeline else "",
        },
        ensure_ascii=False,
    )


def budget_candidates(hits: list[dict], max_context_tokens: int | None) -> list[dict]:
    """Match the legacy len(text)//4 budget, including the first crossing case."""
    if max_context_tokens is None:
        return hits
    tokens = 0
    for position, hit in enumerate(hits, 1):
        tokens += len(format_case_context(hit)) // 4
        if tokens > max_context_tokens:
            return hits[:position]
    return hits


def score_retrieval(
    result: dict, tests: list[dict], indexed: list[dict], *, max_context_tokens: int | None = None
) -> tuple[list, dict]:
    if len(result["results"]) != len(tests):
        raise RuntimeError("Retrieval result count does not match the holdout count")
    shared_by_id = {case["case_id"]: case["synthetic_information"]["shared_id"] for case in indexed}
    rows = []
    for query_result, case in zip(result["results"], tests, strict=True):
        if query_result["error"] or not query_result["candidates"]:
            raise RuntimeError(f"{case['case_id']}: failed or empty retrieval result")
        before_budget = query_result["candidates"]
        hits = budget_candidates(before_budget, max_context_tokens)
        ids = [hit["id"] for hit in hits]
        if case["case_id"] in ids or any(case_id not in shared_by_id for case_id in ids):
            raise RuntimeError("Retrieval returned a holdout or an unknown corpus case")
        shared_id = case["synthetic_information"]["shared_id"]
        # The original notebook strips the synthetic variant suffix before case_hit.
        parent_ids = [case_id.split("_")[0] for case_id in ids]
        ranks = [i for i, parent_id in enumerate(parent_ids, 1) if parent_id == shared_id]
        rows.append({
            "case_id": case["case_id"],
            "query": query_result["query"],
            "reference": case["synthetic_information"],
            "retrieved_case_ids": ids,
            "retrieved_shared_ids": parent_ids,
            "candidate_ids_before_context_budget": [hit["id"] for hit in before_budget],
            "case_hit_before_context_budget": any(
                hit["id"].split("_")[0] == shared_id for hit in before_budget
            ),
            "first_relevant_rank": ranks[0] if ranks else None,
            "case_hit": bool(ranks),
            "matches": [
                {
                    "id": hit["id"],
                    "item_index": hit["item_index"],
                    "score": hit["score"],
                    "output": hit["case"].output,
                }
                for hit in hits
            ],
        })
    return rows, {
        "queries": len(rows),
        "case_hits": sum(row["case_hit"] for row in rows),
        "case_hit_rate": sum(row["case_hit"] for row in rows) / len(rows),
        "case_hits_before_context_budget": sum(
            row["case_hit_before_context_budget"] for row in rows
        ),
    }


async def index_with_retries(pipeline, inputs, directory):
    """Persist each attempt and retry only missing extraction or incomplete embeddings."""
    from raft.cases import restore_case

    output_path = directory / "extraction.json"
    extracted, filtered = {}, {}
    catalog_path = directory / "pipeline" / "catalog.json"
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        for record in catalog["cases"].values():
            saved = record["case"]
            case = restore_case(saved, CaseExtraction)
            # Saved review models are restored separately from output_type.
            case.review = CaseReview.model_validate(saved["review"])
            records = filtered if record["status"] == "filtered" else extracted
            records[case.id] = case
    for attempt in range(4):
        result = await pipeline.index(inputs)
        current = result["extraction"]
        for case in current["extracted_cases"]:
            extracted[case.id] = case
        for case in current.get("filtered_cases", []):
            filtered[case.id] = case
        extraction = {
            "extracted_cases": list(extracted.values()),
            "filtered_cases": list(filtered.values()),
            "failed_cases": current["failed_cases"],
            "summary": {
                "total": len(inputs),
                "extracted": len(extracted),
                "filtered": len(filtered),
                "failed": len(current["failed_cases"]),
            },
        }
        save_json(output_path, extraction)
        save_json(directory / "index_summary.json", {
            "extraction": extraction["summary"],
            "embedding": result["embedding"]["summary"],
            "embedding_failures": result["embedding"]["failed_cases"],
            "stored": result["summary"],
            "index_attempt": attempt + 1,
        })
        if not current["failed_cases"] and not result["embedding"]["failed_cases"]:
            result["extraction"] = extraction
            return result
        print(
            f"Index attempt {attempt + 1}: {len(current['failed_cases'])} extraction and "
            f"{len(result['embedding']['failed_cases'])} embedding failures; "
            "successful cases are checkpointed.",
            flush=True,
        )
        if attempt < 3:
            await asyncio.sleep(60)
    raise RuntimeError("Index retries exhausted; inspect reports and rerun with --resume")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO / "datasets/mslearn/all_cases.json")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--full", action="store_true", help="Use the entire original notebook split")
    parser.add_argument("--test-cases", type=positive_int)
    parser.add_argument("--index-cases", type=positive_int)
    parser.add_argument("--top-k", type=positive_int)
    parser.add_argument("--max-batch-chars", type=positive_int)
    parser.add_argument("--concurrency", type=positive_int, default=80)
    parser.add_argument("--rpm", type=positive_int, default=100)
    parser.add_argument("--max-output-tokens", type=positive_int)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--embedding-model", default="text-embedding-3-large")
    parser.add_argument("--azure-endpoint")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume the same saved run directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.full and args.index_cases is not None:
        parser.error("--full indexes every remaining case; do not supply --index-cases")
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")
    args.test_cases = args.test_cases or (1000 if args.full else 3)
    args.index_cases = args.index_cases or (None if args.full else 12)
    args.top_k = args.top_k or (10 if args.full else 5)
    args.max_output_tokens = args.max_output_tokens or (8000 if args.full else 4000)
    return args


async def main(argv=None):
    args = parse_args(argv)
    cases = load_dataset(args.dataset)
    if args.full:
        tests, indexed = select_full_cases(cases, seed=args.seed, test_count=args.test_cases)
    else:
        tests, indexed = select_cases(
            cases, seed=args.seed, test_count=args.test_cases, index_count=args.index_cases
        )
    inputs = [agent_case(case) for case in indexed]
    largest = max(
        len(json.dumps(message, ensure_ascii=False))
        for case in inputs for message in case["conversations"]
    )
    batch_chars = args.max_batch_chars or max(400_000 if args.full else 4000, largest)
    if batch_chars < largest:
        raise ValueError(f"--max-batch-chars must fit the largest whole artifact ({largest} chars)")
    manifest = {
        "status": "dry_run" if args.dry_run else "running",
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "dataset_cases": len(cases),
        "test_cases": len(tests),
        "indexed_cases": len(indexed),
        "requested_test_cases": args.test_cases,
        "holdouts_without_indexed_counterpart": len(
            {case["synthetic_information"]["shared_id"] for case in tests}
            - {case["synthetic_information"]["shared_id"] for case in indexed}
        ),
        "seed": args.seed,
        "progresses": list(PROGRESSES),
        "test_case_ids": [case["case_id"] for case in tests],
        "indexed_case_ids": [case["case_id"] for case in indexed],
        "model": args.model,
        "reasoning_effort": "medium",
        "embedding_model": args.embedding_model,
        "provider": "azure" if args.azure_endpoint else "openai",
        "top_k": args.top_k,
        "max_batch_chars": batch_chars,
        "concurrency": args.concurrency,
        "rpm": args.rpm,
        "max_output_tokens": args.max_output_tokens,
        "max_context_tokens": 4000 if args.full else None,
        "scope": "full notebook split" if args.full else "bounded paired migration smoke",
        "retrieval": "new RAFT narrative BM25 + vector RRF, no graph, no LLM judge",
        "context_format": "case_metadata, root_cause, resolution_steps, terminal narrative",
        "http_retries": 6 if args.azure_endpoint else 0,
    }
    print(json.dumps(
        {key: value for key, value in manifest.items() if not key.endswith("_case_ids")}, indent=2
    ), flush=True)
    if args.dry_run:
        return
    load_dotenv(REPO / ".env")
    if not args.azure_endpoint and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in .env or use --azure-endpoint after az login")
    directory = args.output_dir or (
        REPO / "outputs" / "mslearn"
        / f"seed_{args.seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    )
    if args.resume:
        previous_manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for key, value in manifest.items():
            if key != "status" and previous_manifest.get(key) != value:
                raise ValueError(f"Cannot resume with a different {key}; use a fresh output directory")
        previous_defaults = json.loads((directory / "defaults.json").read_text(encoding="utf-8"))
        if (
            previous_defaults["worker_instructions"] != WORKER_INSTRUCTIONS
            or previous_defaults["reviewer_instructions"] != REVIEWER_INSTRUCTIONS
            or previous_defaults["output_schema"] != CaseExtraction.model_json_schema()
            or previous_defaults["review_schema"] != CaseReview.model_json_schema()
        ):
            raise ValueError("Cannot resume with changed prompts or schemas")
    else:
        directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / "manifest.json", manifest)
    save_json(directory / "defaults.json", {
        "worker_instructions": WORKER_INSTRUCTIONS,
        "reviewer_instructions": REVIEWER_INSTRUCTIONS,
        "output_schema": CaseExtraction.model_json_schema(),
        "review_schema": CaseReview.model_json_schema(),
    })
    async with AsyncExitStack() as stack:
        client_options = {"max_retries": 0, "timeout": 180.0}
        if args.azure_endpoint:
            from azure.identity.aio import AzureCliCredential, get_bearer_token_provider

            credential = await stack.enter_async_context(AzureCliCredential())
            http_client = await stack.enter_async_context(
                DefaultAsyncHttpxClient(event_hooks={"response": [azure_retry_headers]})
            )
            client_options.update(
                base_url=args.azure_endpoint.rstrip("/") + "/openai/v1/",
                api_key=get_bearer_token_provider(
                    credential, "https://cognitiveservices.azure.com/.default"
                ),
                http_client=http_client,
                max_retries=6,
            )
        client = await stack.enter_async_context(AsyncOpenAI(**client_options))
        model = OpenAIResponsesModel(model=args.model, openai_client=client)
        settings = ModelSettings(
            reasoning={"effort": "medium"}, max_tokens=args.max_output_tokens, store=False
        )
        config = {
            "output_type": CaseExtraction,
            "worker_agent": Agent(
                name="Case worker", model=model, model_settings=settings,
                instructions=WORKER_INSTRUCTIONS, tools=[query_case_sql, edit_state],
            ),
            "reviewer_agent": Agent(
                name="Final reviewer", model=model, model_settings=settings,
                instructions=REVIEWER_INSTRUCTIONS, tools=[query_case_sql, edit_state],
                output_type=CaseReview,
            ),
            "should_keep": lambda case: case.review.extractable,
            "id_field": "case_id",
            "metadata_field": "metadata",
            "artifacts_field": "conversations",
            "artifact_sort_field": "idx",
            "max_batch_chars": batch_chars,
            "max_query_chars": 12000,
            "concurrency": args.concurrency,
            "agent_concurrency": args.concurrency,
            "rpm": args.rpm,
            "timeout": max(900, 60 * 100 / args.rpm),
            "retries": 3,
            "run_config": RunConfig(tracing_disabled=True),
        }
        embedding = {
            "backend": OpenAIEmbeddings(client=client, model=args.embedding_model),
            "state_to_text": state_to_text,
            "batch_size": 32,
            "concurrency": 200,
            "rpm": 700,
        }
        pipeline = LocalPipeline(
            directory / "pipeline", extraction=config, embedding=embedding, show_progress=True
        )
        result = await index_with_retries(pipeline, inputs, directory)
        pass_counts = check_extraction(result["extraction"], len(inputs), full=args.full)
        if result["summary"]["stored_embedded"] != len(result["extraction"]["extracted_cases"]):
            raise RuntimeError("Embedding incomplete; inspect index_summary.json")
        reopened = LocalPipeline(directory / "pipeline", extraction=config, embedding=embedding)
        reopen_result = await reopened.index(inputs)
        if set(reopen_result["skipped_ids"]) != {case["case_id"] for case in inputs}:
            raise RuntimeError("Reopening the pipeline did not preserve all indexed IDs")
        metrics = {}
        for progress in PROGRESSES:
            queries = [build_query(case["conversations"], progress) for case in tests]
            retrieval = await reopened.retrieve(
                queries, top_k=args.top_k, concurrency=60, rpm=200
            )
            rows, scores = score_retrieval(
                retrieval, tests, indexed, max_context_tokens=manifest["max_context_tokens"]
            )
            metrics[str(progress)] = scores
            save_json(directory / f"progress_{progress:03d}.json", {
                "progress": progress,
                "rows": rows,
                "metrics": scores,
                "embedding_usage": retrieval["embedding_usage"],
            })
            print(f"Progress {progress}%: {scores}", flush=True)
        manifest.update(status="completed", passes=pass_counts, metrics=metrics)
        save_json(directory / "manifest.json", manifest)
        print(json.dumps({"saved": str(directory), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
