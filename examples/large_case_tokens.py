"""Live one-case token experiment; loads existing .env without saving credentials.

Run from the repository root with the development extras installed. Generates at least
22k tokens of synthetic artifact text and writes input, full results, and counts
under outputs/large-case-tokens/<run-id>/. Makes real OpenAI API calls.
Requires the optional tiktoken and python-dotenv packages for this experiment.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import tiktoken
from agents import Agent, ModelSettings, OpenAIResponsesModel, RunConfig
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
from raft.embedding.openai import OpenAIEmbeddings
from raft.storage import save_json
from raft.tools import edit_state, query_case_sql

PHASES = [
    (
        "Incident discovery",
        "The ingestion service stopped acknowledging some accepted events. The customer reported "
        "a growing queue and sporadic request timeouts after a deployment. There was no confirmed "
        "data loss; accepted event IDs remained in the durable input queue. The on-call team "
        "started a timeline and compared affected and unaffected tenants.",
        "Collect the symptom and scope without assigning a root cause. Preserve accepted event IDs "
        "so later reconciliation can distinguish delayed acknowledgement from permanent loss.",
    ),
    (
        "Network hypothesis",
        "Engineers suspected packet loss between ingestion workers and the database. Packet "
        "captures showed established connections without retransmission bursts. An unaffected "
        "control tenant used the same network path. Moving one canary worker to another zone "
        "did not remove its timeouts, so the network explanation remained unsupported.",
        "The cross-zone comparison weakens the network hypothesis. Continue checking the "
        "application path rather than recording network packet loss as a confirmed cause.",
    ),
    (
        "Payload hypothesis",
        "The customer suspected large compressed payloads. Support replayed small and large "
        "events against a canary and disabled compression for a controlled sample. Both groups "
        "still encountered connection acquisition waits. CPU and memory remained below their "
        "operational limits. Event size and compression did not explain the failure pattern.",
        "Record this as a negative experiment. The evidence points toward waiting for a shared "
        "resource; successful small requests alone do not prove that large messages are faulty.",
    ),
    (
        "Pool investigation",
        "Traces located the delay before database query execution, while workers waited to "
        "borrow a pooled connection. The pool advertised a capacity of 32, but available leases "
        "declined after request cancellations. Database-side query latency stayed normal. "
        "A restart restored availability temporarily, then cancellation traffic reproduced it.",
        "Pool exhaustion is now observed, but the cause of the missing leases still needs "
        "verification. A restart is a temporary mitigation, not evidence of a permanent repair.",
    ),
    (
        "Confirmed root cause",
        "A controlled cancellation test and code review found that the newly deployed "
        "ingestion client wrapper released its connection lease only on normal completion. "
        "Cancellation after acquisition bypassed that release. Repeated cancellations exhausted "
        "the pool even though no query was running on the leaked leases.",
        "The confirmed root cause is a missing lease release on the cancellation path in the "
        "new client wrapper. Distinguish this application resource leak from database slowness.",
    ),
    (
        "Temporary mitigation",
        "Operators reduced ingestion concurrency and restarted affected workers in small "
        "groups. Queue progress resumed and timeout frequency decreased, but the cancellation "
        "reproduction still leaked leases on the old build. Durable queue retention was extended "
        "during recovery, and consumers kept their existing idempotency checks enabled.",
        "Describe the mitigation as temporary. Preserve the risk of recurrence until the "
        "corrected wrapper passes cancellation tests; do not claim that restart fixed the bug.",
    ),
    (
        "Fix validation",
        "The patch moved lease cleanup into a finally path covering success, failure, and "
        "cancellation. The team tested cancellation before acquisition, immediately after "
        "acquisition, and during execution. The patched canary returned its active lease count "
        "to baseline after each test, while the old-build control retained leaked leases.",
        "The A/B canary supports the cleanup fix. Roll out gradually and keep the durable "
        "event reconciliation separate from the connection-pool regression test.",
    ),
    (
        "Recovery and closure",
        "The corrected wrapper reached all affected workers. Available leases remained stable "
        "under normal and cancellation-heavy traffic, and the backlog drained. Reconciliation "
        "matched accepted event IDs to committed records with no missing events or duplicate "
        "commits. The customer confirmed recovery after a two-hour monitoring window.",
        "Close with the confirmed resource-leak cause, the cleanup fix, and verification "
        "results. Preserve the earlier unsuccessful hypotheses as investigation history.",
    ),
]


def serialized(value):
    """Match RAFT's JSON serialization for comparing case/state/history sizes."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def make_case(encoding, minimum_tokens):
    def build(per_phase):
        artifacts = []
        started = datetime(2026, 9, 1, 8, tzinfo=timezone.utc)
        for phase_index, (phase, finding, interpretation) in enumerate(PHASES):
            for observation in range(per_phase):
                number = len(artifacts) + 1
                tenant = f"tenant-{observation % 7 + 1:02d}"
                worker = f"ingest-{observation % 5 + 1:02d}"
                sample = 200 + observation * 13
                cancelled = 3 + observation % 9
                pending = max(0, 1800 + phase_index * 170 + observation * 19)
                if phase_index == 7:
                    pending = max(0, 400 - observation * 40)
                text = (
                    f"Observation {number}: {phase}. {finding}\n\n"
                    f"The diagnostic slice covers {tenant} on {worker}, with {sample} accepted "
                    f"events and {cancelled} deliberately cancelled probe requests. Queue depth "
                    f"at the start of this slice was {pending}. Probe IDs in this slice use the "
                    f"prefix EV-{number:04d}; retries retain their original event ID rather than "
                    "creating a new logical event. These probes are isolated from the customer's "
                    "normal requests so a cancelled test must not be counted as lost production data.\n\n"
                    f"Analyst interpretation: {interpretation} The support handoff links this "
                    f"observation to capture CAP-{number:04d} and change record INC-POOL-204. "
                    "The counters describe only this observation window; do not sum overlapping "
                    "windows as a count of unique affected events. Customer-facing statements "
                    "must distinguish confirmed findings from hypotheses and temporary mitigations."
                )
                artifacts.append({
                    "sequence": number,
                    "timestamp": (started + timedelta(minutes=number * 3)).isoformat(),
                    "source": ["support_note", "diagnostic_log", "customer_update"][observation % 3],
                    "text": text,
                })
        return {
            "id": "LARGE-POOL-001",
            "metadata": {"product": "event-ingestion", "severity": "SEV-2", "synthetic": True},
            "artifacts": artifacts,
        }
    per_phase = 1
    while True:
        case = build(per_phase)
        total = sum(len(encoding.encode(a["text"])) for a in case["artifacts"])
        if total >= minimum_tokens:
            return case
        per_phase += 1


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-tokens", type=int, default=22_000)
    parser.add_argument("--max-batch-chars", type=int, default=16_000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    load_dotenv(repo / ".env")
    model = os.getenv("RAFT_AGENT_MODEL", "gpt-5.2")
    encoding = tiktoken.encoding_for_model(model)
    def count(value):
        return len(encoding.encode(serialized(value), disallowed_special=()))
    case = make_case(encoding, args.min_tokens)
    directory = args.output_dir or repo / "outputs" / "large-case-tokens" / uuid4().hex[:8]
    if (directory / "pipeline" / "catalog.json").exists():
        raise ValueError("Use a fresh output directory so indexing usage describes this experiment.")
    save_json(directory / "input_case.json", case)
    source = {
        "artifact_text_tokens": sum(len(encoding.encode(a["text"])) for a in case["artifacts"]),
        "artifact_json_tokens_individually": sum(count(a) for a in case["artifacts"]),
        "artifact_list_tokens": count(case["artifacts"]),
        "full_case_json_tokens": count(case),
        "artifacts": len(case["artifacts"]),
        "max_artifact_chars": max(len(serialized(a)) for a in case["artifacts"]),
    }
    assert source["artifact_text_tokens"] >= args.min_tokens
    assert source["max_artifact_chars"] <= args.max_batch_chars
    print(json.dumps({"directory": str(directory), "source": source}, indent=2), flush=True)
    if args.prepare_only:
        return

    initial_contexts = []

    class MeasuredModel(OpenAIResponsesModel):
        async def get_response(self, *args, **kwargs):
            inputs = kwargs.get("input", [])
            if isinstance(inputs, list) and not any(
                item.get("type") == "function_call_output" for item in inputs
            ):
                for item in inputs:
                    if item.get("role") != "user":
                        continue
                    content = item.get("content", "")
                    if isinstance(content, list):
                        content = "".join(part.get("text", "") for part in content)
                    if content.startswith(("Pass context:\n", "Review context:\n")):
                        payload = json.loads(content.split("\n", 1)[1])
                        initial_contexts.append({
                            "stage": "worker" if "pass_number" in payload else "reviewer",
                            "pass_number": payload.get("pass_number"),
                            "context_tokens": len(encoding.encode(content)),
                            "current_state_tokens": count(payload.get("current_state", payload.get("output"))),
                            "batch_tokens": count(payload["batch"]) if "batch" in payload else None,
                        })
            return await super().get_response(*args, **kwargs)

    async with AsyncOpenAI(max_retries=0) as client:
        sdk_model = MeasuredModel(model=model, openai_client=client)
        worker = Agent(
            name="Large case worker", model=sdk_model,
            instructions=WORKER_INSTRUCTIONS + (
                "\nCompletion rule: finish_pass marks THIS BATCH complete, not the whole case. "
                "Set finish_pass=True after the supplied batch, even if batch.is_last=False. "
                "Before final text, ensure edit_state reports pass_finished=True; if not, "
                "call it again with patch_json='[]' and finish_pass=True."
            ),
            model_settings=ModelSettings(reasoning={"effort": "low"}, max_tokens=6000),
            tools=[query_case_sql, edit_state],
        )
        reviewer = Agent(
            name="Large case reviewer", model=sdk_model,
            instructions=REVIEWER_INSTRUCTIONS + (
                "\nThis is a synthetic evaluation case. Judge eligibility by technical content, "
                "not by the synthetic metadata flag."
            ),
            model_settings=ModelSettings(reasoning={"effort": "low"}, max_tokens=6000),
            tools=[query_case_sql, edit_state], output_type=CaseReview,
        )
        backend = OpenAIEmbeddings(client=client, model="text-embedding-3-small")
        pipeline = LocalPipeline(
            output_dir=directory / "pipeline",
            extraction={
                "worker_agent": worker, "reviewer_agent": reviewer, "output_type": CaseExtraction,
                "should_keep": lambda case: case.review.extractable,
                "id_field": "id", "metadata_field": "metadata", "artifacts_field": "artifacts",
                "artifact_sort_field": "sequence", "max_batch_chars": args.max_batch_chars,
                "max_query_chars": 12_000, "concurrency": 1, "agent_concurrency": 1,
                "max_turns": 20, "timeout": 600, "retries": 1, "rpm": 300,
                "run_config": RunConfig(tracing_disabled=True),
            },
            embedding={"backend": backend, "state_to_text": state_to_text, "batch_size": 64},
            graph={"backend": backend, "case_to_text": case_to_text, "batch_size": 64},
            bm25=True, show_progress=True,
        )
        result = await pipeline.index([case])
        save_json(directory / "index_result.json", result)
        failures = result["extraction"]["failed_cases"] + result["embedding"]["failed_cases"]
        if failures or not result["indexed_cases"]:
            print("Experiment failed or filtered; inspect index_result.json", flush=True)
            raise SystemExit(1)
        extracted = result["indexed_cases"][0]["case"]
        save_json(directory / "extracted_case.json", extracted)
        revisions = extracted.execution["revisions"]
        revision_rows = [
            {
                "revision": r["revision_id"], "stage": r["stage"], "pass_number": r["pass_number"],
                "state_tokens": count(r["state"]), "edits_tokens": count(r["edits"]),
                "full_revision_tokens": count(r),
            }
            for r in revisions
        ]
        history_queries = []
        for invocation in extracted.execution["tool_calls"]:
            for round_ in invocation["rounds"]:
                for call in round_["calls"]:
                    if call["name"] == "query_case_sql" and "state_revisions" in call["args"].get("query", "").lower():
                        history_queries.append({
                            "query": call["args"]["query"],
                            "result_tokens": count(call["output"]),
                            "error": call["output"].get("error") if isinstance(call["output"], dict) else None,
                        })
        retrieval = await pipeline.retrieve(["connection pool exhausted after cancelled requests"], top_k=1)
        save_json(directory / "retrieval_result.json", retrieval)
        report = {
            "case_id": extracted.id, "model": model, "tokenizer": encoding.name,
            "serialization": "json.dumps(ensure_ascii=False, sort_keys=True); default whitespace",
            "source": source,
            "settings": {"max_batch_chars": args.max_batch_chars, "max_query_chars": 12_000},
            "worker_passes": extracted.execution["passes"],
            "attempts": extracted.execution["attempts"],
            "timeline_entries": len(extracted.output.timeline),
            "timeline_entry_tokens": [
                {"item_index": i, "text_tokens": len(encoding.encode(entry.narrative)),
                 "json_tokens": count(entry.model_dump(mode="json")), "narrative": entry.narrative}
                for i, entry in enumerate(extracted.output.timeline)
            ],
            "initial_agent_contexts": initial_contexts,
            "final_state_tokens": count(extracted.output.model_dump(mode="json")),
            "review_output_tokens": count(extracted.review.model_dump(mode="json")),
            "revision_history_tokens": count(revisions),
            "state_snapshots_tokens_sum": sum(r["state_tokens"] for r in revision_rows),
            "edits_tokens_sum": sum(r["edits_tokens"] for r in revision_rows),
            "revisions": revision_rows,
            "history_query_results": history_queries,
            "history_query_result_tokens_sum": sum(q["result_tokens"] for q in history_queries),
            "agent_usage_by_model": {
                name: {k: v for k, v in usage.items() if k != "request_usage_entries"}
                for name, usage in extracted.execution["usage"].items()
            },
            "timeline_embedding_usage": result["embedding"]["embedding_usage"],
            "graph_embedding_usage": result["graph"]["embedding_usage"],
            "query_embedding_usage": retrieval["embedding_usage"],
            "pipeline_summary": result["summary"],
            "graph_summary": result["graph"]["summary"],
            "root_cause": extracted.output.root_cause,
            "retrieval_errors": [r["error"] for r in retrieval["results"]],
            "limitations": "One synthetic, repetitive-template case; not a production benchmark. "
            "Local serialized token counts exclude model/tool wrappers and are not provider billing.",
        }
        save_json(directory / "token_report.json", report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
