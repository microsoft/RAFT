"""Opt-in live extraction check for OpenAI Agents SDK using synthetic cases.

Run: python examples/extraction_smoke.py
Uses .env; makes billable OpenAI calls. Not part of pytest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from raft import run_cases
from raft.defaults import REVIEWER_INSTRUCTIONS
from raft.storage import save_json


class SmokeOutput(BaseModel):
    case_id: str
    facts: list[str]
    reference: str


class SmokeReview(BaseModel):
    keep: bool


INSTRUCTIONS = """You are a synthetic case extraction worker.
Each pass includes target_schema, current_state, metadata, coverage and batch.
Continue from current_state. Read all supplied batch.items artifact_json values.
Set case_id to the id supplied in the pass context.
Use query_case_sql each pass to inspect artifact positions and content.
Collect every artifact's fact field exactly once in facts, in artifact order.
For this smoke test additionally call lookup_reference with the case metadata's
reference_code each pass; store its returned value in reference.
Write through edit_state (RFC6902 JSON Patch encoded in patch_json), not final text.
Create fields/arrays before append. Set finish_pass=true after processing the batch.
Repair any final Pydantic validation errors with edit_state. Once the tool reports
pass_finished=true, return a brief confirmation and stop. Do not skip any batch.
"""


async def lookup_reference(code: str) -> str:
    """Return a synthetic reference value for the provided case reference code."""
    return "reference:" + code


def make_openai(client, model):
    from agents import Agent, ModelSettings, OpenAIResponsesModel, function_tool

    from raft.tools import edit_state, query_case_sql

    worker = Agent(
        name="worker",
        model=OpenAIResponsesModel(model=model, openai_client=client),
        instructions=INSTRUCTIONS,
        model_settings=ModelSettings(reasoning={"effort": "low"}, max_tokens=2000),
        tools=[query_case_sql, edit_state, function_tool(lookup_reference)],
    )
    reviewer = Agent(
        name="reviewer",
        model=OpenAIResponsesModel(model=model, openai_client=client),
        instructions=REVIEWER_INSTRUCTIONS,
        model_settings=ModelSettings(reasoning={"effort": "low"}, max_tokens=2000),
        tools=[query_case_sql, edit_state],
        output_type=SmokeReview,
    )
    return worker, reviewer


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["gpt-5.2", "gpt-5.4"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing; configure .env")
    cases = [
        {
            "id": "short",
            "metadata": {"reference_code": "S", "kind": "incident"},
            "artifacts": [{"fact": "ALPHA"}],
        },
        {
            "id": "long",
            "metadata": {"reference_code": "L", "kind": "incident"},
            "artifacts": [{"fact": value} for value in ["BRAVO", "CHARLIE", "DELTA"]],
        },
        {
            "id": "test-case",
            "metadata": {"reference_code": "F", "kind": "test"},
            "artifacts": [{"fact": "TEST"}],
        },
    ]
    async with AsyncOpenAI(max_retries=0, timeout=60) as client:
        from agents import OpenAIResponsesModel, RunConfig

        worker, reviewer = make_openai(client, args.models[0])
        models = [OpenAIResponsesModel(model=name, openai_client=client) for name in args.models]
        pass_counts = {}

        def configure_run(agent, context):
            index = pass_counts.get(context.case_id, 0)
            if context.stage == "worker":
                pass_counts[context.case_id] = index + 1
                model = models[index % len(models)]
            else:
                model = models[-1]
            return RunConfig(model=model, tracing_disabled=True)

        result = await run_cases(
            cases=cases,
            worker_agent=worker,
            reviewer_agent=reviewer,
            output_type=SmokeOutput,
            run_config=configure_run,
            show_progress=True,
            id_field="id",
            artifacts_field="artifacts",
            metadata_field="metadata",
            max_batch_chars=21,
            max_query_chars=1000,
            concurrency=2,
            rpm=120,
            timeout=120,
            retries=0,
            max_turns=8,
        )
    path = root / "outputs" / "extraction_smoke" / "openai.json"
    save_json(path, result)
    assert not result["failed_cases"], [
        (case["id"], case["error_type"], case["error_message"]) for case in result["failed_cases"]
    ]
    assert len(result["extracted_cases"]) == 3
    by_id = {case.id: case for case in result["extracted_cases"]}
    for original in cases:
        case = by_id[original["id"]]
        assert isinstance(case.output, SmokeOutput)
        assert case.output.case_id == case.id
        assert case.output.facts == [artifact["fact"] for artifact in original["artifacts"]]
        assert case.output.reference == "reference:" + original["metadata"]["reference_code"]
        assert case.execution["passes"] == len(original["artifacts"])
        assert "coverage" not in case.model_dump()
        usage = case.execution["usage"]
        assert usage and set(usage) <= set(args.models)
        assert all(value["total_tokens"] > 0 for value in usage.values())
        if len(args.models) > 1:
            assert args.models[0] in usage and args.models[-1] in usage
        for invocation in case.execution["tool_calls"]:
            if "pass" not in invocation:
                continue
            calls = [call for round_ in invocation["rounds"] for call in round_["calls"]]
            assert {"query_case_sql", "edit_state", "lookup_reference"} <= {
                call["name"] for call in calls
            }
            assert all("output" in call for call in calls)
    print(
        json.dumps(
            {
                "models": args.models,
                "usage": {case.id: case.execution["usage"] for case in by_id.values()},
                "summary": result["summary"],
                "passes": {case.id: case.execution["passes"] for case in by_id.values()},
                "saved": str(path),
                "checks": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
