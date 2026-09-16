import asyncio
import json
from types import SimpleNamespace

import pytest
from agent_helpers import REVIEWER, WorkerTestRunner, run_cases
from agents import Agent, function_tool
from pydantic import BaseModel, RootModel

from raft.extraction import _agent as openai_agents
from raft.extraction.context import _build_case_context
from raft.extraction.state import apply_edit
from raft.tools import edit_state, query_case_sql


class Output(BaseModel):
    extractable: bool
    timeline: list[str]


def case(ticket="a", text="hello"):
    return {
        "ticket": ticket,
        "meta": {"product": "test"},
        "artifacts": [
            {"order": 2, "text": text},
            {"order": 1, "text": "opening"},
        ],
    }


def options(**kwargs):
    return {
        "reviewer_agent": REVIEWER,
        "_agent_runner": WorkerTestRunner(),
        "output_type": Output,
        "worker_agent": Agent(
            name="scan", tools=[query_case_sql, edit_state]
        ),
        "id_field": "ticket",
        "metadata_field": "meta",
        "artifacts_field": "artifacts",
        "rpm": 1000,
        "retries": 0,
        **kwargs,
    }


def edit(context, patch, finish=True):
    return apply_edit(
        context=context,
        patch_json=json.dumps(patch),
        finish_pass=finish,
    )


@pytest.mark.asyncio
async def test_scans_all_case_sizes_and_preserves_agent_and_tools(monkeypatch):
    @function_tool
    def lookup(code: str) -> str:
        return code

    tools = [lookup, query_case_sql, edit_state]
    scan = Agent(name="scan", tools=tools, output_type=Output)
    calls = []

    async def run(agent, prompt, *, context, max_turns, hooks, run_config):
        calls.append(agent.name)
        assert lookup in agent.tools
        payload = json.loads(prompt.split("Pass context:\n")[1])
        assert payload["metadata"] == {"product": "test"}
        assert "artifacts" not in payload
        rows = context.query("SELECT artifact_json FROM artifacts ORDER BY position")
        assert json.loads(rows["rows"][0]["artifact_json"])["order"] == 1
        assert agent is scan
        assert agent.output_type is Output
        assert {tool.name for tool in agent.tools} == {"lookup", "query_case_sql", "edit_state"}
        assert edit(
            context,
            [{"op": "add", "path": "", "value": {"extractable": True, "timeline": ["large"]}}],
        )["ok"]
        return SimpleNamespace(new_items=[], final_output="done")

    monkeypatch.setattr(openai_agents.Runner, "run", run)
    results = await run_cases(
        cases=[case("short"), case("long", "x" * 3000)],
        **options(
            worker_agent=scan,
            artifact_sort_field="order",
        ),
    )
    assert results["summary"] == {"total": 2, "extracted": 2, "failed": 0}
    assert calls == ["scan", "scan"]
    assert scan.tools == tools
    assert scan.output_type is Output  # Caller-owned agent is unchanged.
    assert case()["artifacts"][0]["order"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tools, missing",
    [
        ([], "edit_state, query_case_sql"),
        ([query_case_sql], "edit_state"),
        ([edit_state], "query_case_sql"),
    ],
)
async def test_missing_required_tools_fail_before_sdk_run(monkeypatch, tools, missing):
    async def unexpected_run(*args, **kwargs):
        pytest.fail("SDK must not run when required tools are missing")

    monkeypatch.setattr(openai_agents.Runner, "run", unexpected_run)
    worker = Agent(name="scan", tools=tools)
    with pytest.raises(ValueError, match=f"missing required tools: {missing}\\."):
        await run_cases(cases=[case()], **options(worker_agent=worker))
    assert worker.tools == tools


def test_prepare_preserves_default_tool_identity_and_order():
    worker = Agent(
        name="scan",
        output_type=Output,
        tools=[edit_state, query_case_sql],
    )
    prepared = openai_agents._AgentRunner().prepare(worker)
    assert prepared is worker and prepared.tools is worker.tools
    assert len(prepared.tools) == 2
    assert all(a is b for a, b in zip(prepared.tools, worker.tools, strict=True))
    assert prepared.output_type is Output


@pytest.mark.asyncio
async def test_small_case_starts_unread_and_finishes_in_one_pass(monkeypatch):
    received = []

    async def run(agent, prompt, *, context, max_turns, hooks, run_config):
        received.append(agent.name)
        payload = json.loads(prompt.split("Pass context:\n")[1])
        assert payload["current_state"] == {}
        assert payload["target_output_schema"] == Output.model_json_schema()
        assert payload["pass_number"] == 1
        assert "max_query_chars" not in payload
        assert context.max_query_chars == 1000
        assert payload["validation_error"] is None
        coverage = payload["coverage"]
        assert coverage["covered_count"] == 0 and coverage["covered_ranges"] == []
        assert coverage["remaining_count"] == coverage["total_artifacts"] == 2
        assert coverage["basis"] == "committed_preloaded_batches"
        assert not coverage["complete"]
        rows = payload["batch"]["items"]
        assert len(rows) == 2
        assert [json.loads(row["artifact_json"]) for row in rows] == case()["artifacts"]
        assert payload["batch"]["is_last"]
        edit(
            context,
            [{"op": "add", "path": "", "value": {"extractable": True, "timeline": []}}],
        )
        return SimpleNamespace(new_items=[], final_output="done")

    monkeypatch.setattr(openai_agents.Runner, "run", run)
    result = await run_cases(cases=[case()], **options(max_query_chars=1000))
    assert received == ["scan"]
    assert result["extracted_cases"][0].execution["passes"] == 1
    assert "coverage" not in result["extracted_cases"][0].model_dump()


@pytest.mark.asyncio
async def test_worker_retry_resumes_committed_state_and_ranges(monkeypatch):
    calls = 0
    prompts = []

    async def run(agent, prompt, *, context, max_turns, hooks, run_config):
        nonlocal calls
        calls += 1
        prompts.append(json.loads(prompt.split("Pass context:\n")[1]))
        if calls == 1:
            edit(
                context,
                [{"op": "add", "path": "", "value": {"extractable": True, "timeline": ["first"]}}],
            )
        elif calls == 2:
            edit(context, [{"op": "add", "path": "/timeline/-", "value": "discard"}], finish=False)
            raise TimeoutError("temporary")
        else:
            assert context.pending_state["timeline"] == ["first"]
            edit(context, [{"op": "add", "path": "/timeline/-", "value": "second"}])
        return SimpleNamespace(new_items=[], final_output="done")

    monkeypatch.setattr(openai_agents.Runner, "run", run)
    monkeypatch.setattr("raft.extraction.runner._retry_delay", lambda *args: 0)
    result = await run_cases(cases=[case()], **options(retries=1, max_batch_chars=31))
    output = result["extracted_cases"][0]
    assert output.output.timeline == ["first", "second"]
    assert [r["state"]["timeline"] for r in output.execution["revisions"]] == [
        ["first"], ["first", "second"]
    ]
    assert [r["revision_id"] for r in output.execution["revisions"]] == [1, 2]
    assert (
        output.execution["passes"],
        output.execution["attempts"],
        len(output.execution["tool_calls"]),
    ) == (2, 2, 4)
    assert prompts[1] == prompts[2]
    assert prompts[1]["coverage"]["uncovered_ranges"] == [
        {"start_position": 1, "end_position_exclusive": 2}
    ]


@pytest.mark.asyncio
async def test_retry_budget_is_case_wide_and_failed_finished_pass_is_not_committed(monkeypatch):
    passes = []

    async def run(agent, prompt, *, context, max_turns, hooks, run_config):
        payload = json.loads(prompt.split("Pass context:\n")[1])
        number = payload["pass_number"]
        passes.append(number)
        if len(passes) == 3:
            assert payload["current_state"]["timeline"] == ["pass 1"]
            assert payload["coverage"]["covered_count"] == 1
        edit(
            context,
            [
                {
                    "op": "add",
                    "path": "",
                    "value": {
                        "extractable": True,
                        "timeline": [f"pass {number}"],
                    },
                }
            ],
        )
        # Even a finished edit is not committed if the SDK invocation then fails.
        if len(passes) in (2, 4):
            raise TimeoutError("temporary")
        return SimpleNamespace(new_items=[], final_output="done")

    monkeypatch.setattr(openai_agents.Runner, "run", run)
    monkeypatch.setattr("raft.extraction.runner._retry_delay", lambda *args: 0)
    data = case()
    data["artifacts"].append({"order": 3, "text": "third"})
    result = await run_cases(cases=[data], **options(retries=1, max_batch_chars=31))
    failure = result["failed_cases"][0]
    assert passes == [1, 2, 2, 3]
    assert failure["failure_type"] == "retry_exhausted"
    assert failure["partial_state"]["timeline"] == ["pass 2"]
    assert [r["state"]["timeline"] for r in failure["execution"]["revisions"]] == [
        ["pass 1"], ["pass 2"]
    ]
    assert failure["coverage"]["covered_count"] == 2
    assert failure["execution"]["attempts"] == failure["execution"]["passes"] == 2


@pytest.mark.asyncio
async def test_timeout_bounds_entire_worker_attempt(monkeypatch):
    async def run(agent, prompt, *, context, max_turns, hooks, run_config):
        await asyncio.sleep(0.35)
        # A successful first batch followed by a timeout during the next batch.
        edit(
            context,
            [{"op": "add", "path": "", "value": {"extractable": True, "timeline": ["partial"]}}],
        )
        return SimpleNamespace(new_items=[], final_output="done")

    monkeypatch.setattr(openai_agents.Runner, "run", run)
    result = await run_cases(cases=[case()], **options(timeout=0.6, max_batch_chars=31))
    failure = result["failed_cases"][0]
    assert failure["error_category"] == "timeout"
    assert failure["partial_state"]["timeline"] == ["partial"]
    assert len(failure["execution"]["tool_calls"]) == 2


@pytest.mark.asyncio
async def test_keeps_ineligible_outputs_validation_failure_and_duplicates(monkeypatch):
    async def run(agent, prompt, *, context, max_turns, hooks, run_config):
        value = {"extractable": context.case_id != "rfi", "timeline": []}
        if context.case_id == "invalid":
            value.pop("timeline")
        edit(context, [{"op": "add", "path": "", "value": value}])
        return SimpleNamespace(new_items=[], final_output="done")

    monkeypatch.setattr(openai_agents.Runner, "run", run)
    result = await run_cases(
        cases=[case("ok"), case("rfi"), case("invalid"), case("ok")], **options()
    )
    assert result["summary"] == {"total": 4, "extracted": 2, "failed": 2}
    assert result["extracted_cases"][1].output.extractable is False
    assert "filtered_cases" not in result
    assert result["failed_cases"][1]["execution"]["attempts"] == 0


@pytest.mark.asyncio
async def test_missing_finish_and_pass_limit(monkeypatch):
    async def run(agent, prompt, *, context, max_turns, hooks, run_config):
        return SimpleNamespace(new_items=[], final_output="forgot tool")

    monkeypatch.setattr(openai_agents.Runner, "run", run)
    result = await run_cases(cases=[case()], **options())
    assert result["failed_cases"][0]["error_category"] == "model_behavior"

    async def endless(agent, prompt, *, context, max_turns, hooks, run_config):
        edit(context, [])
        return SimpleNamespace(new_items=[], final_output="done")

    monkeypatch.setattr(openai_agents.Runner, "run", endless)
    result = await run_cases(
        cases=[{**case(), "artifacts": [{"text": "a"}] * 3}], **options(max_passes=2, max_batch_chars=13)
    )
    assert result["failed_cases"][0]["error_category"] == "max_case_passes"


@pytest.mark.asyncio
async def test_concurrency_and_cancellation_close_databases(monkeypatch):
    active = maximum = 0
    contexts = []

    async def run(agent, prompt, *, context, max_turns, hooks, run_config):
        nonlocal active, maximum
        contexts.append(context)
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.01)
            edit(
                context, [{"op": "add", "path": "", "value": {"extractable": True, "timeline": []}}]
            )
            return SimpleNamespace(new_items=[], final_output="done")
        finally:
            active -= 1

    monkeypatch.setattr(openai_agents.Runner, "run", run)
    result = await run_cases(cases=[case(str(i)) for i in range(5)], **options(concurrency=2))
    assert maximum == 2 and result["summary"]["extracted"] == 5
    task = asyncio.create_task(run_cases(cases=[case()], **options()))
    await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for context in contexts:
        with pytest.raises(Exception, match="closed database"):
            context.connection.execute("SELECT 1")


def test_sql_order_budget_and_atomic_patches_with_validation_repair():
    context = _build_case_context(
        case(),
        id_field="ticket",
        metadata_field="meta",
        artifacts_field="artifacts",
        artifact_sort_field="order",
        max_query_chars=200,
        final_output_type=Output,
    )
    try:
        rows = context.query("SELECT position, original_position FROM artifacts ORDER BY position")
        assert rows["rows"][0] == {"position": 0, "original_position": 1}
        assert "error" in context.query("DELETE FROM artifacts")
        for _ in range(10):
            assert context.query("SELECT position FROM artifacts")["row_count"] == 2
        context.begin_pass({"extractable": True, "timeline": []})
        failed = edit(
            context,
            [
                {"op": "add", "path": "/timeline/-", "value": "discard"},
                {"op": "remove", "path": "/missing"},
            ],
        )
        assert not failed["ok"] and context.pending_state["timeline"] == []
        failed = edit(context, [{"op": "remove", "path": "/extractable"}])
        assert failed["patch_applied"] and not context.pass_finished
        assert edit(context, [{"op": "add", "path": "/extractable", "value": False}])["ok"]
    finally:
        context.close()


def test_root_model_and_escaped_json_pointer_keys():
    context = _build_case_context(
        case(),
        id_field="ticket",
        metadata_field="meta",
        artifacts_field="artifacts",
        artifact_sort_field=None,
        final_output_type=RootModel[list[dict[str, str | None]]],
    )
    try:
        context.begin_pass({})
        assert edit(
            context, [{"op": "add", "path": "", "value": [{"a/b~c": "old"}]}], finish=False
        )["ok"]
        assert edit(context, [{"op": "replace", "path": "/0/a~1b~0c", "value": None}])["ok"]
        assert context.pending_state == [{"a/b~c": None}]
    finally:
        context.close()
