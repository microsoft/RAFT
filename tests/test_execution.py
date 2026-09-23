"""Exercise the real Agents SDK loop with scripted model responses, without API calls."""

import asyncio
import json
from ast import literal_eval

import pytest
from agent_helpers import REVIEWER, ReviewResult, WorkerTestRunner, run_cases
from agents import Agent, RunContextWrapper, function_tool
from agents.items import ModelResponse
from agents.models.interface import Model, ModelTracing
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from pydantic import BaseModel

from raft import ExtractedCase
from raft.extraction._agent import _AgentRunner
from raft.extraction.context import CaseContext
from raft.extraction.state import apply_edit
from raft.storage import save_json
from raft.tools import edit_state, query_case_sql, write_handoff_note


class Output(BaseModel):
    text: str


def call(name, args, identifier):
    return ResponseFunctionToolCall(
        type="function_call",
        name=name,
        arguments=json.dumps(args),
        call_id=identifier,
    )


def message(text):
    return ResponseOutputMessage(
        type="message",
        id="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


class ScriptedModel(Model):
    def __init__(self, steps):
        self.steps = iter(steps)

    async def get_response(self, *args, **kwargs):
        assert kwargs["tracing"] == ModelTracing.DISABLED
        step = next(self.steps)
        if isinstance(step, Exception):
            raise step
        return ModelResponse(
            output=step,
            response_id=None,
            usage=Usage(
                requests=1,
                input_tokens=10,
                output_tokens=4,
                total_tokens=14,
                input_tokens_details=InputTokensDetails(cached_tokens=2, cache_write_tokens=0),
                output_tokens_details=OutputTokensDetails(reasoning_tokens=1),
            ),
        )

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


def options(model, tools=(), **kwargs):
    return {
        "reviewer_agent": REVIEWER,
        "review_output_type": ReviewResult,
        "_agent_runner": WorkerTestRunner(),
        "cases": [{"id": "case", "meta": {}, "items": [{"text": "a"}, {"text": "b"}]}],
        "output_type": Output,
        "worker_agent": Agent(
            name="worker", model=model, tools=[query_case_sql, edit_state, *tools]
        ),
        "id_field": "id",
        "metadata_field": "meta",
        "artifacts_field": "items",
        "retries": 0,
        "rpm": 1000,
        **kwargs,
    }


def check_usage(execution, requests):
    assert set(execution) == {
        "usage", "tool_calls", "elapsed_seconds", "passes", "attempts", "revisions", "handoff_notes"
    }
    usage = execution["usage"]["unknown"]
    assert usage["requests"] == requests
    assert usage["input_tokens"] == requests * 10
    assert usage["output_tokens"] == requests * 4
    assert usage["total_tokens"] == requests * 14
    assert usage["input_tokens_details"]["cached_tokens"] == requests * 2
    assert usage["output_tokens_details"]["reasoning_tokens"] == requests
    assert len(usage["request_usage_entries"]) == requests
    assert execution["elapsed_seconds"] >= 0


@pytest.mark.asyncio
async def test_real_sdk_parallel_tools_keep_model_order_and_json_outputs(tmp_path):
    finished = []

    @function_tool
    async def lookup(value: str) -> dict:
        if value == "slow":
            await asyncio.sleep(0.03)
        finished.append(value)
        return {"value": value}

    @function_tool
    def nothing() -> None:
        return None

    model = ScriptedModel(
        [
            [call("lookup", {"value": "slow"}, "a"), call("lookup", {"value": "fast"}, "b")],
            [call("nothing", {}, "c"), edit_call("done", "edit")],
            [message("done")],
        ]
    )
    result = await run_cases(**options(model, [lookup, nothing]))
    assert not result["failed_cases"], result
    case = result["extracted_cases"][0]
    assert isinstance(case, ExtractedCase) and case.output.text == "done"
    check_usage(case.execution, 3)
    assert finished == ["fast", "slow"]
    runs = [r for r in case.execution["tool_calls"] if "pass" in r]
    assert len(runs) == 1 and (runs[0]["attempt"], runs[0]["pass"]) == (1, 1)
    rounds = runs[0]["rounds"]
    assert [r["round"] for r in rounds] == [1, 2, 3]
    assert all(r["agent"] == "worker" for r in rounds)
    assert rounds[0]["calls"] == [
        {"call_id": "a", "name": "lookup", "args": {"value": "slow"}, "output": {"value": "slow"}},
        {"call_id": "b", "name": "lookup", "args": {"value": "fast"}, "output": {"value": "fast"}},
    ]
    assert rounds[1]["calls"][0]["output"] is None
    assert rounds[2]["calls"] == []
    save_json(tmp_path / "result.json", result)
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["extracted_cases"][0]["execution"] == case.execution


def edit_call(text, identifier):
    return call(
        "edit_state",
        {
            "patch_json": json.dumps([{"op": "add", "path": "", "value": {"text": text}}]),
            "finish_pass": True,
        },
        identifier,
    )


def review_call(value=None, identifier="review"):
    return call(
        "edit_state",
        {
            "target": "review",
            "patch_json": json.dumps([{
                "op": "add", "path": "", "value": {"keep": True} if value is None else value,
            }]),
            "finish_pass": True,
        },
        identifier,
    )


@pytest.mark.asyncio
async def test_custom_required_tools_are_preserved_executed_and_recorded():
    invoked = []

    @function_tool(name_override="query_case_sql")
    def custom_query(ctx: RunContextWrapper[CaseContext], query: str) -> dict:
        invoked.append("custom query")
        return {**ctx.context.query(query), "custom": True}

    # Different arguments are allowed; the editor still honors the context contract.
    @function_tool(name_override="edit_state")
    def custom_edit(ctx: RunContextWrapper[CaseContext], text: str) -> dict:
        invoked.append("custom edit")
        return apply_edit(
            context=ctx.context,
            patch_json=json.dumps([{"op": "add", "path": "", "value": {"text": text}}]),
            finish_pass=True,
        )

    @function_tool
    def extra_tool() -> str:
        return "extra"

    model = ScriptedModel(
        [
            [
                call(
                    "query_case_sql",
                    {"query": "SELECT artifact_json FROM artifacts ORDER BY position"},
                    "q",
                )
            ],
            [call("edit_state", {"text": "custom output"}, "e"), call("extra_tool", {}, "x")],
            [message("done")],
        ]
    )
    worker = Agent(name="custom", model=model, tools=[custom_edit, extra_tool, custom_query])
    prepared = _AgentRunner().prepare(worker)
    assert all(a is b for a, b in zip(prepared.tools, worker.tools, strict=True))
    result = await run_cases(**options(model, worker_agent=worker))
    assert not result["failed_cases"], result
    case = result["extracted_cases"][0]
    assert case.output.text == "custom output"
    assert "coverage" not in case.model_dump()
    assert invoked == ["custom query", "custom edit"]
    check_usage(case.execution, 3)
    rounds = case.execution["tool_calls"][0]["rounds"]
    assert rounds[0]["calls"][0]["output"]["custom"] is True
    assert rounds[1]["calls"][0]["args"] == {"text": "custom output"}
    assert rounds[1]["calls"][0]["output"]["pass_finished"] is True
    assert rounds[1]["calls"][1]["output"] == "extra"


@pytest.mark.asyncio
async def test_real_sdk_worker_passes_retry_usage_and_tool_order(monkeypatch):
    query = {"query": "SELECT position FROM artifacts ORDER BY position"}
    model = ScriptedModel(
        [
            [call("query_case_sql", query, "q1"), edit_call("first", "e1")],
            [message("pass finished")],
            [call("query_case_sql", query, "q2")],
            TimeoutError("model timeout after tool result"),
            [call("query_case_sql", query, "q3"), edit_call("second", "e2")],
            [message("done")],
        ]
    )
    monkeypatch.setattr("raft.extraction.runner._retry_delay", lambda *args: 0)
    result = await run_cases(**options(model, retries=1, batch_budget={"unit": "chars", "limit": 13}))
    assert not result["failed_cases"], result
    case = result["extracted_cases"][0]
    assert case.output.text == "second"
    check_usage(case.execution, 5)
    assert case.execution["passes"] == 2 and case.execution["attempts"] == 2
    runs = [r for r in case.execution["tool_calls"] if "pass" in r]
    assert [(r["attempt"], r["pass"]) for r in runs] == [(1, 1), (1, 2), (2, 2)]
    assert [[c["call_id"] for r in run["rounds"] for c in r["calls"]] for run in runs] == [
        ["q1", "e1"],
        ["q2"],
        ["q3", "e2"],
    ]
    assert runs[1]["rounds"][0]["calls"][0]["output"]["rows"] == [{"position": 0}, {"position": 1}]
    assert all(r["agent"] == "worker" for run in runs for r in run["rounds"])


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_failed_and_timed_out_tools_preserve_observed_usage(timeout):
    @function_tool(failure_error_function=None)
    async def fail() -> str:
        if timeout:
            await asyncio.sleep(10)
        raise ValueError("bad tool configuration")

    model = ScriptedModel([[call("fail", {}, "failed-call")]])
    result = await run_cases(**options(model, [fail], timeout=0.05))
    failure = result["failed_cases"][0]
    execution = failure["execution"]
    check_usage(execution, 1)
    assert execution["passes"] == 0 and execution["attempts"] == 1
    assert execution["tool_calls"][0]["rounds"][0]["calls"] == [
        {"call_id": "failed-call", "name": "fail", "args": {}}
    ]  # Missing output means no result was observed, unlike an actual None result.
    assert failure["error_category"] == ("timeout" if timeout else "user_error")
    assert failure["retryable"] is timeout


@pytest.mark.asyncio
async def test_first_pass_retry_retains_both_attempts(monkeypatch):
    @function_tool
    def lookup() -> str:
        return "not extractable"

    model = ScriptedModel(
        [
            [call("lookup", {}, "a")],
            TimeoutError("retry"),
            [call("lookup", {}, "b"), edit_call("RFI", "edit")],
            [message("done")],
        ]
    )
    monkeypatch.setattr("raft.extraction.runner._retry_delay", lambda *args: 0)
    result = await run_cases(**options(model, [lookup], retries=1))
    assert result["summary"] == {"total": 1, "extracted": 1, "failed": 0}
    case = result["extracted_cases"][0]
    check_usage(case.execution, 3)
    assert case.execution["attempts"] == 2 and case.execution["passes"] == 1
    assert [(r["attempt"], r["pass"]) for r in case.execution["tool_calls"] if "pass" in r] == [(1, 1), (2, 1)]


@pytest.mark.asyncio
async def test_handled_tool_error_preserves_the_error_output():
    @function_tool
    def broken() -> str:
        raise ValueError("bad argument")

    model = ScriptedModel(
        [
            [call("broken", {}, "a"), edit_call("recovered", "edit")],
            [message("done")],
        ]
    )
    result = await run_cases(**options(model, [broken]))
    assert not result["failed_cases"], result
    case = result["extracted_cases"][0]
    check_usage(case.execution, 2)
    output = case.execution["tool_calls"][0]["rounds"][0]["calls"][0]["output"]
    assert "bad argument" in output


@pytest.mark.asyncio
async def test_no_response_has_unknown_usage_and_no_tools():
    result = await run_cases(**options(ScriptedModel([ValueError("configuration error")])))
    execution = result["failed_cases"][0]["execution"]
    assert execution["usage"] == {}
    assert execution["tool_calls"][0]["rounds"][0]["calls"] == []


@pytest.mark.asyncio
async def test_concurrent_cases_do_not_share_tool_outputs_or_usage():
    @function_tool
    async def identify(ctx: RunContextWrapper) -> str:
        await asyncio.sleep(0.005)
        return ctx.context.case_id

    class StatelessModel(ScriptedModel):
        async def get_response(self, *args, **kwargs):
            outputs = [i for i in kwargs["input"] if i.get("type") == "function_call_output"]
            if len(outputs) == 2:
                step = [message("done")]
            elif outputs:
                step = [edit_call(outputs[0]["output"], "edit")]
            else:
                step = [call("identify", {}, "same-id-per-case")]
            # Per-call script avoids sharing an iterator across concurrently running cases.
            return await ScriptedModel([step]).get_response(*args, **kwargs)

    seen_agents = []

    class RecordingBackend(WorkerTestRunner):
        async def run(self, agent, *args, **kwargs):
            seen_agents.append(agent)
            return await super().run(agent, *args, **kwargs)

    config = options(
        StatelessModel([]),
        [identify],
        concurrency=3,
        cases=[
            {"id": str(i), "meta": {}, "items": [{"text": "a"}, {"text": "b"}]}
            for i in range(4)
        ],
        batch_budget={"unit": "chars", "limit": 13},
        _agent_runner=RecordingBackend(),
    )
    result = await run_cases(**config)
    assert not result["failed_cases"], result
    assert len(seen_agents) == 8
    assert all(agent is config["worker_agent"] for agent in seen_agents)
    for case in result["extracted_cases"]:
        assert case.output.text == case.id
        assert case.execution["passes"] == 2
        check_usage(case.execution, 6)
        for invocation in case.execution["tool_calls"]:
            if "pass" not in invocation:
                continue
            assert invocation["rounds"][0]["calls"][0]["output"] == case.id


@pytest.mark.asyncio
async def test_native_output_schema_and_guardrail_preserved_separately_from_extraction():
    from agents import GuardrailFunctionOutput, output_guardrail

    checked = []

    @output_guardrail
    async def validate_completion(context, agent, output):
        checked.append((agent, context.context.case_id, output))
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    config = options(ScriptedModel([
        [edit_call("extracted state", "edit")],
        [message('{"text":"native completion"}')],
    ]))
    worker = config["worker_agent"]
    worker.output_type = Output
    worker.output_guardrails = [validate_completion]
    result = await run_cases(**config)
    assert not result["failed_cases"], result
    assert result["extracted_cases"][0].output.text == "extracted state"
    assert len(checked) == 1
    assert checked[0][0] is worker
    assert checked[0][1:] == ("case", Output(text="native completion"))
    assert worker.output_guardrails == [validate_completion]


async def test_real_sdk_reviewer_queries_history_repairs_draft_and_edits_assessment():
    class Assessment(BaseModel):
        keep: bool

    reviewer = Agent(
        name="reviewer", tools=[query_case_sql, edit_state],
        model=ScriptedModel([
            [call("query_case_sql", {
                "query": "SELECT json_extract(state_json, '$.text') AS text FROM state_revisions"
            }, "history")],
            [call("edit_state", {
                "patch_json": '[{"op":"remove","path":"/text"}]',
            }, "invalid")],
            [call("edit_state", {
                "patch_json": '[{"op":"add","path":"/text","value":"corrected"}]',
                "edit_note": "Corrected using original evidence",
                "evidence": [{"artifact_position": 0, "json_pointer": "/text"}],
            }, "repair")],
            [review_call({"keep": False})],
            [message("Review complete.")],
        ]),
    )
    worker_model = ScriptedModel([[edit_call("worker", "worker-edit")], [message("done")]])
    result = await run_cases(**options(
        worker_model, reviewer_agent=reviewer, review_output_type=Assessment,
        should_keep=lambda record: record.review.keep,
    ))
    assert not result["failed_cases"], result
    record = result["filtered_cases"][0]
    assert record.output.text == "corrected"
    assert isinstance(record.review, Assessment) and not record.review.keep
    history = record.execution["revisions"]
    assert [r["state"]["text"] for r in history] == ["worker", "corrected"]
    assert [edit["target"] for edit in history[1]["edits"]] == ["case", "case", "review"]
    assert history[1]["review"] == {"keep": False}
    assert history[1]["edits"][1]["evidence"] == [
        {"artifact_position": 0, "json_pointer": "/text"}
    ]
    rounds = record.execution["tool_calls"][1]["rounds"]
    assert rounds[0]["calls"][0]["output"]["rows"] == [{"text": "worker"}]
    assert rounds[1]["calls"][0]["output"]["patch_applied"]
    assert not rounds[1]["calls"][0]["output"]["ok"]
    assert rounds[2]["calls"][0]["output"]["ok"]


@pytest.mark.asyncio
async def test_real_sdk_worker_sees_advisory_feedback_then_repairs_final_state():
    observed = []

    class WorkerModel(ScriptedModel):
        async def get_response(self, *args, **kwargs):
            tool_outputs = [
                item for item in kwargs["input"] if item.get("type") == "function_call_output"
            ]
            if tool_outputs:
                observed.append(literal_eval(tool_outputs[-1]["output"]))
            return await super().get_response(*args, **kwargs)

    model = WorkerModel([
        # A non-final pass can commit an incomplete draft.
        [call("edit_state", {"patch_json": "[]", "finish_pass": True}, "incomplete")],
        [message("First pass complete.")],
        # On the last batch an ordinary edit still provides advisory feedback.
        [call("edit_state", {
            "patch_json": '[{"op":"add","path":"/text","value":123}]',
        }, "wrong-type")],
        [call("edit_state", {"patch_json": "[]", "finish_pass": True}, "blocked")],
        [call("edit_state", {
            "patch_json": '[{"op":"replace","path":"/text","value":"repaired"}]',
            "finish_pass": True,
        }, "repaired")],
        [message("Final pass complete.")],
    ])
    result = await run_cases(**options(model, batch_budget={"unit": "chars", "limit": 13}))
    assert result["failed_cases"] == []
    case = result["extracted_cases"][0]
    assert case.output.text == "repaired"
    assert case.execution["passes"] == 2 and case.execution["attempts"] == 1
    assert case.execution["revisions"][0]["state"] == {}
    assert case.execution["revisions"][1]["state"] == {"text": "repaired"}
    assert len(observed) == 4
    assert [(item["ok"], item["state_valid"], item["pass_finished"]) for item in observed] == [
        (True, False, True), (True, False, False), (False, False, False), (True, True, True),
    ]
    assert observed[0]["validation_errors"][0]["type"] == "missing"
    assert observed[1]["validation_errors"][0]["type"] == "string_type"
    assert observed[2]["patch_applied"] is True
    assert observed[3]["validation_errors"] == []


@pytest.mark.asyncio
async def test_real_sdk_handoff_tool_carries_pass_records_and_rejects_reviewer_writes():
    class Assessment(BaseModel):
        keep: bool

    observed = []

    class NotesModel(ScriptedModel):
        async def get_response(self, *args, **kwargs):
            items = kwargs["input"]
            if len(items) == 1:
                content = items[0]["content"]
                payload = json.loads(content.split("\n", 1)[1])
                observed.append(payload["handoff_notes"])
                assert "handoff_notes" not in payload["target_output_schema"]["properties"]
            return await super().get_response(*args, **kwargs)

    worker = NotesModel([
        [call("write_handoff_note", {"note": "Check later evidence."}, "notes-first")],
        [edit_call("initial", "initial-edit")],
        [message("done")],
        [call("write_handoff_note", {"note": "Ask reviewer to verify."}, "notes-next")],
        [edit_call("final", "final-edit")],
        [message("done")],
    ])
    reviewer = Agent(
        name="reviewer", tools=[query_case_sql, edit_state, write_handoff_note],
        model=NotesModel([
            [call("query_case_sql", {
                "query": "SELECT handoff_notes_json FROM state_revisions ORDER BY revision_id",
            }, "notes-history")],
            [call("write_handoff_note", {"note": "Not allowed."}, "review-note")],
            [call("edit_state", {
                "patch_json": '[{"op":"replace","path":"/text","value":"reviewed"}]',
            }, "review-edit")],
            [review_call()],
            [message("Review complete.")],
        ]),
    )
    result = await run_cases(**options(
        worker, tools=[write_handoff_note], reviewer_agent=reviewer, review_output_type=Assessment,
        batch_budget={"unit": "chars", "limit": 13},
    ))
    assert not result["failed_cases"], result["failed_cases"]
    record = result["extracted_cases"][0]
    expected = [
        {"pass_number": 1, "artifact_range": {"start_position": 0, "end_position_exclusive": 1},
         "note": "Check later evidence."},
        {"pass_number": 2, "artifact_range": {"start_position": 1, "end_position_exclusive": 2},
         "note": "Ask reviewer to verify."},
    ]
    assert observed == [[], expected[:1], expected]
    assert record.output.text == "reviewed"
    assert record.execution["handoff_notes"] == expected
    history = record.execution["revisions"]
    assert [r["handoff_notes"] for r in history] == [
        expected[:1], expected, expected,
    ]
    outputs = record.execution["tool_calls"][-1]["rounds"]
    assert [json.loads(row["handoff_notes_json"])
            for row in outputs[0]["calls"][0]["output"]["rows"]] == [expected[:1], expected]
    assert outputs[1]["calls"][0]["output"] == {
        "ok": False, "error": "Handoff notes are read-only in this context.",
    }
