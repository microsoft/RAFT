"""Per-case, per-model totals through real SDK routing, handoffs, and retries."""

import json

from agents import Agent, RunConfig
from agents.models.interface import ModelProvider
from test_execution import ScriptedModel, call, edit_call, message
from test_openai_run_config import execution_options

from raft import run_cases
from raft.storage import save_json
from raft.tools import edit_state, query_case_sql


class NamedModel(ScriptedModel):
    def __init__(self, name, steps):
        super().__init__(steps)
        self.model = name


def assert_usage(record, expected):
    usage = record.execution["usage"] if hasattr(record, "execution") else record["execution"]["usage"]
    assert set(usage) == set(expected)
    for name, requests in expected.items():
        assert usage[name]["requests"] == requests
        assert usage[name]["input_tokens"] == 10 * requests
        assert usage[name]["output_tokens"] == 4 * requests
        assert usage[name]["total_tokens"] == 14 * requests
        assert usage[name]["input_tokens_details"]["cached_tokens"] == 2 * requests
        assert usage[name]["output_tokens_details"]["reasoning_tokens"] == requests
        assert len(usage[name]["request_usage_entries"]) == requests


async def test_five_passes_and_review_aggregate_only_by_model(tmp_path):
    def steps(passes):
        return [step for i in passes for step in ([edit_call(str(i), f"edit-{i}")], [message("done")])]

    a = NamedModel("model-a", steps([0, 2, 4]))
    b = NamedModel("model-b", [*steps([1, 3]), [message('{"keep":false}')]])
    selections = iter([a, b, a, b, a, b])
    result = await run_cases(**execution_options(
        ScriptedModel([]),
        cases=[{"id": "case", "meta": {}, "items": [{"text": str(i)} for i in range(5)]}],
        max_batch_chars=13, run_config=lambda *_: RunConfig(model=next(selections), tracing_disabled=True),
        should_keep=lambda c: c.review.keep,
    ))
    assert not result["failed_cases"], result
    record = result["filtered_cases"][0]
    assert record.execution["passes"] == 5
    assert_usage(record, {"model-a": 6, "model-b": 5})
    save_json(tmp_path / "result.json", result)
    saved = json.loads((tmp_path / "result.json").read_text())["filtered_cases"][0]
    assert_usage(saved, {"model-a": 6, "model-b": 5})


async def test_custom_provider_and_handoff_use_resolved_model_not_route_alias():
    class Provider(ModelProvider):
        def __init__(self):
            self.calls = []
            self.models = {
                "worker-route": NamedModel("worker-model", [[call("transfer_to_specialist", {}, "handoff")]]),
                "specialist-route": NamedModel("specialist-model", [[edit_call("done", "edit")], [message("done")]]),
                "review-route": NamedModel("review-model", [[message('{"keep":true}')]]),
            }

        def get_model(self, name):
            self.calls.append(name)
            return self.models[name]

    provider = Provider()
    settings = execution_options(None)
    specialist = Agent(name="specialist", model="specialist-route", tools=[query_case_sql, edit_state])
    worker = settings["worker_agent"]
    worker.model = "worker-route"
    worker.handoffs = [specialist]
    # Clone the test reviewer before setting its route; never mutate a shared fixture.
    settings["reviewer_agent"] = settings["reviewer_agent"].clone(model="review-route")
    config = RunConfig(model_provider=provider, tracing_disabled=True)
    result = await run_cases(**settings, run_config=config)
    assert not result["failed_cases"], result
    record = result["extracted_cases"][0]
    assert_usage(record, {"worker-model": 1, "specialist-model": 2, "review-model": 1})
    assert config.model_provider is provider
    assert worker.model == "worker-route"
    assert set(provider.calls) == {"worker-route", "specialist-route", "review-route"}


async def test_failed_attempt_usage_is_kept_under_original_model(monkeypatch):
    monkeypatch.setattr("raft.extraction.runner._retry_delay", lambda *_: 0)
    first = NamedModel("first", [[edit_call("discard", "first-edit")], TimeoutError("retry")])
    second = NamedModel("second", [[edit_call("done", "second-edit")], [message("done")], [message('{"keep":true}')]])
    selected = iter([first, second, second])
    result = await run_cases(**execution_options(
        None, retries=1, run_config=lambda *_: RunConfig(model=next(selected), tracing_disabled=True),
    ))
    assert not result["failed_cases"], result
    assert_usage(result["extracted_cases"][0], {"first": 1, "second": 3})


async def test_failed_case_retains_per_model_usage():
    model = NamedModel("failed-model", [[edit_call("draft", "edit")], ValueError("terminal")])
    result = await run_cases(**execution_options(model))
    assert len(result["failed_cases"]) == 1
    assert_usage(result["failed_cases"][0], {"failed-model": 1})


async def test_concurrent_cases_do_not_mix_model_usage():
    import asyncio

    class YieldingModel(NamedModel):
        async def get_response(self, *args, **kwargs):
            await asyncio.sleep(0)
            return await super().get_response(*args, **kwargs)

    models = {
        str(i): YieldingModel(f"model-{i}", [
            [edit_call(str(i), f"edit-{i}")], [message("done")], [message('{"keep":true}')],
        ]) for i in range(4)
    }
    result = await run_cases(**execution_options(
        None, cases=[{"id": str(i), "meta": {}, "items": []} for i in range(4)],
        concurrency=4,
        run_config=lambda _, context: RunConfig(model=models[context.case_id], tracing_disabled=True),
    ))
    assert not result["failed_cases"], result
    for case in result["extracted_cases"]:
        assert_usage(case, {f"model-{case.id}": 3})
