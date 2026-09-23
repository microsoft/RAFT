import asyncio
from types import SimpleNamespace

import pytest
from agent_helpers import finish_review
from agents import Agent, RunConfig
from test_execution import ScriptedModel, check_usage, edit_call, message, review_call
from test_execution import options as _execution_options
from test_extraction import case, edit
from test_review import Review
from test_review import options as _options

from raft import run_cases
from raft.extraction import _agent as sdk
from raft.extraction import runner
from raft.tools import edit_state, query_case_sql


def options(**kwargs):
    values = _options(**kwargs)
    values.pop("_agent_runner")
    return values


def execution_options(*args, **kwargs):
    values = _execution_options(*args, **kwargs)
    values.pop("_agent_runner")
    return values


def finish_worker(context):
    assert edit(context, [{
        "op": "add", "path": "", "value": {"extractable": True, "timeline": []},
    }])["ok"]


@pytest.mark.parametrize("custom", [False, True])
async def test_default_and_static_config_cover_worker_and_reviewer(monkeypatch, custom):
    supplied = RunConfig(
        model="configured-model", tracing_disabled=False,
        trace_include_sensitive_data=False, workflow_name="custom-workflow",
    ) if custom else None
    observed = []
    reviewer = Agent(name="review", tools=[query_case_sql, edit_state])

    async def run(agent, prompt, *, context, max_turns, hooks, run_config):
        observed.append(run_config)
        assert max_turns == 7 and hooks is not None
        assert context.case_id == "a"
        if custom:
            assert run_config.model_provider.provider is supplied.model_provider
            assert not run_config.tracing_disabled
            assert not run_config.trace_include_sensitive_data
            assert run_config.model == "configured-model"
            assert run_config.workflow_name == "custom-workflow"
        else:
            assert run_config.tracing_disabled and run_config.model is None
        if agent is reviewer:
            value = finish_review(context, Review(keep=True, reason="done"))
        else:
            finish_worker(context)
            value = "done"
        return SimpleNamespace(new_items=[], final_output=value)

    monkeypatch.setattr(sdk.Runner, "run", run)
    result = await run_cases(
        cases=[case()], **options(
            run_config=supplied,
            reviewer_agent=reviewer, max_turns=7,
        ),
    )
    assert not result["failed_cases"], result
    assert len(observed) == 2
    assert result["extracted_cases"][0].review.keep


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_callback_selects_fresh_config_for_passes_review_and_retry(monkeypatch, asynchronous):
    selected = []
    observed = []
    reviewer = Agent(name="review", tools=[query_case_sql, edit_state])
    worker = options()["worker_agent"]

    def configure(agent, context):
        assert agent is (reviewer if context.stage == "reviewer" else worker)
        assert context.metadata == {"product": "test"}
        config = RunConfig(model=f"route-{len(selected)}", tracing_disabled=True)
        selected.append((context.stage, config))
        return config

    async def aconfigure(agent, context):
        await asyncio.sleep(0)
        return configure(agent, context)

    async def run(agent, prompt, *, context, run_config, **kwargs):
        observed.append(run_config)
        if agent is reviewer:
            if len(observed) == 3:
                raise TimeoutError("retry this review")
            value = finish_review(context, Review(keep=True, reason="done"))
        else:
            finish_worker(context)
            value = "done"
        return SimpleNamespace(new_items=[], final_output=value)

    monkeypatch.setattr(sdk.Runner, "run", run)
    monkeypatch.setattr(runner, "_retry_delay", lambda *_: 0)
    result = await run_cases(
        cases=[case()], **options(
            run_config=aconfigure if asynchronous else configure,
            worker_agent=worker, reviewer_agent=reviewer, batch_budget={"unit": "chars", "limit": 31}, retries=1,
        ),
    )
    assert not result["failed_cases"], result
    assert [stage for stage, _ in selected] == ["worker", "worker", "reviewer", "reviewer"]
    assert [config.model for config in observed] == [f"route-{i}" for i in range(4)]
    assert all(actual.model == chosen.model and actual.model_provider.provider is chosen.model_provider
               for actual, (_, chosen) in zip(observed, selected, strict=True))
    assert worker.model is None and reviewer.model is None


async def test_real_sdk_concurrent_routing_keeps_choice_across_model_turns():
    class YieldingModel(ScriptedModel):
        async def get_response(self, *args, **kwargs):
            await asyncio.sleep(0)
            return await super().get_response(*args, **kwargs)

    models = {
        str(i): YieldingModel([[edit_call(str(i), "edit")], [message("done")]])
        for i in range(4)
    }
    review_models = {
        str(i): YieldingModel([[review_call()], [message("Review complete.")]]) for i in range(4)
    }
    selected = []

    async def configure(agent, context):
        selected.append((context.case_id, context.stage))
        await asyncio.sleep(0)
        routes = models if context.stage == "worker" else review_models
        return RunConfig(model=routes[context.case_id], tracing_disabled=True)

    # Exhausted if called: the override must win over the shared agent's model.
    unused_model = ScriptedModel([])
    settings = execution_options(
        unused_model, run_config=configure,
        cases=[{"id": str(i), "meta": {}, "items": []} for i in range(4)],
        concurrency=4,
    )
    result = await run_cases(**settings)
    assert not result["failed_cases"], result
    assert sorted(selected) == sorted((str(i), stage) for i in range(4)
                                       for stage in ("worker", "reviewer"))
    assert [item.output.text for item in result["extracted_cases"]] == [str(i) for i in range(4)]
    assert settings["worker_agent"].model is unused_model
    for item in result["extracted_cases"]:
        check_usage(item.execution, 4)
        assert item.execution["tool_calls"][0]["rounds"][0]["calls"][0]["name"] == "edit_state"


@pytest.mark.parametrize("value", [{}, "model-name", False])
async def test_invalid_run_config_rejected_before_processing(value):
    with pytest.raises(TypeError, match="run_config must be a RunConfig"):
        await run_cases(cases=[], **options(run_config=value))


@pytest.mark.parametrize("value", [None, {}, "model-name"])
async def test_invalid_callback_result_fails_before_sdk_invocation(monkeypatch, value):
    async def unexpected(*args, **kwargs):
        pytest.fail("Invalid run configuration must not reach the SDK")

    monkeypatch.setattr(sdk.Runner, "run", unexpected)
    result = await run_cases(
        cases=[case()],
        **options(run_config=lambda *_: value),
    )
    assert not result["extracted_cases"]
    failure = result["failed_cases"][0]
    assert failure["error_type"] == "TypeError"
    assert failure["error_message"] == "run_config callback must return a RunConfig"


async def test_pipeline_passes_native_run_config_to_both_agents(tmp_path):
    from test_local_pipeline import Embeddings

    from raft import LocalPipeline

    model = ScriptedModel([
        [edit_call("extracted", "edit")], [message("done")],
        [review_call()], [message("Review complete.")],
    ])
    supplied = RunConfig(model=model, tracing_disabled=True)
    settings = execution_options(ScriptedModel([]), run_config=supplied)
    cases = settings.pop("cases")
    pipeline = LocalPipeline(
        tmp_path,
        extraction=settings,
        embedding={"backend": Embeddings(), "state_to_text": lambda output: [output.text]},
        bm25=False,
    )
    assert pipeline.extraction["run_config"] is supplied
    result = await pipeline.index(cases)
    case = result["indexed_cases"][0]["case"]
    assert case.output.text == "extracted"
    assert case.review.keep
    # Both worker and reviewer tool/final turns use the supplied model override.
    check_usage(case.execution, 4)


async def test_removed_backend_parameter_is_rejected():
    with pytest.raises(TypeError, match="backend"):
        await run_cases(cases=[], backend=object(), **options())
