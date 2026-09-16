import json

import pytest
from agent_helpers import FakeReview, run_cases
from pydantic import BaseModel

from raft._json import _to_json
from raft.extraction.batching import next_batch
from raft.extraction.context import _build_case_context
from raft.extraction.state import apply_edit
from raft.runtime import RetryDecision


class State(BaseModel):
    fragments: list[str]
    extractable: bool = True


def data():
    return {
        "id": "case",
        "metadata": {"product": "test"},
        "items": [
            {"seq": 2, "text": "later"},
            {"seq": 1, "text": '中文🙂\n"\\\u0000'},
            {"seq": 3, "notes": {"numbers": [1, 2, 3]}},
        ],
    }


def finish(context, state):
    return apply_edit(
        context=context,
        patch_json=json.dumps([{"op": "add", "path": "", "value": state}]),
        finish_pass=True,
    )


class Backend(FakeReview):
    def __init__(self, handler):
        self.handler = handler
        self.prompts = []

    def prepare(self, agent):
        return agent

    async def run(self, agent, prompt, *, context, max_turns, telemetry):
        payload = json.loads(prompt.split("Pass context:\n")[1])
        self.prompts.append(payload)
        self.handler(context, payload, len(self.prompts))
        return "done"

    def aggregate_usage(self, usages):
        return {}

    def classify_error(self, error):
        return RetryDecision(isinstance(error, TimeoutError), type(error).__name__)


async def extract(backend, *, case=None, **kwargs):
    return await run_cases(
        cases=[data() if case is None else case],
        worker_agent=object(),
        reviewer_agent=object(),
        _agent_runner=backend,
        output_type=State,
        id_field="id",
        artifacts_field="items",
        metadata_field="metadata",
        artifact_sort_field="seq",
        rpm=1000,
        **kwargs,
    )


@pytest.mark.parametrize("limit", [50, 100, 1000])
def test_whole_artifact_batches_are_ordered_bounded_and_retryable(limit):
    context = _build_case_context(
        data(),
        id_field="id",
        artifacts_field="items",
        metadata_field="metadata",
        artifact_sort_field="seq",
        final_output_type=State,
        max_query_chars=1,
    )
    try:
        position = offset = 0
        fragments = {i: [] for i in range(3)}
        expected = sorted(enumerate(data()["items"]), key=lambda pair: pair[1]["seq"])
        while position < 3:
            batch = next_batch(context, position, offset, limit)
            assert batch == next_batch(context, position, offset, limit)
            assert 0 < batch.source_chars <= limit
            assert batch.source_chars == sum(len(item["artifact_json"]) for item in batch.items)
            for item in batch.items:
                index = item["position"]
                before = len("".join(fragments[index]))
                assert item["start_char"] == before
                assert item["end_char_exclusive"] == before + len(item["artifact_json"])
                assert item["original_position"] == expected[index][0]
                fragments[index].append(item["artifact_json"])
            assert (batch.next_position, batch.next_offset) > (position, offset)
            position, offset = batch.next_position, batch.next_offset
        assert batch.is_last and offset == 0
        assert ["".join(fragments[i]) for i in range(3)] == [_to_json(item) for _, item in expected]
    finally:
        context.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1, 1.5, "10"])
async def test_invalid_batch_budget_fails_before_agent_run(limit):
    backend = Backend(lambda *args: pytest.fail("Agent should not run"))
    with pytest.raises(ValueError, match="max_batch_chars must be a positive integer"):
        await extract(backend, max_batch_chars=limit)


@pytest.mark.asyncio
async def test_complete_source_delivery_without_any_sql_calls():
    def handler(context, prompt, count):
        fragments = prompt["current_state"].get("fragments", [])
        fragments += [item["artifact_json"] for item in prompt["batch"]["items"]]
        assert finish(context, {"fragments": fragments})["ok"]

    backend = Backend(handler)
    result = await extract(backend, max_batch_chars=50, max_query_chars=1)
    assert not result["failed_cases"], result
    case = result["extracted_cases"][0]
    expected = "".join(_to_json(item) for item in sorted(data()["items"], key=lambda x: x["seq"]))
    assert "".join(case.output.fragments) == expected
    assert "coverage" not in case.model_dump()
    assert case.execution["passes"] == len(backend.prompts) > 1


@pytest.mark.asyncio
async def test_failed_batch_retries_exact_batch_without_advancing(monkeypatch):
    def handler(context, prompt, count):
        fragments = prompt["current_state"].get("fragments", [])
        fragments += [item["artifact_json"] for item in prompt["batch"]["items"]]
        assert finish(context, {"fragments": fragments})["ok"]
        if count == 2:
            raise TimeoutError("fail after finished edit")

    monkeypatch.setattr("raft.extraction.runner._retry_delay", lambda *args: 0)
    backend = Backend(handler)
    result = await extract(backend, max_batch_chars=50, retries=1)
    assert not result["failed_cases"], result
    assert backend.prompts[1] == backend.prompts[2]
    coverage = backend.prompts[2]["coverage"]
    assert coverage["covered_count"] == 1
    assert coverage["partial_artifact"] is None
    case = result["extracted_cases"][0]
    expected = "".join(_to_json(item) for item in sorted(data()["items"], key=lambda x: x["seq"]))
    assert "".join(case.output.fragments) == expected
    assert case.execution["attempts"] == 2


@pytest.mark.asyncio
async def test_sql_can_revisit_or_look_ahead_without_advancing_batch_coverage():
    def handler(context, prompt, count):
        rows = context.query("SELECT position, artifact_json FROM artifacts ORDER BY position")
        assert len(rows["rows"]) == 3 and "error" not in rows
        assert finish(context, {"fragments": []})["ok"]

    backend = Backend(handler)
    result = await extract(backend, max_batch_chars=50, max_query_chars=10000)
    assert not result["failed_cases"], result
    assert len(backend.prompts) == 3
    assert backend.prompts[1]["coverage"]["covered_count"] == 1
    assert "coverage" not in result["extracted_cases"][0].model_dump()


@pytest.mark.asyncio
async def test_ineligible_output_still_reads_all_batches_and_is_retained():
    def handler(context, prompt, count):
        assert finish(context, {"fragments": [], "extractable": False})["ok"]

    backend = Backend(handler)
    result = await extract(backend, max_batch_chars=50)
    assert result["summary"]["extracted"] == 1
    assert len(backend.prompts) > 1
    assert "coverage" not in result["extracted_cases"][0].model_dump()
    assert result["extracted_cases"][0].output.extractable is False


@pytest.mark.asyncio
async def test_pass_limit_returns_failure_with_whole_artifact_coverage():
    backend = Backend(lambda context, prompt, count: finish(context, {"fragments": []}))
    result = await extract(backend, max_batch_chars=50, max_passes=2)
    assert not result["extracted_cases"]
    failure = result["failed_cases"][0]
    assert failure["error_category"] == "max_case_passes"
    assert not failure["coverage"]["complete"]
    assert failure["coverage"]["covered_count"] == 2
    assert failure["coverage"]["partial_artifact"] is None


@pytest.mark.asyncio
async def test_missing_finish_does_not_commit_delivery():
    backend = Backend(lambda *args: None)
    result = await extract(backend, max_batch_chars=50, retries=0)
    failure = result["failed_cases"][0]
    assert failure["coverage"]["covered_count"] == 0
    assert failure["coverage"]["partial_artifact"] is None


@pytest.mark.asyncio
async def test_empty_case_still_produces_validated_output():
    backend = Backend(lambda context, prompt, count: finish(context, {"fragments": []}))
    result = await extract(backend, case={"id": "empty", "metadata": {}, "items": []})
    assert not result["failed_cases"], result
    assert backend.prompts[0]["batch"] == {"items": [], "source_chars": 0, "is_last": True}
    assert "coverage" not in result["extracted_cases"][0].model_dump()


@pytest.mark.asyncio
async def test_custom_editor_cannot_bypass_final_validation_and_gets_repair_pass():
    def handler(context, prompt, count):
        if count == 1:
            # Custom tools may bypass apply_edit, but not the runner's final validation.
            context.pending_state = {"fragments": "invalid"}
            context.pass_finished = True
        else:
            assert prompt["batch"]["items"] == [] and prompt["batch"]["is_last"]
            assert prompt["validation_error"] and prompt["coverage"]["complete"]
            assert finish(context, {"fragments": []})["ok"]

    result = await extract(Backend(handler))
    assert not result["failed_cases"], result
    assert result["extracted_cases"][0].execution["passes"] == 2


@pytest.mark.asyncio
async def test_oversized_last_artifact_fails_entire_case_without_agents_or_retry():
    class NoAgents(Backend):
        async def review(self, *args, **kwargs):
            pytest.fail("Oversized case must not reach review")

    raw = data()
    raw["items"][0]["text"] = "x" * 100
    backend = NoAgents(lambda *args: pytest.fail("Preflight must check ALL artifacts"))
    result = await extract(backend, case=raw, max_batch_chars=50, retries=3)
    assert not result["extracted_cases"]
    failure = result["failed_cases"][0]
    assert failure["error_category"] == "artifact_too_large"
    assert failure["retryable"] is False
    assert failure["execution"]["attempts"] == 0
    assert failure["execution"]["tool_calls"] == []
    assert failure["details"] == {
        "artifact_position": 1, "original_position": 0,
        "artifact_chars": len(_to_json(raw["items"][0])), "max_batch_chars": 50,
    }


@pytest.mark.asyncio
async def test_exact_character_boundary_is_accepted_and_other_cases_continue():
    item = {"text": "中文🙂"}
    limit = len(_to_json(item))
    assert len(_to_json(item).encode()) > limit
    backend = Backend(lambda context, *args: finish(context, {"fragments": []}))
    result = await run_cases(
        cases=[
            {"id": "oversized", "metadata": {}, "items": [{"text": "x" * 100}]},
            {"id": "boundary", "metadata": {}, "items": [item]},
        ],
        worker_agent=object(), reviewer_agent=object(), _agent_runner=backend, output_type=State,
        id_field="id", artifacts_field="items", metadata_field="metadata",
        max_batch_chars=limit, retries=2, rpm=1000,
    )
    assert [c.id for c in result["extracted_cases"]] == ["boundary"]
    assert [c["id"] for c in result["failed_cases"]] == ["oversized"]
    assert len(backend.prompts) == 1
