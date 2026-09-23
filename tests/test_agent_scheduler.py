import asyncio
import time
from collections import defaultdict
from types import SimpleNamespace

import pytest
from agent_helpers import finish_review, run_cases
from agents import Agent
from test_extraction import case, edit
from test_review import Review, options

from raft.extraction import _agent as sdk
from raft.extraction import runner
from raft.extraction.context import CaseContext
from raft.runtime import _AgentScheduler
from raft.tools import edit_state, query_case_sql


async def test_rpm_is_charged_at_start_after_capacity_becomes_available():
    scheduler = _AgentScheduler(1, 1, window_seconds=0.04)
    starts = []

    async def invoke(label):
        async with scheduler.slot():
            starts.append((label, time.monotonic()))

    async with asyncio.timeout(2):
        async with scheduler.slot():
            second = asyncio.create_task(invoke("second"))
            third = asyncio.create_task(invoke("third"))
            # Capacity stays occupied past two RPM windows. Queued calls must
            # neither start nor reserve an allowance that goes stale meanwhile.
            await asyncio.sleep(0.09)
            assert starts == []
        await asyncio.gather(second, third)

    assert [label for label, _ in starts] == ["second", "third"]
    assert starts[1][1] - starts[0][1] >= 0.035


async def test_active_cancellation_releases_capacity_and_queued_cancellation_is_free():
    scheduler = _AgentScheduler(1, 2)
    entered = asyncio.Event()
    tail_entered = asyncio.Event()

    async def active_run():
        async with scheduler.slot():
            entered.set()
            await asyncio.Event().wait()

    async def cancelled_waiter():
        async with scheduler.slot():
            pytest.fail("Cancelled waiter must never be admitted")

    async def tail_run():
        async with scheduler.slot():
            tail_entered.set()

    active = asyncio.create_task(active_run())
    await entered.wait()
    head = asyncio.create_task(cancelled_waiter())
    tail = asyncio.create_task(tail_run())
    try:
        await asyncio.sleep(0)
        head.cancel()
        with pytest.raises(asyncio.CancelledError):
            await head
        assert not tail_entered.is_set()
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        # Only two actual starts fit this minute; charging the cancelled waiter
        # or leaking the active slot would block the tail until this times out.
        await asyncio.wait_for(tail, timeout=1)
        assert tail_entered.is_set()
    finally:
        for task in (active, head, tail):
            task.cancel()
        await asyncio.gather(active, head, tail, return_exceptions=True)


async def test_failed_run_releases_capacity_but_keeps_its_rpm_charge():
    scheduler = _AgentScheduler(1, 1, window_seconds=0.04)
    start = time.monotonic()
    with pytest.raises(ValueError, match="failed"):
        async with scheduler.slot():
            raise ValueError("failed")
    async with asyncio.timeout(1):
        async with scheduler.slot():
            assert time.monotonic() - start >= 0.035


async def test_timed_out_rpm_queue_head_does_not_strand_followers():
    scheduler = _AgentScheduler(2, 1, window_seconds=0.04)
    async with scheduler.slot():
        start = time.monotonic()

    async def head():
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.01), scheduler.slot():
                pytest.fail("RPM allowance has not expired")

    async def tail():
        async with scheduler.slot():
            assert time.monotonic() - start >= 0.035

    await asyncio.wait_for(asyncio.gather(head(), tail()), timeout=1)


@pytest.mark.parametrize("agent_concurrency", [None, 1, 2, 10])
async def test_worker_passes_and_reviews_share_agent_budget(monkeypatch, agent_concurrency):
    active_cases = set()
    active_agents = set()
    maximum_cases = maximum_agents = 0
    all_cases_started = asyncio.Event()
    calls = defaultdict(list)
    build = runner._build_case_context
    close = CaseContext.close

    def build_context(*args, **kwargs):
        nonlocal maximum_cases
        context = build(*args, **kwargs)
        active_cases.add(context.case_id)
        maximum_cases = max(maximum_cases, len(active_cases))
        if len(active_cases) == 3:
            all_cases_started.set()
        return context

    def close_context(context):
        close(context)
        if context._owns_writer:
            active_cases.remove(context.case_id)

    reviewer = Agent(name="review", tools=[query_case_sql, edit_state])

    async def run(agent, prompt, *, context, **kwargs):
        nonlocal maximum_agents
        identifier = context.case_id
        assert identifier not in active_agents  # Each case remains sequential.
        active_agents.add(identifier)
        maximum_agents = max(maximum_agents, len(active_agents))
        calls[identifier].append(context.stage)
        try:
            await all_cases_started.wait()
            await asyncio.sleep(0.005)
            if agent is reviewer:
                assert calls[identifier] == ["worker", "worker", "reviewer"]
                value = finish_review(context, Review(keep=True, reason="done"))
            else:
                assert edit(context, [{
                    "op": "add", "path": "", "value": {
                        "extractable": True, "timeline": [identifier],
                    },
                }])["ok"]
                value = "done"
            return SimpleNamespace(new_items=[], final_output=value)
        finally:
            active_agents.remove(identifier)

    monkeypatch.setattr(runner, "_build_case_context", build_context)
    monkeypatch.setattr(CaseContext, "close", close_context)
    monkeypatch.setattr(sdk.Runner, "run", run)
    async with asyncio.timeout(3):
        result = await run_cases(
            cases=[case(str(i)) for i in range(5)],
            **options(
                reviewer_agent=reviewer, concurrency=3, agent_concurrency=agent_concurrency,
                batch_budget={"unit": "chars", "limit": 31},
            ),
        )
    assert not result["failed_cases"], result
    assert maximum_cases == 3
    assert maximum_agents == min(3, agent_concurrency or 3)
    assert not active_cases and not active_agents
    assert [item.id for item in result["extracted_cases"]] == [str(i) for i in range(5)]
    assert all(stages == ["worker", "worker", "reviewer"] for stages in calls.values())


@pytest.mark.parametrize("failed_stage", ["worker", "reviewer"])
async def test_retries_release_agent_slots_and_all_stages_count_toward_rpm(
    monkeypatch, failed_stage
):
    starts = []
    failed = False
    reviewer = Agent(name="review", tools=[query_case_sql, edit_state])

    async def run(agent, prompt, *, context, **kwargs):
        nonlocal failed
        starts.append((context.case_id, context.stage, time.monotonic()))
        await asyncio.sleep(0)
        if context.case_id == "a" and context.stage == failed_stage and not failed:
            failed = True
            raise TimeoutError("temporary")
        if agent is reviewer:
            value = finish_review(context, Review(keep=True, reason="done"))
        else:
            assert edit(context, [{
                "op": "add", "path": "", "value": {"extractable": True, "timeline": []},
            }])["ok"]
            value = "done"
        return SimpleNamespace(new_items=[], final_output=value)

    monkeypatch.setattr(sdk.Runner, "run", run)
    monkeypatch.setattr(runner, "_retry_delay", lambda *_: 0.1)
    monkeypatch.setattr(
        runner, "_AgentScheduler",
        lambda concurrency, rpm: _AgentScheduler(concurrency, rpm, window_seconds=0.02),
    )
    async with asyncio.timeout(3):
        result = await run_cases(
            cases=[case("a"), case("b")],
            **options(reviewer_agent=reviewer, concurrency=2, agent_concurrency=1, rpm=1, retries=1),
        )
    assert not result["failed_cases"], result
    calls = [(identifier, stage) for identifier, stage, _ in starts]
    attempts = [i for i, call in enumerate(calls) if call == ("a", failed_stage)]
    assert len(attempts) == 2
    assert attempts[0] < calls.index(("b", "reviewer")) < attempts[1]
    assert len(starts) == 5
    assert all(b[2] - a[2] >= 0.018 for a, b in zip(starts, starts[1:]))
    assert result["extracted_cases"][0].execution["attempts"] == 2


async def test_queue_timeout_does_not_record_an_invocation(monkeypatch):
    calls = []

    async def run(agent, prompt, *, context, **kwargs):
        calls.append(context.case_id)
        assert edit(context, [{
            "op": "add", "path": "", "value": {"extractable": True, "timeline": []},
        }])["ok"]
        return SimpleNamespace(new_items=[], final_output="done")

    reviewer = Agent(name="review", tools=[query_case_sql, edit_state])
    monkeypatch.setattr(sdk.Runner, "run", run)
    result = await run_cases(
        cases=[case("a"), case("b")],
        **options(
            reviewer_agent=reviewer,
            concurrency=2, agent_concurrency=1, rpm=1, timeout=0.02,
        ),
    )
    assert calls == ["a"]
    assert len(result["failed_cases"]) == 2
    for failure in result["failed_cases"]:
        assert failure["error_category"] == "timeout"
        groups = failure["execution"]["tool_calls"]
        assert len(groups) == (1 if failure["id"] == "a" else 0)
        assert all(group.get("stage") != "review" for group in groups)


@pytest.mark.parametrize("limit", [0, -1, True, False, 1.5, "2"])
async def test_invalid_agent_concurrency_is_rejected_before_agent_setup(limit):
    with pytest.raises(ValueError, match="agent_concurrency must be a positive integer or None"):
        await run_cases(cases=[], **options(worker_agent=None, agent_concurrency=limit))
