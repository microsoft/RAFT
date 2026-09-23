"""Explicit deterministic reviewers for tests focused on worker behavior."""

import json
from unittest.mock import patch

from agents import Agent
from pydantic import BaseModel

from raft import LocalPipeline as _LocalPipeline
from raft import run_cases as _run_cases
from raft.extraction._agent import _AgentRunner
from raft.extraction.state import apply_edit
from raft.tools import edit_state, query_case_sql


class ReviewResult(BaseModel):
    keep: bool


REVIEWER = Agent(name="test-reviewer", tools=[query_case_sql, edit_state])


def finish_review(context, value):
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    result = apply_edit(
        context=context, target="review",
        patch_json=json.dumps([{"op": "add", "path": "", "value": value}]),
        finish_pass=True,
    )
    assert result["ok"], result
    return "Review complete."


class WorkerTestRunner(_AgentRunner):
    async def review(self, agent, prompt, *, context, **kwargs):
        if agent is REVIEWER:
            assert context.stage == "reviewer"
            return finish_review(context, ReviewResult(keep=True))
        return await super().review(agent, prompt, context=context, **kwargs)


class FakeReview:
    def prepare_reviewer(self, agent):
        return agent

    async def review(self, agent, prompt, *, context, **kwargs):
        assert context.stage == "reviewer"
        return finish_review(context, ReviewResult(keep=True))


# Unit tests replace private SDK execution, not the public extraction API.
# The real-SDK integration tests below still drive Runner.run with scripted models.


async def run_cases(*, _agent_runner=None, **kwargs):
    kwargs.setdefault("review_output_type", ReviewResult)
    if _agent_runner is None:
        return await _run_cases(**kwargs)
    with patch("raft.extraction._agent._AgentRunner", return_value=_agent_runner):
        return await _run_cases(**kwargs)


class LocalPipeline(_LocalPipeline):
    def __init__(self, *args, extraction, **kwargs):
        extraction = dict(extraction)
        extraction.setdefault("review_output_type", ReviewResult)
        self._test_runner = extraction.pop("_agent_runner", None)
        super().__init__(*args, extraction=extraction, **kwargs)

    async def index(self, *args, **kwargs):
        if self._test_runner is None:
            return await super().index(*args, **kwargs)
        with patch("raft.extraction._agent._AgentRunner", return_value=self._test_runner):
            return await super().index(*args, **kwargs)
