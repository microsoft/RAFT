from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any, Awaitable, Callable

from agents import Agent, RunConfig, Runner
from agents.exceptions import (
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ModelBehaviorError,
    ModelRefusalError,
    ModelTimeoutError,
    OutputGuardrailTripwireTriggered,
    ToolInputGuardrailTripwireTriggered,
    ToolOutputGuardrailTripwireTriggered,
    ToolTimeoutError,
    UserError,
)
from agents.models.interface import Model
from pydantic import ValidationError

from raft._json import _snake_case
from raft._openai_errors import classify_error as classify_client_error
from raft.extraction.context import CaseContext
from raft.extraction.telemetry import RunTelemetry
from raft.runtime import RetryDecision

from ._openai_telemetry import ToolCallHooks, UsageModelProvider, aggregate_usage


class _AgentRunner:
    """Reuse a caller-owned Agent with fresh input and hooks and case-local context.

    run_config optionally supplies a native RunConfig or a sync/async callback
    (agent, context) -> RunConfig. The callback runs once per invocation, including
    worker passes, reviews, and retries. None keeps tracing disabled by default.
    Caller settings are preserved; provider resolution is observed on a run-local config copy.
    """

    def __init__(
        self,
        *,
        run_config: RunConfig
        | Callable[[Agent, CaseContext], RunConfig | Awaitable[RunConfig]]
        | None = None,
    ) -> None:
        if run_config is not None and not isinstance(run_config, RunConfig) and not callable(run_config):
            raise TypeError("run_config must be a RunConfig, a callback, or None")
        self.run_config = run_config

    def prepare(self, agent: Agent) -> Agent:
        return _validate_agent(agent)

    def prepare_reviewer(self, agent: Agent) -> Agent:
        if not isinstance(agent, Agent):
            raise TypeError("reviewer_agent must be an initialized agents.Agent")
        required = {"query_case_sql", "edit_state"}
        missing = required - {getattr(tool, "name", None) for tool in agent.tools}
        if missing:
            raise ValueError(
                f"reviewer_agent.tools is missing required tools: {', '.join(sorted(missing))}. "
                "Attach the defaults from raft.tools or custom tools "
                "with these names that honor the CaseContext/review-correction contract."
            )
        if agent.output_type is None or agent.output_type is str:
            raise ValueError("Configure the reviewer's structured output_type on the Agent")
        return agent

    async def review(self, agent, prompt, **kwargs):
        return await self.run(agent, prompt, **kwargs)

    async def run(
        self,
        agent: Agent,
        prompt: str,
        *,
        context: CaseContext,
        max_turns: int,
        telemetry: RunTelemetry,
    ) -> Any:
        config = self.run_config
        if config is None:
            config = RunConfig(tracing_disabled=True)
        elif callable(config):
            config = config(agent, context)
            if inspect.isawaitable(config):
                config = await config
        if not isinstance(config, RunConfig):
            raise TypeError("run_config callback must return a RunConfig")
        hooks = ToolCallHooks(telemetry, config)
        if not isinstance(config.model, Model):
            # Observe the SDK's actual provider selection, including per-turn
            # routing and handoffs. Caller config/provider/model objects stay intact.
            config = replace(config, model_provider=UsageModelProvider(config.model_provider, hooks))
        result = await Runner.run(
            agent,
            prompt,
            context=context,
            max_turns=max_turns,
            hooks=hooks,
            run_config=config,
        )
        hooks.finish(result)
        return result.final_output

    def aggregate_usage(self, usages: list[dict[str, Any]]) -> dict[str, Any]:
        return aggregate_usage(usages)

    def classify_error(self, exc: Exception) -> RetryDecision:
        return classify_error(exc)


def _validate_agent(agent: Agent[CaseContext]) -> Agent[CaseContext]:
    """Check the worker contract without injecting or replacing caller-owned tools."""
    if not isinstance(agent, Agent):
        raise TypeError("worker_agent must be an initialized agents.Agent")
    required = {"query_case_sql", "edit_state"}
    missing = required - {getattr(tool, "name", None) for tool in agent.tools}
    if missing:
        raise ValueError(
            f"worker_agent.tools is missing required tools: {', '.join(sorted(missing))}. "
            "Attach the defaults from raft.tools or custom tools "
            "with these names that honor the CaseContext/pass-completion contract."
        )
    return agent


def classify_error(exc: Exception) -> RetryDecision:
    """Classify one failed case attempt. Unknown errors are terminal."""
    if isinstance(exc, (ModelTimeoutError, ToolTimeoutError)):
        return RetryDecision(True, "timeout")

    if isinstance(exc, ModelBehaviorError):
        return RetryDecision(True, "model_behavior")

    if isinstance(
        exc,
        (
            InputGuardrailTripwireTriggered,
            MaxTurnsExceeded,
            ModelRefusalError,
            OutputGuardrailTripwireTriggered,
            ToolInputGuardrailTripwireTriggered,
            ToolOutputGuardrailTripwireTriggered,
            UserError,
            ValidationError,
            TypeError,
            ValueError,
        ),
    ):
        return RetryDecision(False, _snake_case(type(exc).__name__))

    return classify_client_error(exc)
