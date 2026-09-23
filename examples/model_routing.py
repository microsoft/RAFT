"""Static settings and per-invocation model selection through RAFT's run_config.

Set routing="static" to apply one model/settings override to both agents,
"weighted" for separate role-specific pools, or "round_robin" to rotate workers
and pin the reviewer. All three are wired below in index_with_model_pool.

A Model object carries its model ID, client, endpoint, and credentials.
RunConfig.model overrides Agent.model; non-None model_settings override the
agent's settings, while omitted settings are inherited. Compaction can be set
through ModelSettings.context_management for supported Responses endpoints.

The callback receives the actual Agent and its CaseContext. RAFT calls it once
per worker pass, review, or retry, and passes the returned RunConfig to the SDK.
Async callbacks are supported too. Selection stays fixed for that invocation's
model/tool turns. Return overrides instead of mutating a shared agent.

Weights express probabilities, not quotas, capacity limits, or automatic failover.
All selected models must support your tools and effective settings. Use
OpenAIChatCompletionsModel for endpoints without the Responses API.

index_with_model_pool is opt-in and makes billable calls. Set OPENAI_API_KEY,
PRIMARY_MODEL, SECONDARY_BASE_URL, SECONDARY_API_KEY, and SECONDARY_MODEL in
your environment before calling it. Import from the repository root; no clients
are created on import. Credentials are never stored in these examples.
"""

import os
import random
from collections.abc import Callable, Sequence
from itertools import cycle
from math import isfinite
from pathlib import Path
from typing import Any, Literal

from agents import Agent, ModelSettings, OpenAIResponsesModel, RunConfig
from agents.models.interface import Model
from openai import AsyncOpenAI

from examples.custom_agents import build_pipeline
from raft.embedding.openai import OpenAIEmbeddings
from raft.extraction.context import CaseContext


def round_robin_run_config(
    worker_models: Sequence[Model], *, reviewer_model: Model
) -> Callable[[Agent[CaseContext], CaseContext], RunConfig]:
    """Create one routing callback per extraction job; worker passes share the pool."""
    if not worker_models:
        raise ValueError("Supply at least one worker model")
    choices = cycle(worker_models)

    def configure(agent: Agent[CaseContext], context: CaseContext) -> RunConfig:
        model = reviewer_model if context.stage == "reviewer" else next(choices)
        return RunConfig(model=model, tracing_disabled=True)

    return configure


def weighted_run_config(
    *,
    worker_pool: Sequence[tuple[Model, ModelSettings, float]],
    reviewer_pool: Sequence[tuple[Model, ModelSettings, float]],
) -> Callable[[Agent[CaseContext], CaseContext], RunConfig]:
    """Choose (model, settings, weight) from the pool for the current role."""
    pools = {"worker": tuple(worker_pool), "reviewer": tuple(reviewer_pool)}
    for role, pool in pools.items():
        weights = [weight for _, _, weight in pool]
        if (
            not weights or any(not isfinite(weight) or weight < 0 for weight in weights)
            or not 0 < sum(weights) < float("inf")
        ):
            raise ValueError(f"{role} weights must be finite, nonnegative, and have a positive sum")

    def configure(agent: Agent[CaseContext], context: CaseContext) -> RunConfig:
        pool = pools[context.stage]
        model, settings, _ = random.choices(pool, weights=[item[2] for item in pool], k=1)[0]
        return RunConfig(model=model, model_settings=settings, tracing_disabled=True)

    return configure


async def index_with_model_pool(
    cases: list[dict[str, Any]], output_dir: str | Path,
    *, routing: Literal["static", "weighted", "round_robin"] = "weighted",
) -> dict[str, Any]:
    """Index caller-supplied id/metadata/artifacts records; close both clients afterward.

    The return value is the normal pipeline result, including per-case failures.
    Inspect result['extraction']['failed_cases'] and result['embedding']['failed_cases'].
    """
    if routing not in ("static", "weighted", "round_robin"):
        raise ValueError("routing must be static, weighted, or round_robin")
    async with (
        AsyncOpenAI(max_retries=0) as primary,
        AsyncOpenAI(
            base_url=os.environ["SECONDARY_BASE_URL"],
            api_key=os.environ["SECONDARY_API_KEY"],
            max_retries=0,
        ) as secondary,
    ):
        first = OpenAIResponsesModel(model=os.environ["PRIMARY_MODEL"], openai_client=primary)
        second = OpenAIResponsesModel(model=os.environ["SECONDARY_MODEL"], openai_client=secondary)
        fast = ModelSettings(reasoning={"effort": "low"}, max_tokens=8_000, store=False)
        thorough = ModelSettings(reasoning={"effort": "medium"}, max_tokens=16_000, store=False)
        # Optional, only when the selected model/endpoint supports Responses compaction:
        # fast.context_management = [{"type": "compaction", "compact_threshold": 80_000}]

        configurations = {
            "static": RunConfig(model=first, model_settings=fast, tracing_disabled=True),
            "weighted": weighted_run_config(
                worker_pool=[(first, fast, 70), (second, thorough, 30)],
                reviewer_pool=[(second, thorough, 100)],
            ),
            "round_robin": round_robin_run_config([first, second], reviewer_model=second),
        }
        pipeline = build_pipeline(
            output_dir,
            worker_model=first,
            reviewer_model=second,
            embedding_backend=OpenAIEmbeddings(client=primary, model="text-embedding-3-large"),
            run_config=configurations[routing],
        )
        return await pipeline.index(cases)
