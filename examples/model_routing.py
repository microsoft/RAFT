"""Use multiple models/endpoints with the SAME agent via SDK RunConfig.

RunConfig(model=...) overrides Agent.model for one complete SDK run, including
its tool turns. A Model object carries its own client/endpoint. RAFT accepts a
run_config(agent, context) callback, invoked for each worker pass, review, and
retry; it can also be async. Never mutate a shared Agent.model during a run.
Passing a static RunConfig(model=...) instead applies the same override to BOTH
roles. A callback can select by context.stage for separate worker/reviewer pools.

This example round-robins worker invocations and pins reviews to one model.
It is not automatic failover or per-endpoint rate limiting; RAFT's limits still
apply across the extraction call. All selected endpoints must support tool
calling; native structured final outputs are not required. Use
OpenAIChatCompletionsModel if an endpoint supports Chat Completions but not Responses.

index_with_model_pool is opt-in and makes billable calls. Set OPENAI_API_KEY,
PRIMARY_MODEL, SECONDARY_BASE_URL, SECONDARY_API_KEY, and SECONDARY_MODEL in
your environment before calling it. Import from the repository root; no clients
are created on import. Credentials are never stored in these examples.
"""

import os
from collections.abc import Callable, Sequence
from itertools import cycle
from pathlib import Path
from typing import Any

from agents import Agent, OpenAIResponsesModel, RunConfig
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


async def index_with_model_pool(
    cases: list[dict[str, Any]], output_dir: str | Path
) -> dict[str, Any]:
    """Index caller-supplied id/metadata/artifacts records; close both clients afterward.

    The return value is the normal pipeline result, including per-case failures.
    Inspect result['extraction']['failed_cases'] and result['embedding']['failed_cases'].
    """
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
        pipeline = build_pipeline(
            output_dir,
            worker_model=first,
            reviewer_model=second,
            embedding_backend=OpenAIEmbeddings(client=primary, model="text-embedding-3-large"),
            run_config=round_robin_run_config([first, second], reviewer_model=second),
        )
        return await pipeline.index(cases)
