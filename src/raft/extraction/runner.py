from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Sequence

from pydantic import BaseModel, ValidationError

from raft._json import _id_key
from raft.cases import ExtractedCase
from raft.progress import CaseProgress
from raft.runtime import (
    IncompletePassError,
    PassLimitError,
    RetryDecision,
    _AgentScheduler,
    _failed_case,
    _retry_delay,
    map_concurrent,
    validate_limits,
)

from ._logging import quiet_response_errors
from .batching import next_batch
from .context import (
    DEFAULT_BATCH_CHARS,
    DEFAULT_QUERY_CHARS,
    ArtifactTooLargeError,
    CaseContext,
    _build_case_context,
    _validate_case,
)
from .coverage import _coverage_payload
from .prompts import _case_prompt, _review_prompt
from .telemetry import RunTelemetry

if TYPE_CHECKING:
    from agents import Agent, RunConfig

    from ._agent import _AgentRunner


@dataclass
class _Progress:
    state: Any = field(default_factory=dict)
    position: int = 0
    offset: int = 0
    passes: int = 0
    runs: list[RunTelemetry] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    validation_error: str | None = None


async def run_cases(
    *,
    cases: Sequence[dict[str, Any]],
    worker_agent: Agent[CaseContext],
    id_field: str,
    artifacts_field: str,
    metadata_field: str,
    output_type: type[BaseModel],
    reviewer_agent: Agent[CaseContext],
    should_keep: Callable[[ExtractedCase], bool] | None = None,
    artifact_sort_field: str | None = None,
    max_batch_chars: int = DEFAULT_BATCH_CHARS,
    max_query_chars: int = DEFAULT_QUERY_CHARS,
    concurrency: int = 4,
    agent_concurrency: int | None = None,
    timeout: float = 300.0,
    retries: int = 1,
    rpm: int = 30,
    max_passes: int = 100,
    max_turns: int = 20,
    run_config: RunConfig
    | Callable[[Agent, CaseContext], RunConfig | Awaitable[RunConfig]]
    | None = None,
    show_progress: bool = False,
    suppress_response_errors: bool = False,
) -> dict[str, Any]:
    """Extract every case through ordered preloaded batches and tool-based edits.

    Each pass receives committed state, runner-owned coverage, and the next batch.
    max_batch_chars bounds source JSON characters; max_query_chars separately
    bounds each complete serialized SQL-tool response. Neither bounds the full
    model context window.
    All artifacts must be supplied before completion. Every successful output
    is retained. Each successful pass saves a revision. The required reviewer can
    query history and correct the output after all worker passes. Its draft is
    committed only after a structured review and valid case state are returned.
    should_keep receives the corrected ExtractedCase; False retains all fields
    in filtered_cases, including execution.revisions.

    concurrency bounds active cases. agent_concurrency bounds active SDK runs
    across worker passes and reviews; None uses concurrency. Both it and rpm must
    permit a start before a queued run is admitted. Limits are per run_cases call.
    Agent slots are released between passes and before retry backoff.
    run_config is a native OpenAI RunConfig or a sync/async (agent, context) callback.
    It is resolved once per invocation; None disables tracing by default.
    timeout bounds one whole case attempt, including all passes and scheduler waits,
    after a worker picks it up. retries is a case-wide budget; a worker retry
    resumes the last committed state and coverage with a fresh timeout.
    rpm counts SDK runs (including worker passes, reviews, and retries), not model turns.
    Progress retries count scheduled RAFT case retries; rate_limited counts observed
    transient throttles, and waiting_retry counts case attempts currently backing off.
    Provider-internal retries are not visible. suppress_response_errors optionally
    hides repeated SDK response-error log lines only within this extraction call.
    Exceptions still propagate to RAFT and terminal failures retain their details.
    """
    validate_limits(concurrency, timeout, retries, rpm)
    if type(suppress_response_errors) is not bool:
        raise ValueError("suppress_response_errors must be a bool")
    if agent_concurrency is not None and (
        type(agent_concurrency) is not int or agent_concurrency < 1
    ):
        raise ValueError("agent_concurrency must be a positive integer or None")
    if type(max_batch_chars) is not int or max_batch_chars < 1:
        raise ValueError("max_batch_chars must be a positive integer")
    if type(max_query_chars) is not int or max_query_chars < 1:
        raise ValueError("max_query_chars must be a positive integer")
    if min(max_passes, max_turns) < 1:
        raise ValueError("pass/turn limits must be >= 1")
    from ._agent import _AgentRunner

    agent_runner = _AgentRunner(run_config=run_config)
    if not isinstance(output_type, type) or not issubclass(output_type, BaseModel):
        raise ValueError("output_type must be a Pydantic model class")
    agent = agent_runner.prepare(worker_agent)
    if reviewer_agent is None:
        raise ValueError("reviewer_agent is required")
    reviewer = agent_runner.prepare_reviewer(reviewer_agent)
    if should_keep is not None and not callable(should_keep):
        raise TypeError("should_keep must be callable")
    scheduler = _AgentScheduler(
        concurrency if agent_concurrency is None else agent_concurrency, rpm
    )
    seen: set[str] = set()
    work = []
    for case in cases:
        key = _id_key(case.get(id_field)) if isinstance(case, dict) else None
        work.append((case, key in seen))
        seen.add(key)

    async def process(entry: tuple[Any, bool]) -> tuple[str, ExtractedCase | dict[str, Any]]:
        case, duplicate = entry
        started = time.monotonic()
        case_id = case.get(id_field) if isinstance(case, dict) else None
        context: CaseContext | None = None
        progress = _Progress()
        attempts = 0
        output = None
        review = None
        stage = "worker"

        def execution():
            return {
                "usage": agent_runner.aggregate_usage([run.usage for run in progress.runs if run.usage]),
                "tool_calls": progress.tool_calls,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "passes": progress.passes,
                "attempts": attempts,
                "revisions": context.revisions() if context is not None else [],
            }

        def failure(exc: Exception, decision: RetryDecision, failure_type: str):
            item = _failed_case(
                case_id=case_id,
                case=case,
                category=decision.category,
                error=exc,
                retryable=decision.retryable,
                attempts=attempts,
                elapsed=time.monotonic() - started,
                failure_type=failure_type,
            )
            item.pop("attempts")
            item.pop("elapsed_seconds")
            item.update(partial_state=progress.state, execution=execution(), stage=stage)
            if output is not None:
                item.update(output=output.model_dump(mode="json"), review=review)
            if context is not None:
                item["coverage"] = _coverage_payload(context, progress.position, progress.offset)
            return "failed", item

        try:
            try:
                _validate_case(
                    case,
                    id_field=id_field,
                    artifacts_field=artifacts_field,
                    metadata_field=metadata_field,
                    duplicate=duplicate,
                )
                context = _build_case_context(
                    case,
                    id_field=id_field,
                    artifacts_field=artifacts_field,
                    metadata_field=metadata_field,
                    artifact_sort_field=artifact_sort_field,
                    max_query_chars=max_query_chars,
                    final_output_type=output_type,
                    max_batch_chars=max_batch_chars,
                )
            except ArtifactTooLargeError as exc:
                status, item = failure(exc, RetryDecision(False, "artifact_too_large"), "invalid_case")
                item["details"] = exc.details
                return status, item
            except (TypeError, ValueError) as exc:
                return failure(exc, RetryDecision(False, "invalid_case"), "invalid_case")

            for attempt in range(retries + 1):
                attempts = attempt + 1
                try:
                    async with asyncio.timeout(timeout):
                        if output is None:
                            output = await _work(
                                agent_runner,
                                agent,
                                context,
                                progress,
                                scheduler,
                                max_passes=max_passes,
                                max_turns=max_turns,
                                max_batch_chars=max_batch_chars,
                                attempt=attempts,
                            )
                        stage = "review"
                        worker_state = output.model_dump(mode="json")
                        review_prompt = _review_prompt(
                            case_id=case_id,
                            metadata=context.metadata,
                            output=worker_state,
                            target_schema=output_type.model_json_schema(),
                            worker_final_revision=progress.passes,
                            coverage=_coverage_payload(
                                context, progress.position, progress.offset
                            ),
                        )
                        review_context = context.for_review(worker_state)
                        try:
                            async with scheduler.slot():
                                telemetry = RunTelemetry()
                                progress.runs.append(telemetry)
                                progress.tool_calls.append({
                                    "attempt": attempts,
                                    "stage": "review",
                                    "rounds": telemetry.rounds,
                                })
                                candidate_review = await agent_runner.review(
                                    reviewer, review_prompt,
                                    context=review_context,
                                    max_turns=max_turns,
                                    telemetry=telemetry,
                                )
                            if not isinstance(candidate_review, (BaseModel, dict)):
                                raise ValueError("Reviewer must return a structured object")
                            candidate_output = output_type.model_validate(
                                review_context.pending_state, by_name=True
                            )
                            corrected_state = candidate_output.model_dump(mode="json")
                            if corrected_state != worker_state:
                                review_context.commit_revision(corrected_state, pass_number=None)
                            # No await between the journal commit and publishing the result.
                            progress.state = corrected_state
                            output, review = candidate_output, candidate_review
                        finally:
                            review_context.close()
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if isinstance(exc, IncompletePassError):
                        decision = RetryDecision(True, "model_behavior")
                    elif isinstance(exc, PassLimitError):
                        decision = RetryDecision(False, "max_case_passes")
                    else:
                        decision = agent_runner.classify_error(exc)
                    case_progress.observe_error(decision.category)
                    if not decision.retryable or attempt == retries:
                        return failure(
                            exc,
                            decision,
                            "retry_exhausted" if decision.retryable else "terminal_error",
                        )
                    with case_progress.retry_wait():
                        await asyncio.sleep(_retry_delay(decision, attempts))

            item = ExtractedCase[output_type](
                id=case_id,
                metadata=context.metadata,
                output=output,
                review=review,
                execution=execution(),
            )
            if should_keep is not None:
                stage = "filter"
                keep = should_keep(item)
                if type(keep) is not bool:
                    raise TypeError("should_keep must return a bool (True keeps, False filters)")
                if not keep:
                    return "filtered", item
            return "extracted", item
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Non-model errors must not abort the batch.
            return failure(exc, RetryDecision(False, "unexpected_error"), "terminal_error")
        finally:
            if context is not None:
                context.close()

    with (
        CaseProgress(len(work), enabled=show_progress, desc="Extraction") as case_progress,
        quiet_response_errors(suppress_response_errors),
    ):
        outcomes = await map_concurrent(
            work, process, concurrency, progress=case_progress,
            progress_status=lambda result: "succeeded" if result[0] == "extracted" else result[0],
        )
    statuses = (
        ("extracted", "failed", "filtered") if should_keep is not None else ("extracted", "failed")
    )
    result = {
        f"{status}_cases": [item for label, item in outcomes if label == status]
        for status in statuses
    }
    result["summary"] = {
        "total": len(cases),
        **{status: len(result[f"{status}_cases"]) for status in statuses},
    }
    return result


async def _work(
    agent_runner: _AgentRunner,
    agent: Any,
    context: CaseContext,
    progress: _Progress,
    scheduler: _AgentScheduler,
    *,
    max_passes: int,
    max_turns: int,
    max_batch_chars: int,
    attempt: int,
) -> BaseModel:
    while progress.passes < max_passes:
        batch = next_batch(context, progress.position, progress.offset, max_batch_chars)
        coverage = _coverage_payload(context, progress.position, progress.offset)
        prompt = _case_prompt(
            case_id=context.case_id,
            metadata=context.metadata,
            current_state=progress.state,
            coverage=coverage,
            batch=batch.payload(),
            target_schema=context.final_output_type.model_json_schema(),
            max_batch_chars=max_batch_chars,
            pass_number=progress.passes + 1,
            validation_error=progress.validation_error,
        )
        context.begin_pass(progress.state, is_final_batch=batch.is_last)
        async with scheduler.slot():
            await _invoke(agent_runner, agent, prompt, context, progress, attempt, max_turns)
        if not context.pass_finished:
            raise IncompletePassError("Worker ended without finishing the pass through edit_state")
        validated = None
        if batch.is_last:
            try:
                validated = context.final_output_type.model_validate(
                    context.pending_state, by_name=True
                )
            except ValidationError as exc:
                progress.validation_error = str(exc)
        committed_state = (
            validated.model_dump(mode="json")
            if validated is not None else deepcopy(context.pending_state)
        )
        context.commit_revision(committed_state, pass_number=progress.passes + 1)
        progress.state = committed_state
        progress.position, progress.offset = batch.next_position, batch.next_offset
        progress.passes += 1
        if validated is not None:
            return validated
    raise PassLimitError(f"Case exceeded {max_passes} worker passes")


async def _invoke(agent_runner, agent, prompt, context, progress, attempt, max_turns):
    telemetry = RunTelemetry()
    progress.runs.append(telemetry)
    progress.tool_calls.append(
        {"attempt": attempt, "pass": progress.passes + 1, "rounds": telemetry.rounds}
    )
    return await agent_runner.run(
        agent, prompt, context=context, max_turns=max_turns, telemetry=telemetry
    )
