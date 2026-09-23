from __future__ import annotations

import asyncio
import email.utils
import random
import time
from collections import deque
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, Sequence, TypeVar

from .progress import CaseProgress

T = TypeVar("T")
R = TypeVar("R")


def validate_limits(concurrency: int, timeout: float, retries: int, rpm: int) -> None:
    if concurrency < 1 or rpm < 1:
        raise ValueError("concurrency and rpm must be at least 1")
    if timeout <= 0 or retries < 0:
        raise ValueError("timeout must be positive and retries cannot be negative")


async def map_concurrent(
    items: Sequence[T], operation: Callable[[T], Awaitable[R]], concurrency: int,
    *, show_progress: bool = False, progress_desc: str = "Processing",
    progress_status: Callable[[R], str] | None = None, progress_unit: str = "case",
    progress: CaseProgress | None = None,
) -> list[R]:
    """Use fixed workers in input order; an optional opened progress is caller-owned."""
    iterator = iter(enumerate(items))
    results: list[Any] = [None] * len(items)

    async def worker() -> None:
        for index, item in iterator:
            results[index] = await operation(item)
            active_progress.advance(
                progress_status(results[index]) if progress_status else "succeeded"
            )

    with (
        nullcontext(progress) if progress is not None else
        CaseProgress(len(items), enabled=show_progress, desc=progress_desc, unit=progress_unit)
    ) as active_progress:
        workers = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(items)))]
        try:
            # TaskGroup treats a child's cancellation as normal completion, which
            # would leave an unfilled result slot. gather propagates it to the caller.
            await asyncio.gather(*workers)
        finally:
            for task in workers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
    return results


class IncompletePassError(Exception):
    """An agent stopped without finishing its invocation through edit_state."""


class PassLimitError(Exception):
    """Extraction did not finish within the configured number of passes."""


@dataclass
class RetryDecision:
    retryable: bool
    category: str
    retry_after: float | None = None


class _RollingRateLimiter:
    def __init__(self, rpm: int, *, window_seconds: float = 60.0) -> None:
        self.rpm = rpm
        self.window_seconds = window_seconds
        self._starts: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                cutoff = now - self.window_seconds
                while self._starts and self._starts[0] <= cutoff:
                    self._starts.popleft()

                if len(self._starts) < self.rpm:
                    self._starts.append(now)
                    return

                wait_seconds = self.window_seconds - (now - self._starts[0])
            await asyncio.sleep(max(0.0, wait_seconds))


class _AgentScheduler:
    """FIFO admission with joint concurrency/RPM limits within one event loop."""

    def __init__(self, concurrency: int, rpm: int, *, window_seconds: float = 60.0) -> None:
        self.concurrency = concurrency
        self.rpm = rpm
        self.window_seconds = window_seconds
        self._active = 0
        self._starts: deque[float] = deque()
        self._waiters: deque[object] = deque()
        self._changed = asyncio.Event()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        ticket = object()
        self._waiters.append(ticket)
        admitted = False
        try:
            while True:
                now = time.monotonic()
                while self._starts and self._starts[0] <= now - self.window_seconds:
                    self._starts.popleft()

                first = self._waiters[0] is ticket
                available = self._active < self.concurrency
                if first and available and len(self._starts) < self.rpm:
                    # No await between checking both limits and recording the start.
                    self._waiters.popleft()
                    self._active += 1
                    self._starts.append(now)
                    admitted = True
                    self._changed.set()
                    break

                self._changed.clear()
                if first and available:
                    # Only the queue head needs a timer, and only if RPM is the
                    # remaining blocker. Otherwise a release/cancellation wakes it.
                    delay = self._starts[0] + self.window_seconds - now
                    try:
                        async with asyncio.timeout(max(0.0, delay)):
                            await self._changed.wait()
                    except TimeoutError:
                        pass
                else:
                    await self._changed.wait()
            yield
        finally:
            # Synchronous cleanup cannot itself be interrupted by cancellation.
            # Failed/cancelled runs retain their start in the rolling RPM window.
            if admitted:
                self._active -= 1
            else:
                self._waiters.remove(ticket)
            self._changed.set()


def _failed_case(
    *,
    case_id: Any,
    case: Any,
    category: str,
    error: Exception,
    retryable: bool,
    attempts: int,
    elapsed: float,
    failure_type: str = "invalid_case",
) -> dict[str, Any]:
    record = {
        "id": case_id,
        "case": case,
        "failure_type": failure_type,
        "error_type": type(error).__name__,
        "error_category": category,
        "error_message": str(error),
        "retryable": retryable,
        "attempts": attempts,
        "elapsed_seconds": round(elapsed, 3),
    }
    details = _error_details(error)
    if details:
        record["error_details"] = details
    return record


def _error_details(error: Exception) -> dict[str, Any]:
    """Select provider diagnostics without copying response bodies or arbitrary headers."""
    details = {}
    status = getattr(error, "status_code", None)
    if type(status) is int:
        details["status_code"] = status
    request_id = getattr(error, "request_id", None)
    if isinstance(request_id, str) and request_id:
        details["request_id"] = request_id
    code = getattr(error, "code", None)
    if not isinstance(code, str):
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            nested = body.get("error")
            code = nested.get("code") if isinstance(nested, dict) else body.get("code")
    if isinstance(code, str) and code:
        details["code"] = code
    headers = getattr(getattr(error, "response", None), "headers", None)
    retry_after = _retry_after(headers)
    if retry_after is not None:
        details["retry_after"] = retry_after
    return details


def _retry_delay(decision: RetryDecision, attempts: int) -> float:
    if decision.retry_after is not None:
        return max(0.0, decision.retry_after)
    return min(2**attempts + random.random(), 30.0)


def _retry_after(headers: Mapping[str, str] | None) -> float | None:
    if not headers:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        try:
            target = email.utils.parsedate_to_datetime(raw)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None
