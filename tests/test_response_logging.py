import asyncio
import logging

import pytest

from raft.extraction._logging import quiet_response_errors


@pytest.fixture
def sdk_logs(monkeypatch):
    """Capture raw records without formatting potentially sensitive exceptions."""
    logger = logging.getLogger("openai.agents")
    records = []

    class RecordHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    original_level = logger.level
    monkeypatch.setattr(logger, "handlers", [RecordHandler()])
    monkeypatch.setattr(logger, "filters", [])
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "disabled", False)
    logger.setLevel(logging.DEBUG)
    try:
        yield logger, records
    finally:
        logger.setLevel(original_level)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("message", ["Error getting response", "Error streaming response"])
@pytest.mark.parametrize("suffix", ["", " (request_id: req_123)", " (request_id: None)"])
@pytest.mark.parametrize("style", ["literal", "redacted", "unredacted"])
def test_response_errors_are_opt_in(sdk_logs, enabled, message, suffix, style):
    logger, records = sdk_logs
    message += suffix
    error = RuntimeError("detailed final failure")
    with quiet_response_errors(enabled):
        if style == "literal":
            logger.error(message)
        elif style == "redacted":
            logger.error("%s", message)
        else:
            logger.error("%s: %s", message, error, exc_info=(type(error), error, None))
    assert len(records) == (0 if enabled else 1)
    assert logger.filters == []

    logger.error("%s: %s", message, error, exc_info=(type(error), error, None))
    assert records[-1].getMessage() == f"{message}: detailed final failure"
    assert records[-1].exc_info[1] is error


def test_disabled_does_not_touch_logging(sdk_logs, monkeypatch):
    logger, records = sdk_logs
    handlers, filters = logger.handlers, logger.filters
    settings = logger.level, logger.propagate, logger.disabled

    def unexpected(*args, **kwargs):
        pytest.fail("A disabled scope must not configure or look up a logger")

    with monkeypatch.context() as patch:
        patch.setattr(logging, "getLogger", unexpected)
        patch.setattr(logger, "addFilter", unexpected)
        patch.setattr(logger, "removeFilter", unexpected)
        patch.setattr(logger, "setLevel", unexpected)
        with quiet_response_errors(False):
            logger.error("%s", "Error getting response")
    assert len(records) == 1
    assert logger.handlers is handlers and logger.filters is filters
    assert (logger.level, logger.propagate, logger.disabled) == settings
    assert filters == []


def test_caller_logging_configuration_is_preserved(sdk_logs):
    logger, records = sdk_logs
    caller_filter = logging.Filter()
    added_during_scope = logging.Filter()
    logger.addFilter(caller_filter)
    handlers, filters = logger.handlers, logger.filters
    settings = logger.level, logger.propagate, logger.disabled

    with quiet_response_errors(True):
        assert logger.handlers is handlers and logger.filters is filters
        assert (logger.level, logger.propagate, logger.disabled) == settings
        assert caller_filter in filters and len(filters) == 2
        logger.addFilter(added_during_scope)
        logger.error("Detailed terminal failure: %s", "retry limit exceeded")

    assert logger.handlers is handlers and logger.filters is filters
    assert (logger.level, logger.propagate, logger.disabled) == settings
    assert filters == [caller_filter, added_during_scope]
    assert records[0].getMessage() == "Detailed terminal failure: retry limit exceeded"


@pytest.mark.parametrize(
    ("template", "args"),
    [
        ("%s", ("Error getting response from a custom worker",)),
        ("%s", ("Error streaming response audio",)),
        ("%s", ("Error getting response (request_id: req_123) extra",)),
        ("%s", ("Error getting response\n",)),
        ("%s: %s", ("Unexpected error in output guardrails", RuntimeError("details"))),
        ("%s: %s", ("Error merging transcripts", RuntimeError("details"))),
        ("Error getting response: %s", (RuntimeError("custom failure"),)),
        ("custom: %s", ("Error getting response",)),
        ("%(message)s", ({"message": "Error getting response"},)),
    ],
)
def test_unrelated_errors_pass_unchanged(sdk_logs, template, args):
    logger, records = sdk_logs
    record = logger.makeRecord(logger.name, logging.ERROR, __file__, 0, template, args, None)
    original = record.__dict__.copy()
    with quiet_response_errors(True):
        logger.handle(record)
    assert records == [record]
    assert {key: record.__dict__[key] for key in original} == original


@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.CRITICAL])
def test_only_error_level_is_suppressed(sdk_logs, level):
    logger, records = sdk_logs
    with quiet_response_errors(True):
        logger.log(level, "%s", "Error getting response")
    assert len(records) == 1 and records[0].levelno == level


@pytest.mark.parametrize("name", ["openai.agents.custom", "openai", "raft.response_logging_test"])
def test_other_loggers_remain_visible(sdk_logs, monkeypatch, name):
    logger, records = sdk_logs
    other = logging.getLogger(name)
    monkeypatch.setattr(other, "handlers", logger.handlers)
    monkeypatch.setattr(other, "propagate", False)
    monkeypatch.setattr(other, "disabled", False)
    with quiet_response_errors(True):
        other.error("%s", "Error getting response")
        # Also verify the exact-name check if a caller routes another record here.
        logger.handle(other.makeRecord(name, logging.ERROR, __file__, 0, "%s",
                                       ("Error streaming response",), None))
    assert len(records) == 2
    assert all(record.name == name for record in records)


class UnrenderableError(Exception):
    def __str__(self):
        raise AssertionError("Exception text must not be inspected")

    def __repr__(self):
        raise AssertionError("Exception representation must not be inspected")

    def __bool__(self):
        raise AssertionError("Exception truthiness must not be inspected")


@pytest.mark.parametrize("redacted", [False, True])
def test_filter_never_formats_errors_or_inputs(sdk_logs, monkeypatch, redacted):
    logger, records = sdk_logs
    error = UnrenderableError()

    def unexpected(*args, **kwargs):
        pytest.fail("The filter must not render the log record")

    with monkeypatch.context() as patch:
        patch.setattr(logging.LogRecord, "getMessage", unexpected)
        with quiet_response_errors(True):
            args = ("Error getting response",) if redacted else ("Error getting response", error)
            logger.error(
                "%s" if redacted else "%s: %s",
                *args,
                exc_info=(type(error), error, None),
                extra={"model_input": error},
            )
    assert records == []


def test_non_string_payload_is_not_inspected(sdk_logs):
    logger, _ = sdk_logs
    error = UnrenderableError()
    records = [
        logger.makeRecord(logger.name, logging.ERROR, __file__, 0, "%s", (error,), None),
        logger.makeRecord(logger.name, logging.ERROR, __file__, 0, error, (), None),
    ]
    with quiet_response_errors(True):
        # Test before dispatch: caller formatters may legitimately render passed records.
        assert all(logger.filter(record) for record in records)
    assert records[0].args[0] is error
    assert records[1].msg is error


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_sdk_logging_helper(sdk_logs, enabled):
    sdk = pytest.importorskip("agents.logger")
    logger, records = sdk_logs
    error = RuntimeError("SDK failure detail")
    with quiet_response_errors(enabled):
        sdk.log_model_action_error(logger, "Error getting response", error)
        sdk.log_model_action_error(logger, "Error streaming response", error)
        sdk.log_model_action_error(logger, "Unexpected error in output guardrails", error)
    assert len(records) == (1 if enabled else 3)
    assert records[-1].args[0] == "Unexpected error in output guardrails"


def test_actual_sdk_helper_does_not_inspect_suppressed_exception(sdk_logs):
    sdk = pytest.importorskip("agents.logger")
    logger, records = sdk_logs
    with quiet_response_errors(True):
        sdk.log_model_action_error(logger, "Error getting response", UnrenderableError())
    assert records == []


def test_sdk_redacted_path_never_inspects_exception_or_diagnostics(sdk_logs):
    sdk = pytest.importorskip("agents.logger")
    logger, records = sdk_logs

    def unexpected():
        pytest.fail("Redacted diagnostics must not be inspected")

    with quiet_response_errors(True):
        sdk._log_action_error(
            logger,
            "Error streaming response",
            UnrenderableError(),
            redact=True,
            stacklevel=1,
            diagnostic_extra=unexpected,
        )
    assert records == []


async def test_child_tasks_inherit_quiet_but_parallel_task_does_not(sdk_logs):
    logger, records = sdk_logs
    entered = asyncio.Event()
    unquiet_finished = asyncio.Event()

    async def child():
        await asyncio.sleep(0)
        logger.error("%s", "Error streaming response")

    async def quiet_task():
        with quiet_response_errors(True):
            entered.set()
            await unquiet_finished.wait()
            await asyncio.create_task(child())
            logger.error("%s", "Error getting response")

    async def unquiet_task():
        await entered.wait()
        try:
            logger.error("%s", "Error getting response")
        finally:
            unquiet_finished.set()

    await asyncio.wait_for(asyncio.gather(quiet_task(), unquiet_task()), timeout=5)
    assert len(records) == 1
    assert records[0].args == ("Error getting response",)
    assert logger.filters == []


@pytest.mark.parametrize("first_exit", [0, 1])
async def test_overlapping_scopes_can_exit_in_either_order(sdk_logs, first_exit):
    logger, records = sdk_logs
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]

    async def quiet_task(index):
        with quiet_response_errors(True):
            entered[index].set()
            await release[index].wait()
            logger.error("%s", "Error getting response")
        logger.error("%s: %s", "Error getting response", f"outside {index}")

    tasks = [asyncio.create_task(quiet_task(index)) for index in range(2)]
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), timeout=5)
        assert len(logger.filters) == 1
        release[first_exit].set()
        await asyncio.wait_for(tasks[first_exit], timeout=5)
        assert len(logger.filters) == 1
        release[1 - first_exit].set()
        await asyncio.wait_for(tasks[1 - first_exit], timeout=5)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert logger.filters == []
    assert [record.args[1] for record in records] == [
        f"outside {first_exit}", f"outside {1 - first_exit}",
    ]


@pytest.mark.parametrize("inner_enabled", [False, True])
async def test_nested_scope_overrides_then_restores_outer_scope(sdk_logs, inner_enabled):
    logger, records = sdk_logs

    async def child():
        logger.error("%s", "Error getting response")

    with quiet_response_errors(True):
        logger.error("%s", "Error getting response")
        outer_filters = logger.filters.copy()
        with quiet_response_errors(inner_enabled):
            assert logger.filters == outer_filters
            await asyncio.create_task(child())
        logger.error("%s", "Error getting response")

    assert len(records) == (0 if inner_enabled else 1)
    assert logger.filters == []


@pytest.mark.parametrize("enabled", [False, True])
def test_exception_cleanup_preserves_exception_and_restores_logging(sdk_logs, enabled):
    logger, records = sdk_logs
    error = UnrenderableError()
    with pytest.raises(UnrenderableError) as caught:
        with quiet_response_errors(enabled):
            raise error
    assert caught.value is error
    assert logger.filters == []
    logger.error("%s", "Error getting response")
    assert len(records) == 1


async def test_cancellation_removes_filter_and_restores_task_context(sdk_logs):
    logger, records = sdk_logs
    entered = asyncio.Event()

    async def worker():
        try:
            with quiet_response_errors(True):
                entered.set()
                await asyncio.Event().wait()
        finally:
            logger.error("%s", "Error getting response")

    task = asyncio.create_task(worker())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert len(logger.filters) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert logger.filters == []
    assert len(records) == 1
