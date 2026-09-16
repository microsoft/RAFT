"""Task-scoped suppression of the Agents SDK's generic response error logs."""

import logging
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_LOGGER_NAME = "openai.agents"
_RESPONSE_ERROR = re.compile(
    r"Error (?:getting|streaming) response(?: \(request_id: [^()\r\n]+\))?"
)
_quiet = ContextVar("raft_quiet_response_errors", default=False)
_filter_lock = threading.Lock()
_active_scopes = 0


class _ResponseErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not _quiet.get() or record.name != _LOGGER_NAME or record.levelno != logging.ERROR:
            return True
        if type(record.msg) is not str or type(record.args) is not tuple:
            return True
        if not record.args:
            message = record.msg
        elif (record.msg == "%s" and len(record.args) == 1) or (
            record.msg == "%s: %s" and len(record.args) == 2
        ):
            message = record.args[0]
        else:
            return True
        # Match the SDK's template argument, never format its exception or payload.
        return type(message) is not str or _RESPONSE_ERROR.fullmatch(message) is None


_response_error_filter = _ResponseErrorFilter()


@contextmanager
def quiet_response_errors(enabled: bool) -> Iterator[None]:
    """Hide generic SDK response ERROR records in this context and its child tasks.

    No exception is caught or changed, and other logs remain visible. A disabled
    scope makes no logging changes and temporarily overrides an enclosing quiet
    scope. Enabled scopes share one temporary filter, removed when the last scope
    exits; child tasks should therefore be awaited before leaving their scope.
    Caller handlers, levels, filters, and propagation settings are left intact.
    """
    global _active_scopes

    token = _quiet.set(enabled)
    try:
        if not enabled:
            yield
            return

        logger = logging.getLogger(_LOGGER_NAME)
        with _filter_lock:
            if _active_scopes == 0:
                logger.addFilter(_response_error_filter)
            _active_scopes += 1
        try:
            yield
        finally:
            with _filter_lock:
                _active_scopes -= 1
                if _active_scopes == 0:
                    logger.removeFilter(_response_error_filter)
    finally:
        _quiet.reset(token)
