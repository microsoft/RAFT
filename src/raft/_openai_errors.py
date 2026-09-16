"""OpenAI client errors shared by extraction and embedding."""

import openai

from raft._json import _snake_case
from raft.runtime import RetryDecision, _retry_after


def _openai_error_code(exc: Exception) -> str | None:
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    nested = body.get("error")
    if isinstance(nested, dict):
        return nested.get("code")
    code = body.get("code")
    return code if isinstance(code, str) else None


def classify_error(exc: Exception) -> RetryDecision:
    if isinstance(exc, openai.RateLimitError):
        if _openai_error_code(exc) in {
            "billing_hard_limit_reached",
            "insufficient_quota",
            "organization_spend_limit_exceeded",
            "organization_usage_limit_exceeded",
            "project_spend_limit_exceeded",
        }:
            return RetryDecision(False, "quota_or_billing")
        headers = getattr(getattr(exc, "response", None), "headers", None)
        return RetryDecision(True, "rate_limit", _retry_after(headers))
    if isinstance(exc, (TimeoutError, openai.APITimeoutError)):
        return RetryDecision(True, "timeout")
    if isinstance(exc, openai.APIConnectionError):
        return RetryDecision(True, "connection")
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        headers = getattr(getattr(exc, "response", None), "headers", None)
        return RetryDecision(
            status in {408, 409} or status >= 500, f"http_{status}", _retry_after(headers)
        )
    if isinstance(exc, (TypeError, ValueError)):
        return RetryDecision(False, _snake_case(type(exc).__name__))
    return RetryDecision(False, "unexpected_error")
