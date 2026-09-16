"""Usage and tool-call collection for one agent invocation, including interrupted runs."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunTelemetry:
    # Model name -> native Usage totals; no cross-model total.
    usage: dict[str, Any] = field(default_factory=dict)
    rounds: list[dict[str, Any]] = field(default_factory=list)
