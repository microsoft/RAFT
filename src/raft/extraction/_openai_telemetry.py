"""OpenAI usage and ordered tool calls, collected without trace processors."""

import json
from typing import Any

from agents import RunConfig, RunHooks
from agents.models.interface import Model, ModelProvider
from agents.usage import Usage
from pydantic import TypeAdapter

from raft.extraction.telemetry import RunTelemetry

_usage_adapter = TypeAdapter(Usage)
_json_adapter = TypeAdapter(Any)


def aggregate_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Usage] = {}
    for by_model in usages:
        for name, usage in by_model.items():
            totals.setdefault(name, Usage()).add(_usage_adapter.validate_python(usage))
    return {name: _usage_adapter.dump_python(usage, mode="json") for name, usage in totals.items()}


def _model_name(model: Model) -> str:
    # Native OpenAI models expose .model. Never use repr(model): it may contain
    # client configuration, and it does not identify the model used for billing.
    for attribute in ("model", "model_name"):
        name = getattr(model, attribute, None)
        if isinstance(name, str) and name:
            return name
    return "unknown"


class UsageModelProvider(ModelProvider):
    """Observe normal SDK resolution without resolving twice or wrapping models."""

    def __init__(self, provider: ModelProvider, hooks: "ToolCallHooks"):
        self.provider, self.hooks = provider, hooks

    def get_model(self, model_name):
        model = self.provider.get_model(model_name)
        self.hooks.resolved_model = _model_name(model)
        return model


class ToolCallHooks(RunHooks):
    def __init__(self, telemetry: RunTelemetry, config: RunConfig):
        self.telemetry = telemetry
        self.config = config
        self.usage: dict[str, Usage] = {}
        self.resolved_model = "unknown"
        self.current_model = "unknown"
        self.calls: dict[str, dict[str, Any]] = {}

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        selected = self.config.model if self.config.model is not None else agent.model
        self.current_model = _model_name(selected) if isinstance(selected, Model) else self.resolved_model
        self.telemetry.rounds.append(
            {"round": len(self.telemetry.rounds) + 1, "agent": agent.name,
             "model": self.current_model, "calls": []}
        )

    async def on_llm_end(self, context, agent, response):
        usage = self.usage.setdefault(self.current_model, Usage())
        usage.add(response.usage)
        self.telemetry.usage[self.current_model] = _usage_adapter.dump_python(usage, mode="json")
        for item in response.output:
            if not item.type.endswith("_call"):
                continue
            raw = item.model_dump(mode="json")
            args = raw.get("arguments", raw.get("input", raw.get("action", raw)))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    pass
            call_id = raw.get("call_id") or raw.get("id")
            call = {"call_id": call_id, "name": raw.get("name", item.type), "args": args}
            self.telemetry.rounds[-1]["calls"].append(call)
            self.calls[call_id] = call

    async def on_tool_end(self, context, agent, tool, result):
        call = self.calls.get(getattr(context, "tool_call_id", None))
        if call is not None:
            call["output"] = _json_adapter.dump_python(result, mode="json", fallback=str)

    def finish(self, result):
        # Also collect outputs that bypass function hooks (e.g. handoffs or native tools).
        for item in result.new_items:
            if item.type not in {"tool_call_output_item", "handoff_output_item"}:
                continue
            raw = item.raw_item
            call = self.calls.get(raw.get("call_id"))
            if call is not None:
                output = item.output if item.type == "tool_call_output_item" else raw.get("output")
                call["output"] = _json_adapter.dump_python(output, mode="json", fallback=str)
