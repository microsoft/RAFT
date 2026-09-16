import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from agent_helpers import FakeReview, run_cases
from pydantic import BaseModel, Field, RootModel

from raft import ExtractedCase, build_case_graph, embed_cases, load_cases
from raft.cases import restore_case
from raft.extraction.state import apply_edit
from raft.runtime import RetryDecision
from raft.storage import load_jsonl, save_json, save_jsonl


class State(BaseModel):
    entities: list[str]
    timestamp: datetime = Field(default_factory=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))


class Embeddings:
    name = "test"
    model = "test"

    async def embed(self, texts, *, input_type="document"):
        return [[1.0, 2.0] for _ in texts]

    def classify_error(self, exc):
        return RetryDecision(False, "error")


@pytest.mark.asyncio
async def test_extraction_keeps_every_model_and_generic_id():
    models = {name: State(entities=[name]) for name in ("kept", "filtered")}

    class AgentBackend(FakeReview):
        def prepare(self, agent):
            return agent

        async def run(self, agent, prompt, *, context, max_turns, telemetry):
            apply_edit(
                context=context,
                patch_json=json.dumps(
                    [
                        {
                            "op": "add",
                            "path": "",
                            "value": models[context.case_id].model_dump(mode="json"),
                        }
                    ]
                ),
                finish_pass=True,
            )
            return "ignored final text"

        def aggregate_usage(self, usages):
            return {}

        def classify_error(self, exc):
            return RetryDecision(False, "error")

    result = await run_cases(
        cases=[{"number": key, "info": {"x": 1}, "items": []} for key in models],
        id_field="number",
        metadata_field="info",
        artifacts_field="items",
        output_type=State,
        worker_agent=SimpleNamespace(),
        reviewer_agent=SimpleNamespace(),
        _agent_runner=AgentBackend(),
    )
    for case, identifier in zip(result["extracted_cases"], models, strict=True):
        assert case.id == identifier
        assert isinstance(case, ExtractedCase)
        assert isinstance(case.output, State) and case.output == models[identifier]
        assert case.metadata == {"x": 1}
        assert set(case.execution) == {
            "usage",
            "tool_calls",
            "elapsed_seconds",
            "passes",
            "attempts",
            "revisions",
        }


@pytest.mark.asyncio
async def test_embedding_graph_filters_reuse_original_cases_and_models(tmp_path):
    cases = [
        ExtractedCase(id=key, metadata={"product": "p"}, output=State(entities=["sso"]))
        for key in [1, "1"]
    ]
    seen_models, seen_cases = [], []

    def texts(state):
        seen_models.append(state)
        return state.entities

    def allow(source, candidate):
        seen_cases.extend([source, candidate])
        return bool(set(source.output.entities) & set(candidate.output.entities))

    embedded = await embed_cases(cases=cases, backend=Embeddings(), state_to_text=texts)
    assert all(item["case"] is cases[i] for i, item in enumerate(embedded["embedded_cases"]))
    assert all(model is cases[i].output for i, model in enumerate(seen_models))
    graph = await build_case_graph(
        cases=cases,
        backend=Embeddings(),
        case_to_text=lambda state: state.entities[0],
        neighbor_filter=allow,
    )
    from raft.storage import save_graph

    save_graph(graph, tmp_path)
    assert all(node is cases[i] for i, node in enumerate(graph["nodes"]))
    assert seen_cases and all(any(record is case for case in cases) for record in seen_cases)
    assert graph["edges"][0]["source"] != graph["edges"][0]["target"]
    assert all("output" not in row and "metadata" not in row for row in graph["embeddings"])
    assert load_jsonl(tmp_path / "nodes.jsonl") == [{"id": 1}, {"id": "1"}]
    assert cases[0].output.entities == ["sso"]


@pytest.mark.parametrize("root", [False, True])
def test_save_load_restores_models_without_mutating_live_cases(tmp_path, root):
    model_type = RootModel[list[str]] if root else State
    model = model_type(["one"]) if root else model_type(entities=["one"])
    case = ExtractedCase(id="a", metadata={"x": 1}, output=model)
    path = tmp_path / "nested/extraction.json"
    save_json(path, {"extracted_cases": [case]})
    assert case.output is model
    restored = load_cases(path, output_type=model_type)
    assert isinstance(restored[0].output, model_type)
    assert restored[0].output == model
    assert restore_case(case, model_type) is case
    save_json(tmp_path / "cases.json", [case])
    assert load_cases(tmp_path / "cases.json", output_type=model_type) == restored
    save_jsonl(tmp_path / "cases.jsonl", [case])
    assert load_jsonl(tmp_path / "cases.jsonl")[0]["output"] == model.model_dump(mode="json")


def test_serialized_case_restore_is_explicit_and_leaves_input_unchanged():
    raw = {"id": "a", "metadata": {}, "output": {"entities": ["sso"]}}
    with pytest.raises(ValueError, match="output_type"):
        restore_case(raw)
    restored = restore_case(raw, State)
    assert restored is not raw and isinstance(restored.output, State)
    assert isinstance(raw["output"], dict)


def test_coverage_removed_from_success_schema_and_old_saved_cases_still_load(tmp_path):
    raw = {
        "id": "old",
        "metadata": {},
        "output": {"entities": ["sso"]},
        "coverage": {"complete": True},
    }
    save_json(tmp_path / "old.json", [raw])
    case = load_cases(tmp_path / "old.json", output_type=State)[0]
    assert not hasattr(case, "coverage")
    assert "coverage" not in case.model_dump()
    assert "coverage" not in ExtractedCase[State].model_json_schema()["properties"]
    assert raw["coverage"] == {"complete": True}
