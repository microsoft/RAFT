"""Offline checks for the connected customization examples and SDK model routing."""

import json
from types import SimpleNamespace

import pytest
from agents import Agent
from agents.tool_context import ToolContext
from pydantic import ValidationError
from test_execution import ScriptedModel, call, message, review_call
from test_local_pipeline import Embeddings

from examples import model_routing
from examples.custom_agents import build_pipeline, lookup_error
from examples.custom_extraction import (
    REVIEWER_PROMPT,
    WORKER_PROMPT,
    SupportCase,
    SupportEntity,
)
from examples.custom_text import case_to_text, format_case, state_to_text
from examples.model_routing import round_robin_run_config
from raft import ExtractedCase
from raft.defaults import REVIEWER_INSTRUCTIONS, WORKER_INSTRUCTIONS, CaseExtraction, CaseReview
from raft.retrieval.types import RetrievalHit


@pytest.fixture
def state():
    return SupportCase(
        entities=[
            SupportEntity(name="login-service", kind="component"),
            SupportEntity(name="AUTH_CERT_EXPIRED", kind="error_code"),
        ],
        timeline=[
            (
                "The login-service returned AUTH_CERT_EXPIRED during authentication requests. "
                "The support engineer requested the configured certificate's validity period "
                "and the system clock reading to distinguish an expired certificate from clock "
                "skew. Neither explanation had yet been confirmed by the available evidence."
            ),
            (
                "The login-service certificate inspection showed an expired validity period, "
                "while the system clock matched the time source. The engineer rotated the "
                "certificate and repeated the failed login request. Authentication succeeded "
                "after rotation, confirming the resolution for this case."
            ),
        ],
        root_cause="The authentication certificate had expired.",
        resolution_steps="Rotated the certificate and confirmed successful authentication.",
    )


def make_hit(state, index=1) -> RetrievalHit:
    case = ExtractedCase(
        id="support-1",
        metadata={"internal": "omit from formatted text"},
        output=state,
        review=CaseReview(extractable=True, non_extractable_reasoning=None),
        execution={"handoff_notes": [{"note": "internal working note"}]},
    )
    return {
        "id": case.id, "case": case, "item_index": index, "entry_id": f"entry-{index}",
        "score": 1.0, "cosine_similarity": 1.0, "bm25_score": None, "source": "direct",
    }


def test_custom_schema_and_prompts_preserve_workflow_contract(state):
    assert SupportCase.model_validate_json(state.model_dump_json()) == state
    assert "handoff_notes" not in SupportCase.model_fields
    assert WORKER_PROMPT.startswith(WORKER_INSTRUCTIONS)
    assert REVIEWER_PROMPT.startswith(REVIEWER_INSTRUCTIONS)
    for prompt in (WORKER_PROMPT, REVIEWER_PROMPT):
        assert "BOTH name and kind" in prompt
        assert "reference guidance, not proof" in prompt
    with pytest.raises(ValidationError):
        SupportEntity(name="AUTH_CERT_EXPIRED")
    assert SupportEntity(name=" ", kind="error_code").name == " "
    with pytest.raises(ValidationError):
        SupportEntity(name="AUTH_CERT_EXPIRED", kind="unrecognized")


@pytest.mark.parametrize("field", ["root_cause", "resolution_steps"])
def test_custom_conclusions_use_only_maximum_length_constraints(state, field):
    value = state.model_dump()
    value[field] = "  "
    assert getattr(SupportCase.model_validate(value), field) == "  "
    value[field] = None
    assert getattr(SupportCase.model_validate(value), field) is None


def test_text_views_preserve_anchor_mapping_and_exclude_execution(state):
    assert state_to_text(state) == state.timeline
    assert state_to_text(state) is not state.timeline
    linking = case_to_text(state)
    assert linking == "\n".join([
        "Error codes: AUTH_CERT_EXPIRED", state.root_cause, state.resolution_steps,
    ])
    hit = make_hit(state)
    before = hit["case"].model_dump()
    payload = json.loads(format_case(hit))
    assert payload == {
        "id": "support-1", "item_index": 1,
        "entities": [entity.model_dump() for entity in state.entities],
        "timeline": state_to_text(state),
        "root_cause": state.root_cause, "resolution_steps": state.resolution_steps,
    }
    assert hit["case"].model_dump() == before


def test_unresolved_and_empty_graph_text(state):
    unresolved = state.model_copy(update={"root_cause": None, "resolution_steps": None})
    assert case_to_text(unresolved) == (
        "Error codes: AUTH_CERT_EXPIRED\n" + unresolved.timeline[-1]
    )
    empty = SupportCase(entities=[], timeline=[], root_cause=None, resolution_steps=None)
    assert state_to_text(empty) == []
    assert case_to_text(empty) == ""


@pytest.mark.parametrize("index", [-1, 2])
def test_formatter_rejects_invalid_anchor_indices(state, index):
    with pytest.raises(ValueError, match="item_index"):
        format_case(make_hit(state, index))


def test_formatter_rejects_other_output_models():
    default = CaseExtraction(entities=[], timeline=[], root_cause=None, resolution_steps=None)
    with pytest.raises(TypeError, match="SupportCase"):
        format_case(make_hit(default, 0))


@pytest.mark.parametrize("code,status", [
    ("AUTH_CERT_EXPIRED", "found"), ("UNKNOWN_CODE", "not_found"),
])
async def test_domain_tool_returns_explicit_lookup_status(code, status):
    arguments = json.dumps({"error_code": code})
    result = await lookup_error.on_invoke_tool(
        ToolContext(
            context=None, tool_name="lookup_error", tool_call_id="lookup",
            tool_arguments=arguments,
        ),
        arguments,
    )
    assert result["status"] == status
    assert result["error_code"] == code
    assert ("guidance" in result) is (status == "found")


def test_round_robin_routes_same_agent_without_mutating_it():
    first, second, review = (ScriptedModel([]) for _ in range(3))
    worker = Agent(name="shared-worker", model=first)
    configure = round_robin_run_config([first, second], reviewer_model=review)
    context = SimpleNamespace(stage="worker")
    configs = [configure(worker, context) for _ in range(3)]
    assert [config.model for config in configs] == [first, second, first]
    assert len({id(config) for config in configs}) == 3
    assert configure(worker, SimpleNamespace(stage="reviewer")).model is review
    assert configure(worker, context).model is second
    assert all(config.tracing_disabled for config in configs)
    assert worker.model is first
    with pytest.raises(ValueError, match="at least one"):
        round_robin_run_config([], reviewer_model=review)


@pytest.mark.parametrize("with_graph", [False, True])
async def test_custom_pipeline_real_sdk_passes_retrieval_and_reopen(tmp_path, state, with_graph):
    def worker_steps(output, note):
        return [
            [
                call("write_handoff_note", {"note": note}, "note"),
                call("edit_state", {
                    "patch_json": json.dumps([{
                        "op": "add", "path": "", "value": output.model_dump(),
                    }]),
                    "finish_pass": True,
                }, "edit"),
            ],
            [message("Pass complete.")],
        ]

    initial = state.model_copy(update={
        "timeline": state.timeline[:1], "root_cause": None, "resolution_steps": None,
    })
    first = ScriptedModel(worker_steps(initial, "Check certificate validity."))
    second = ScriptedModel(worker_steps(state, "Validity checked; rotation resolved the case."))
    reviewer = ScriptedModel([
        [review_call({"extractable": True, "non_extractable_reasoning": None})],
        [message("Review complete.")],
    ])
    unused = ScriptedModel([])
    backend = Embeddings()
    pipeline = build_pipeline(
        tmp_path, worker_model=unused, reviewer_model=unused,
        embedding_backend=backend, with_graph=with_graph,
        run_config=round_robin_run_config([first, second], reviewer_model=reviewer),
    )
    pipeline.extraction.update(batch_budget={"unit": "chars", "limit": 2}, retries=0)
    worker_agent = pipeline.extraction["worker_agent"]
    review_agent = pipeline.extraction["reviewer_agent"]
    assert worker_agent.output_type is None
    assert review_agent.output_type is None
    assert pipeline.extraction["review_output_type"] is CaseReview
    assert [tool.name for tool in worker_agent.tools] == [
        "query_case_sql", "read_state", "edit_state", "write_handoff_note", "lookup_error",
    ]
    assert [tool.name for tool in review_agent.tools] == [
        "query_case_sql", "read_state", "edit_state", "lookup_error",
    ]

    raw = [{"id": "support-1", "metadata": {}, "artifacts": [{}, {}]}]
    indexed = await pipeline.index(raw)
    assert not indexed["extraction"]["failed_cases"], indexed
    assert not indexed["embedding"]["failed_cases"], indexed
    if with_graph:
        assert not indexed["graph"]["failed_cases"], indexed
        assert [case_to_text(state)] in backend.calls
    case = indexed["indexed_cases"][0]["case"]
    assert case.output == state
    assert case.review.extractable
    assert case.execution["passes"] == 2
    assert [note["pass_number"] for note in case.execution["handoff_notes"]] == [1, 2]
    assert state_to_text(state) in backend.calls
    assert worker_agent.model is review_agent.model is unused

    reopened = build_pipeline(
        tmp_path, worker_model=unused, reviewer_model=unused, embedding_backend=backend,
    )
    assert (await reopened.index(raw))["skipped_ids"] == ["support-1"]
    result = (await reopened.retrieve(
        ["AUTH_CERT_EXPIRED"], format_case=format_case, context_budget=None,
    ))["results"][0]
    assert result["error"] is None
    hit = result["candidates"][0]
    assert isinstance(hit["case"].output, SupportCase)
    assert result["formatted_context"] == format_case(hit)
    limit = len(result["formatted_context"])
    for budget, fits in ((limit, True), (limit - 1, False)):
        bounded = (await reopened.retrieve(
            ["AUTH_CERT_EXPIRED"], format_case=format_case,
            context_budget={"unit": "chars", "limit": budget},
        ))["results"][0]
        assert bounded["error"] is None
        assert bool(bounded["candidates"]) is fits


@pytest.mark.parametrize("fail", [False, True])
async def test_endpoint_example_configures_distinct_clients_and_closes_them(
    monkeypatch, tmp_path, fail,
):
    for key, value in {
        "OPENAI_API_KEY": "test-primary-key",
        "OPENAI_BASE_URL": "https://primary.example.invalid/v1",
        "PRIMARY_MODEL": "primary-model",
        "SECONDARY_API_KEY": "test-secondary-key",
        "SECONDARY_BASE_URL": "https://secondary.example.invalid/v1",
        "SECONDARY_MODEL": "secondary-model",
    }.items():
        monkeypatch.setenv(key, value)
    clients = []
    raw = [{"id": "test", "metadata": {}, "artifacts": []}]
    expected = {"extraction": {"failed_cases": []}, "embedding": {"failed_cases": []}}

    def build(output_dir, **options):
        assert output_dir == tmp_path
        first, second = options["worker_model"], options["reviewer_model"]
        clients.extend([first._client, second._client])
        assert first.model == "primary-model" and second.model == "secondary-model"
        assert str(first._client.base_url) == "https://primary.example.invalid/v1/"
        assert str(second._client.base_url) == "https://secondary.example.invalid/v1/"
        assert all(client.max_retries == 0 for client in clients)
        assert options["embedding_backend"].client is first._client
        configure = options["run_config"]
        worker = Agent(name="same-agent")
        context = SimpleNamespace(stage="worker")
        assert configure(worker, context).model is first
        assert configure(worker, context).model is second
        assert configure(worker, SimpleNamespace(stage="reviewer")).model is second

        async def index(cases):
            assert cases is raw
            assert not any(client.is_closed() for client in clients)
            if fail:
                raise RuntimeError("Example indexing failure")
            return expected

        return SimpleNamespace(index=index)

    monkeypatch.setattr(model_routing, "build_pipeline", build)
    if fail:
        with pytest.raises(RuntimeError, match="Example indexing failure"):
            await model_routing.index_with_model_pool(raw, tmp_path)
    else:
        assert await model_routing.index_with_model_pool(raw, tmp_path) is expected
    assert all(client.is_closed() for client in clients)
