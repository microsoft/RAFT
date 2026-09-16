"""The live migration example's data split and progress handling are tested offline."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "mslearn_smoke", Path(__file__).resolve().parents[1] / "examples" / "mslearn_smoke.py"
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def cases():
    return [
        {
            "case_id": f"group{group}_{variant}",
            "metadata": {"product": "example"},
            "conversations": [
                {"idx": i, "body": f"message-{i:02d}", "from": "customer"}
                for i in range(11)
            ],
            "synthetic_information": {
                "shared_id": f"group{group}", "shared_root_cause": "GOLD_ANSWER"
            },
        }
        for group in range(8) for variant in range(2)
    ]


def test_seeded_split_is_disjoint_reproducible_bounded_and_has_positives():
    original = cases()
    snapshot = deepcopy(original)
    first = smoke.select_cases(original, seed=1, test_count=3, index_count=6)
    assert first == smoke.select_cases(original, seed=1, test_count=3, index_count=6)
    assert first != smoke.select_cases(original, seed=2, test_count=3, index_count=6)
    tests, indexed = first
    assert (len(tests), len(indexed)) == (3, 6)
    assert not {c["case_id"] for c in tests} & {c["case_id"] for c in indexed}
    assert len({c["synthetic_information"]["shared_id"] for c in tests}) == 3
    for test in tests:
        assert any(
            c["synthetic_information"]["shared_id"] == test["synthetic_information"]["shared_id"]
            for c in indexed
        )
    assert original == snapshot


@pytest.mark.parametrize("test_count,index_count", [(0, 1), (3, 2), (9, 9), (3, 20)])
def test_invalid_splits_fail_instead_of_silently_reducing_sample(test_count, index_count):
    with pytest.raises(ValueError):
        smoke.select_cases(cases(), seed=1, test_count=test_count, index_count=index_count)


@pytest.mark.parametrize("progress,last_index", [(0, 0), (30, 3), (60, 6), (100, 10)])
def test_progress_uses_inclusive_message_index_without_future_content(progress, last_index):
    messages = cases()[0]["conversations"]
    query = smoke.build_query(list(reversed(messages)), progress)
    assert [f"message-{i:02d}" in query for i in range(11)] == [
        i <= last_index for i in range(11)
    ]
    assert query.startswith("From: customer\n\nmessage-00")


@pytest.mark.parametrize("progress", [-1, 101, True, 30.0])
def test_invalid_progress_is_rejected(progress):
    with pytest.raises(ValueError):
        smoke.build_query(cases()[0]["conversations"], progress)


def test_single_message_at_every_progress_and_empty_conversation_rejection():
    message = [{"idx": 0, "text": "Only message.", "subject": "Help"}]
    for progress in smoke.PROGRESSES:
        assert smoke.build_query(message, progress) == "From: unknown\nSubject: Help\n\nOnly message."
    with pytest.raises(ValueError):
        smoke.build_query([], 0)


def test_reference_answers_are_never_in_agent_inputs_or_queries():
    original = cases()[0]
    value = smoke.agent_case(original)
    assert set(value) == {"case_id", "metadata", "conversations"}
    assert "GOLD_ANSWER" not in json.dumps(value)
    assert "GOLD_ANSWER" not in smoke.build_query(value["conversations"], 60)
    value["metadata"]["product"] = "changed"
    value["conversations"][0]["body"] = "changed"
    assert original["metadata"]["product"] == "example"
    assert original["conversations"][0]["body"] == "message-00"


def test_local_dataset_validation_and_default_experiment_shape(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(cases()), encoding="utf-8")
    assert smoke.load_dataset(path) == cases()
    args = smoke.parse_args(["--dry-run"])
    assert (args.seed, smoke.PROGRESSES, args.test_cases, args.index_cases) == (1, (0, 30, 60), 3, 12)
    assert (args.model, args.embedding_model) == ("gpt-5.4", "text-embedding-3-large")
    assert (args.concurrency, args.rpm) == (80, 100)
    malformed = cases()
    malformed[1]["case_id"] = malformed[0]["case_id"]
    path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        smoke.load_dataset(path)


def test_live_checks_require_a_separate_reviewer_and_committed_worker_passes():
    record = SimpleNamespace(
        id="case-1",
        output=smoke.CaseExtraction(
            entities=[{"name": "AUTH-401"}],
            timeline=[{"narrative": "Evidence-supported narrative. " * 8}],
            root_cause=None,
            resolution_steps=None,
        ),
        review=smoke.CaseReview(extractable=True, non_extractable_reasoning=None),
        execution={
            "passes": 2,
            "revisions": [{"stage": "worker"}, {"stage": "worker"}],
            "tool_calls": [{"pass": 1}, {"pass": 2}, {"stage": "review"}],
            "usage": {"gpt-5.2": {"total_tokens": 123}},
        },
    )
    result = {"extracted_cases": [record], "filtered_cases": [], "failed_cases": []}
    assert smoke.check_extraction(result, 1) == {"case-1": 2}
    record.execution["tool_calls"].pop()
    with pytest.raises(RuntimeError, match="reviewer"):
        smoke.check_extraction(result, 1)


def test_full_split_preserves_notebook_order_all_rows_and_singleton_holdouts():
    import random

    data = cases()
    singleton = deepcopy(data[0])
    singleton["case_id"] = "singleton_1"
    singleton["synthetic_information"]["shared_id"] = "singleton"
    data.append(singleton)
    snapshot = deepcopy(data)
    shuffled = list(data)
    random.Random(1).shuffle(shuffled)
    expected_test, expected_index, seen = [], [], set()
    for case in shuffled:
        sid = case["synthetic_information"]["shared_id"]
        if len(expected_test) < 1000 and sid not in seen:
            expected_test.append(case)
            seen.add(sid)
        else:
            expected_index.append(case)
    tests, indexed = smoke.select_full_cases(data, seed=1)
    assert (tests, indexed) == (expected_test, expected_index)
    assert (len(tests), len(indexed)) == (9, 8)
    assert singleton in tests
    assert {c["case_id"] for c in tests + indexed} == {c["case_id"] for c in data}
    assert data == snapshot
    assert not {c["case_id"] for c in tests} & {c["case_id"] for c in indexed}


def test_full_defaults_match_notebook_sample_size_top_k_and_requested_model():
    args = smoke.parse_args(["--full", "--dry-run"])
    assert (args.test_cases, args.index_cases, args.top_k) == (1000, None, 10)
    assert (args.model, args.seed, args.concurrency, args.rpm) == ("gpt-5.4", 1, 80, 100)
    assert args.max_output_tokens == 8000
    with pytest.raises(SystemExit):
        smoke.parse_args(["--full", "--index-cases", "12"])
    with pytest.raises(SystemExit):
        smoke.parse_args(["--resume"])


def test_context_budget_matches_legacy_first_crossing_case_behavior(monkeypatch):
    hits = [{"id": i} for i in range(4)]
    monkeypatch.setattr(smoke, "format_case_context", lambda hit: "x" * 12)
    assert smoke.budget_candidates(hits, 5) == hits[:2]
    assert smoke.budget_candidates(hits, 6) == hits[:3]
    assert smoke.budget_candidates(hits, None) == hits


def test_case_hit_uses_original_prefix_rule_and_keeps_unmatchable_queries():
    data = cases()
    tests = [data[0], deepcopy(data[2])]
    tests[1]["case_id"] = "singleton_1"
    tests[1]["synthetic_information"]["shared_id"] = "singleton"
    output = smoke.CaseExtraction(
        entities=[], timeline=[], root_cause=None, resolution_steps=None
    )
    hit = {
        "id": data[1]["case_id"], "item_index": 0, "score": 1,
        "case": SimpleNamespace(output=output),
    }
    result = {"results": [
        {"error": None, "query": f"query{i}", "candidates": [hit]} for i in range(2)
    ]}
    rows, metrics = smoke.score_retrieval(result, tests, [data[1]])
    assert [r["case_hit"] for r in rows] == [True, False]
    assert metrics["queries"] == 2
    assert metrics["case_hits"] == 1
    assert metrics["case_hit_rate"] == 0.5


@pytest.mark.asyncio
async def test_index_resume_restores_checkpoint_without_repeating_successes(tmp_path):
    from raft.cases import ExtractedCase

    record = ExtractedCase(
        id="group0_1",
        metadata={},
        output=smoke.CaseExtraction(
            entities=[], timeline=[], root_cause=None, resolution_steps=None
        ),
        review=smoke.CaseReview(extractable=True, non_extractable_reasoning=None),
    )
    smoke.save_json(tmp_path / "pipeline" / "catalog.json", {
        "cases": {"group0_1": {"status": "embedded", "case": record}}
    })
    inputs = [{"case_id": "group0_1"}]

    class SavedPipeline:
        async def index(self, received):
            assert received == inputs
            return {
                "extraction": {"extracted_cases": [], "failed_cases": []},
                "embedding": {"failed_cases": [], "summary": {"embedded": 0}},
                "summary": {"stored_embedded": 1},
            }

    result = await smoke.index_with_retries(SavedPipeline(), inputs, tmp_path)
    restored = result["extraction"]["extracted_cases"][0]
    assert restored.id == record.id
    assert isinstance(restored.review, smoke.CaseReview)
    assert json.loads((tmp_path / "extraction.json").read_text())["summary"]["extracted"] == 1


@pytest.mark.asyncio
async def test_index_retries_preserve_successful_outputs_and_report_errors(tmp_path, monkeypatch):
    from raft.cases import ExtractedCase

    record = ExtractedCase(
        id="group0_1", metadata={},
        output=smoke.CaseExtraction(
            entities=[], timeline=[], root_cause=None, resolution_steps=None
        ),
        review=smoke.CaseReview(extractable=True, non_extractable_reasoning=None),
    )
    second = record.model_copy(update={"id": "group1_1"})
    sleeps = []

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(smoke.asyncio, "sleep", sleep)

    class RetryingPipeline:
        count = 0

        async def index(self, received):
            self.count += 1
            first = self.count == 1
            return {
                "extraction": {
                    "extracted_cases": [record if first else second],
                    "failed_cases": [{"id": second.id, "error_type": "TimeoutError"}] if first else [],
                },
                "embedding": {"failed_cases": [], "summary": {"embedded": 1}},
                "summary": {"stored_embedded": self.count},
            }

    pipeline = RetryingPipeline()
    result = await smoke.index_with_retries(pipeline, [{}, {}], tmp_path)
    assert [c.id for c in result["extraction"]["extracted_cases"]] == [record.id, second.id]
    assert result["extraction"]["failed_cases"] == []
    assert sleeps == [60]
    assert pipeline.count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status,headers,expected", [
    (429, {"x-ratelimit-reset-tokens": "57"}, "57"),
    (429, {"x-ratelimit-reset-tokens": "0"}, "1"),
    (429, {"x-ratelimit-reset-tokens": "90"}, "60"),
    (429, {"retry-after": "25", "x-ratelimit-reset-tokens": "57"}, "25"),
    (429, {"x-ratelimit-reset-tokens": "unknown"}, None),
    (200, {"x-ratelimit-reset-tokens": "57"}, None),
])
async def test_azure_token_reset_drives_http_retry_without_overriding_retry_after(
    status, headers, expected
):
    import httpx2

    response = httpx2.Response(status, headers=headers)
    await smoke.azure_retry_headers(response)
    assert response.headers.get("retry-after") == expected
