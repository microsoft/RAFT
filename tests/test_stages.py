"""Independent extraction and embedding composed with deterministic fake backends."""

import asyncio
import json

import pytest
from agent_helpers import FakeReview, run_cases
from pydantic import BaseModel

from raft import ExtractedCase, embed_cases
from raft.extraction.state import apply_edit
from raft.runtime import RetryDecision


class Output(BaseModel):
    extractable: bool
    timeline: list[str]


async def run_stages(*, cases, extraction, embedding):
    extracted = await run_cases(cases=cases, **extraction)
    embedded = await embed_cases(cases=extracted["extracted_cases"], **embedding)
    return {
        "extraction": extracted,
        "embedding": embedded,
        "indexed_cases": embedded["embedded_cases"],
    }


class Worker(FakeReview):
    def __init__(self):
        self.calls = []

    def prepare(self, agent):
        return agent

    async def run(self, agent, prompt, *, context, max_turns, telemetry):
        self.calls.append(context.case_id)
        if context.case_id == "scan-fails":
            raise ValueError("Extraction failed")
        telemetry.usage = {"test-model": {"input_tokens": 12, "output_tokens": 7}}
        telemetry.rounds.append({"round": 1, "tool_calls": [{"name": "edit_state"}]})
        result = apply_edit(
            context=context,
            patch_json=json.dumps(
                [
                    {
                        "op": "add",
                        "path": "",
                        "value": {
                            "extractable": context.case_id != "filtered",
                            "timeline": []
                            if context.case_id == "empty"
                            else [str(context.case_id)],
                        },
                    }
                ]
            ),
            finish_pass=True,
        )
        assert result["ok"]

    def aggregate_usage(self, usages):
        return {"test-model": {key: sum(u["test-model"].get(key, 0) for u in usages)
                               for key in ("input_tokens", "output_tokens")}}

    def classify_error(self, exc):
        return RetryDecision(False, "worker_error")


class Embeddings:
    name = "fake"
    model = "test"

    def __init__(self):
        self.calls = []
        self.fail = True

    async def embed(self, texts, *, input_type="document"):
        self.calls.append(texts)
        if self.fail and "embed-fails" in texts:
            raise ValueError("Embedding failed")
        return [[1.0, 2.0] for _ in texts]

    def classify_error(self, exc):
        return RetryDecision(False, "embedding_error")


def config(worker, embeddings):
    return {
        "extraction": {
            "worker_agent": object(),
            "reviewer_agent": object(),
            "_agent_runner": worker,
            "output_type": Output,
            "id_field": "ticket",
            "metadata_field": "metadata",
            "artifacts_field": "artifacts",
            "retries": 0,
            "rpm": 1000,
        },
        "embedding": {
            "should_embed": lambda state: state.extractable,
            "backend": embeddings,
            "state_to_text": lambda state: state.timeline,
            "retries": 0,
            "rpm": 1000,
        },
    }


def cases(*ids):
    return [{"ticket": id, "metadata": {"product": "p"}, "artifacts": []} for id in ids]


@pytest.mark.asyncio
async def test_stages_keep_models_telemetry_failures_and_write_nothing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Processing must not call storage")

    for name in ("save_json", "save_jsonl", "save_embeddings", "save_graph"):
        monkeypatch.setattr(f"raft.storage.{name}", forbidden)
    monkeypatch.setattr("raft.embedding.BM25Index.from_records", forbidden)
    worker, embeddings = Worker(), Embeddings()
    options = config(worker, embeddings)
    before = {key: dict(value) for key, value in options.items()}
    result = await run_stages(
        cases=cases("ok", "filtered", "scan-fails", "embed-fails", "empty"), **options
    )
    assert options == before
    assert result["extraction"]["summary"] == {
        "total": 5,
        "extracted": 4,
        "failed": 1,
    }
    assert result["embedding"]["summary"] == {
        "total": 4,
        "embedded": 2,
        "skipped": 1,
        "failed": 1,
        "items": 1,
    }
    assert result["indexed_cases"] is result["embedding"]["embedded_cases"]
    assert [item["id"] for item in result["indexed_cases"]] == ["ok", "empty"]
    item = result["indexed_cases"][0]
    case = item["case"]
    assert case is result["extraction"]["extracted_cases"][0]
    assert isinstance(case, ExtractedCase) and isinstance(case.output, Output)
    assert case.metadata == {"product": "p"}
    assert case.execution["usage"] == {"test-model": {"input_tokens": 12, "output_tokens": 7}}
    assert case.execution["tool_calls"][0]["rounds"][0]["tool_calls"][0]["name"] == "edit_state"
    assert item["requests"] == 1 and item["attempts"] == 1
    assert item["embeddings"][0]["case_id"] == "ok"
    assert item["embeddings"][0]["text"] == "ok"
    assert result["indexed_cases"][1]["embeddings"] == []
    failed = result["embedding"]["failed_cases"][0]
    assert failed["case"] is result["extraction"]["extracted_cases"][2]
    assert result["embedding"]["skipped_cases"][0] is result["extraction"]["extracted_cases"][1]
    assert sorted(embeddings.calls) == [["embed-fails"], ["ok"]]

    # Retry embedding directly; no extraction invocation or model serialization.
    embeddings.fail = False
    worker_calls = list(worker.calls)
    retried = await embed_cases(cases=[failed["case"]], **options["embedding"])
    assert retried["embedded_cases"][0]["case"] is failed["case"]
    assert worker.calls == worker_calls
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [(), ("filtered",), ("scan-fails",)])
async def test_stages_with_no_kept_cases(ids):
    worker, embeddings = Worker(), Embeddings()
    result = await run_stages(cases=cases(*ids), **config(worker, embeddings))
    assert result["indexed_cases"] == []
    assert result["embedding"]["summary"]["total"] == (1 if ids == ("filtered",) else 0)
    assert not embeddings.calls


@pytest.mark.asyncio
async def test_stages_preserve_id_types_and_stable_embedding_ids():
    worker, embeddings = Worker(), Embeddings()
    options = config(worker, embeddings)
    first = await run_stages(cases=cases(1, "1"), **options)
    again = await run_stages(cases=cases(1, "1"), **options)

    def ids(result):
        return [item["embeddings"][0]["id"] for item in result["indexed_cases"]]

    assert [item["case"].id for item in first["indexed_cases"]] == [1, "1"]
    assert len(set(ids(first))) == 2
    assert ids(first) == ids(again)


@pytest.mark.asyncio
async def test_stages_propagate_cancellation():
    class Cancelled(Embeddings):
        async def embed(self, texts, *, input_type="document"):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_stages(cases=cases("ok"), **config(Worker(), Cancelled()))
