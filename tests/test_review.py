import json
from types import SimpleNamespace

import pytest
from agent_helpers import run_cases
from agents import Agent
from pydantic import BaseModel
from test_extraction import case, edit, options
from test_local_pipeline import pipeline, raw

from raft.cases import ExtractedCase, restore_case
from raft.defaults import CaseExtraction
from raft.extraction import _agent as sdk
from raft.extraction.handoff import apply_handoff_note
from raft.tools import edit_state, query_case_sql


class Review(BaseModel):
    keep: bool
    reason: str


@pytest.mark.asyncio
@pytest.mark.parametrize("keep", [True, False])
async def test_complete_review_and_filter_preserve_fields(monkeypatch, keep):
    reviewer = Agent(name="review", tools=[query_case_sql, edit_state], output_type=Review)
    passes = []
    seen = []

    async def run(agent, prompt, *, context, **kwargs):
        if agent is reviewer:
            payload = json.loads(prompt.split("Review context:\n")[1])
            assert "query_budget" not in payload
            assert payload["output"]["timeline"] == passes
            assert len(passes) > 1
            assert context.query("select count(*) as n from artifacts")["rows"] == [{"n": 2}]
            assert context.stage == "reviewer"
            assert context.query("select count(*) as n from state_revisions")["rows"] == [
                {"n": len(passes)}
            ]
            result = Review(keep=keep, reason="All evidence reviewed")
        else:
            passes.append(str(len(passes)))
            assert edit(
                context,
                [
                    {
                        "op": "add",
                        "path": "",
                        "value": {
                            "extractable": False,
                            "timeline": list(passes),
                        },
                    }
                ],
            )["ok"]
            result = "done"
        return SimpleNamespace(new_items=[], final_output=result)

    def should_keep(record):
        seen.append(record)
        assert isinstance(record.review, Review)
        return record.review.keep

    monkeypatch.setattr(sdk.Runner, "run", run)
    result = await run_cases(
        cases=[case(text="x" * 140)],
        **options(reviewer_agent=reviewer, should_keep=should_keep, batch_budget={"unit": "chars", "limit": 170}),
    )
    record = result["extracted_cases" if keep else "filtered_cases"][0]
    assert seen == [record]
    assert record.metadata == {"product": "test"}
    assert record.output.timeline == passes
    assert record.review.reason == "All evidence reviewed"
    assert record.execution["passes"] == len(passes)
    assert len(record.execution["revisions"]) == len(passes)
    saved = record.model_dump(mode="json")
    restored = restore_case(saved, type(record.output))
    assert restored.model_dump(mode="json") == saved


@pytest.mark.asyncio
async def test_review_retry_does_not_repeat_worker(monkeypatch):
    reviewer = Agent(name="review", tools=[query_case_sql, edit_state], output_type=Review)
    calls = []

    async def run(agent, prompt, *, context, **kwargs):
        calls.append(agent.name)
        if agent is reviewer:
            if calls.count("review") == 1:
                raise TimeoutError("temporary")
            result = Review(keep=True, reason="done")
        else:
            assert edit(
                context,
                [
                    {
                        "op": "add",
                        "path": "",
                        "value": {
                            "extractable": True,
                            "timeline": ["complete"],
                        },
                    }
                ],
            )["ok"]
            result = "done"
        return SimpleNamespace(new_items=[], final_output=result)

    monkeypatch.setattr(sdk.Runner, "run", run)
    result = await run_cases(
        cases=[case()], **options(reviewer_agent=reviewer, retries=1, max_passes=1)
    )
    assert calls == ["scan", "review", "review"]
    assert result["extracted_cases"][0].execution["attempts"] == 2


@pytest.mark.asyncio
async def test_filtered_catalog_preserves_record_and_skips_embedding(tmp_path):
    p = pipeline(tmp_path)
    p.extraction["should_keep"] = lambda case: False
    result = await p.index([raw("a")])
    assert result["extraction"]["filtered_cases"][0].output.timeline == ["a"]
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    record = next(iter(catalog["cases"].values()))
    assert record["status"] == "filtered"
    assert record["case"]["output"]["timeline"] == ["a"]
    assert record["case"]["execution"]["revisions"][0]["state"]["timeline"] == ["a"]
    assert record["embedding"] is None
    assert p.embedding["backend"].calls == []


def test_default_handoff_notes_independent():
    state = CaseExtraction(entities=[], timeline=[], root_cause=None, resolution_steps=None)
    a = ExtractedCase(id="a", metadata={}, output=state)
    b = ExtractedCase(id="b", metadata={}, output=state)
    a.execution["handoff_notes"].append("Check certificate date")
    assert b.execution["handoff_notes"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("review_value", ["unstructured", None])
async def test_invalid_review_keeps_worker_in_failure_and_never_filters(monkeypatch, review_value):
    reviewer = Agent(name="review", tools=[query_case_sql, edit_state], output_type=Review)

    async def run(agent, prompt, *, context, **kwargs):
        if agent is not reviewer:
            assert edit(
                context,
                [
                    {
                        "op": "add",
                        "path": "",
                        "value": {
                            "extractable": True,
                            "timeline": ["preserved"],
                        },
                    }
                ],
            )["ok"]
        return SimpleNamespace(new_items=[], final_output=review_value)

    def should_keep(case):
        pytest.fail("filter must not run on review failure")

    monkeypatch.setattr(sdk.Runner, "run", run)
    result = await run_cases(
        cases=[case()], **options(reviewer_agent=reviewer, should_keep=should_keep)
    )
    assert not result["filtered_cases"]
    failure = result["failed_cases"][0]
    assert failure["stage"] == "review"
    assert failure["output"]["timeline"] == ["preserved"]


@pytest.mark.asyncio
async def test_handoff_notes_carried_between_worker_passes(monkeypatch):
    seen = []

    async def run(agent, prompt, *, context, **kwargs):
        payload = json.loads(prompt.split("Pass context:\n")[1])
        notes = [record["note"] for record in payload["handoff_notes"]]
        assert notes == seen
        seen.append(f"Note from pass {len(seen) + 1}")
        assert apply_handoff_note(context=context, note=seen[-1])["ok"]
        assert edit(
            context,
            [
                {
                    "op": "add",
                    "path": "",
                    "value": {
                        "entities": [],
                        "timeline": [],
                        "root_cause": None,
                        "resolution_steps": None,
                    },
                }
            ],
        )["ok"]
        return SimpleNamespace(new_items=[], final_output="done")

    monkeypatch.setattr(sdk.Runner, "run", run)
    result = await run_cases(
        cases=[case(text="x" * 140)], **options(output_type=CaseExtraction, batch_budget={"unit": "chars", "limit": 170})
    )
    assert len(seen) > 1
    assert [record["note"] for record in result["extracted_cases"][0].execution["handoff_notes"]] == seen
    assert "handoff_notes" not in result["extracted_cases"][0].output.model_dump()


@pytest.mark.asyncio
async def test_rewrite_to_filtered_removes_old_embedding_and_survives_reopen(tmp_path):
    p = pipeline(tmp_path)
    await p.index([raw("a")])
    p.extraction["should_keep"] = lambda case: False
    result = await p.index([raw("a", ["new facts"])], rewrite=True)
    assert result["summary"]["stored_embedded"] == 0
    assert result["summary"]["stored_filtered"] == 1
    reopened = pipeline(tmp_path)
    result = await reopened.index([])
    assert result["summary"]["stored_filtered"] == 1
    assert result["summary"]["stored_embedded"] == 0
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    record = next(iter(catalog["cases"].values()))
    assert record["case"]["output"]["timeline"] == ["new facts"]
    assert record["embedding"] is None


@pytest.mark.asyncio
async def test_reviewer_is_required_before_processing():
    settings = options()
    settings.pop("reviewer_agent")
    with pytest.raises(TypeError, match="reviewer_agent"):
        await run_cases(cases=[], **settings)
    with pytest.raises(ValueError, match="reviewer_agent is required"):
        await run_cases(cases=[], **options(reviewer_agent=None))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tools, missing",
    [
        ([], "edit_state, query_case_sql"),
        ([query_case_sql], "edit_state"),
        ([edit_state], "query_case_sql"),
    ],
)
async def test_reviewer_registration_checked_before_processing(monkeypatch, tools, missing):
    async def unexpected_run(*args, **kwargs):
        pytest.fail("SDK must not run when required reviewer tools are missing")

    monkeypatch.setattr(sdk.Runner, "run", unexpected_run)
    reviewer = Agent(name="review", tools=tools, output_type=Review)
    with pytest.raises(ValueError, match=f"reviewer_agent.tools is missing required tools: {missing}\\."):
        await run_cases(cases=[case()], **options(reviewer_agent=reviewer))
    assert reviewer.tools == tools


@pytest.mark.asyncio
async def test_reviewer_native_output_is_checked():
    with pytest.raises(ValueError, match="output_type"):
        await run_cases(
            cases=[],
            **options(reviewer_agent=Agent(name="review", tools=[query_case_sql, edit_state])),
        )
