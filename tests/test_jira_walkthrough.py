"""Offline checks for the human-facing Jira notebook."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples/jira_walkthrough.ipynb"


@pytest.fixture
def cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"][-1]["id"] == "evaluate"
    sources = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, cell["id"], "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            sources[cell["id"]] = source
    return sources


@pytest.fixture
def state(cells, monkeypatch, tmp_path):
    import dotenv

    pytest.importorskip("IPython", reason="Notebook execution checks require the notebook extra")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    ns = {}
    exec(cells["setup"], ns)
    data_dir = tmp_path / "datasets/Apache_Jira"
    data_dir.mkdir(parents=True)
    corpus = [
        {
            "key": "JIRA-1", "case_id": "123", "project": "JIRA",
            "summary": "A technical failure", "metadata": {"source": "Apache Jira"},
            "conversations": [{"from": "reporter", "body": "The operation fails."}],
            "description": "Do not concatenate this duplicate representation.",
            "cluster": "hidden-label", "role": "gold_target",
        },
        {
            "key": "JIRA-2", "case_id": "456", "project": "JIRA",
            "summary": "Another technical failure", "metadata": {"source": "Apache Jira"},
            "conversations": [{"body": "Another failure."}],
            "cluster": "another-label", "role": "fixed_distractor",
        },
    ]
    queries = [
        {
            "key": "JIRA-3", "target_key": "JIRA-1", "project": "JIRA",
            "query_0": "Initial symptoms only.", "query_30": "Do not use later evidence.",
            "progress_valid": {"0": True, "30": True, "60": False},
        },
        {
            "key": "JIRA-4", "target_key": "JIRA-2", "project": "JIRA",
            "query_0": "Invalid initial query.", "progress_valid": {"0": False},
        },
    ]
    for name, rows in (("corpus", corpus), ("queries", queries)):
        (data_dir / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )
    ns.update(REPO=tmp_path, OUTPUT_DIR=tmp_path / "outputs")
    exec(cells["data"], ns)
    return ns


def test_ten_case_default_and_labels_never_reach_model_inputs(state):
    assert state["CASE_LIMIT"] == 10
    assert len(state["cases"]) == 2
    assert state["MODEL"] == "gpt-5.4"
    first = state["cases"][0]
    assert first == {
        "id": "JIRA-1",
        "metadata": {"source": "Apache Jira", "project": "JIRA", "summary": "A technical failure"},
        "artifacts": [{"from": "reporter", "body": "The operation fails."}],
    }
    assert "hidden-label" not in json.dumps(state["cases"])
    assert "Do not concatenate" not in json.dumps(state["cases"])


def test_explicit_small_run_does_not_change_held_out_query_set(state, cells):
    state["CASE_LIMIT"] = 1
    exec(cells["data"], state)
    assert len(state["cases"]) == 1
    assert len(state["queries"]) == 2


def test_full_corpus_option_keeps_every_case(state, cells):
    state["CASE_LIMIT"] = None
    exec(cells["data"], state)
    assert len(state["cases"]) == len(state["corpus"])


def test_data_rejects_corpus_query_overlap(state, cells):
    path = state["REPO"] / "datasets/Apache_Jira/queries.jsonl"
    path.write_text(json.dumps({"key": "JIRA-1"}), encoding="utf-8")
    with pytest.raises(AssertionError):
        exec(cells["data"], state)


async def execute(source, namespace):
    await eval(compile(source, "notebook-cell", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), namespace)


@pytest.mark.asyncio
@pytest.mark.parametrize("returned_id,expected_hit", [("JIRA-1", True), ("JIRA-10", False)])
async def test_final_cell_uses_valid_query_zero_and_exact_target_key(
    state, cells, returned_id, expected_hit
):
    class Retriever:
        async def retrieve(self, queries, **options):
            assert queries == ["Initial symptoms only."]
            assert options["top_k"] == 5
            assert "case_filter" not in options
            return {"results": [{"error": None, "candidates": [{"id": returned_id}]}]}

    closed = []

    async def close():
        closed.append(True)

    state.update(
        reopened=Retriever(), client=SimpleNamespace(close=close), display=lambda *args: None,
        stored_cases=[SimpleNamespace(id="JIRA-1"), SimpleNamespace(id="JIRA-2")],
    )
    await execute(cells["evaluate"], state)
    report = json.loads((state["OUTPUT_DIR"] / "case_hits.json").read_text())
    assert report["progress"] == 0
    assert report["indexed_cases"] == 2
    assert len(report["results"]) == 1
    assert report["results"][0]["hit"] is expected_hit
    assert closed == [True]


@pytest.mark.asyncio
async def test_failed_query_cannot_be_scored_as_a_miss(state, cells):
    class Retriever:
        async def retrieve(self, *args, **kwargs):
            return {"results": [{"error": {"type": "RateLimitError"}, "candidates": []}]}

    state["reopened"] = Retriever()
    with pytest.raises(RuntimeError, match="RateLimitError"):
        await execute(cells["evaluate"], state)
    assert not (state["OUTPUT_DIR"] / "case_hits.json").exists()


@pytest.mark.asyncio
async def test_index_failure_is_saved_and_blocks_continuation(state, cells):
    class Pipeline:
        async def index(self, cases):
            return {
                "extraction": {"failed_cases": [
                    {"id": "JIRA-1", "error_type": "TimeoutError", "error_message": "Timed out"}
                ]},
                "embedding": {"failed_cases": []},
            }

    state["pipeline"] = Pipeline()
    with pytest.raises(RuntimeError, match="Indexing incomplete"):
        await execute(cells["index"], state)
    assert (state["OUTPUT_DIR"] / "last_extraction.json").exists()


def test_notebook_has_no_legacy_synthetic_source_or_agent_directive_overrides(cells):
    joined = "\n".join(cells.values())
    assert "synthetic_data" not in joined
    assert "Completion rule:" not in joined
    assert "INSTRUCTIONS +" not in joined
    assert "RunConfig(tracing_disabled=True)" in joined
    assert "graph=None" in joined


@pytest.mark.asyncio
async def test_notebook_filter_demonstrates_custom_state_formatter_and_budget(state, cells):
    state["question"] = state["queries"][0]

    class Retriever:
        async def retrieve(self, queries, **options):
            assert queries == [state["question"]["query_0"]]
            assert options["max_chars"] == 16_000
            assert options["top_k"] == state["TOP_K"]
            from pydantic import RootModel

            case = SimpleNamespace(
                metadata={"project": state["question"]["project"]},
                output=RootModel[list[str]](["The matched state"]),
            )
            assert options["format_case"]({"case": case}) == '["The matched state"]'
            assert options["case_filter"](queries[0], case)
            return {"results": [{"error": None, "candidates": []}]}

    state["pipeline"] = Retriever()
    await execute(cells["filter"], state)
