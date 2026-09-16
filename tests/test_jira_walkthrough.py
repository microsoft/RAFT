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


def test_install_cell_uses_selected_kernel_without_installing_during_tests(cells, monkeypatch):
    import subprocess
    import sys

    commands = []
    monkeypatch.chdir(ROOT / "examples")
    monkeypatch.setattr(subprocess, "check_call", lambda command: commands.append(command))
    exec(cells["install"], {})
    assert commands == [[
        sys.executable, "-m", "pip", "install", "--quiet", "-e",
        str(ROOT), "python-dotenv", "ipywidgets", "tiktoken",
    ]]


@pytest.fixture
def state(cells, monkeypatch, tmp_path):
    import dotenv

    pytest.importorskip("IPython", reason="Notebook execution checks require IPython")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    ns = {}
    exec(cells["setup"], ns)
    budget_settings = ast.Module(
        body=[
            node for node in ast.parse(cells["pipeline"]).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "context_budget"
        ],
        type_ignores=[],
    )
    exec(compile(budget_settings, "extraction-budget", "exec"), ns)
    retrieval_settings = ast.Module(
        body=[
            node for node in ast.parse(cells["retrieve"]).body
            if isinstance(node, ast.Import) or (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {
                    "TOP_K", "RETRIEVAL_CONCURRENCY", "RETRIEVAL_RPM",
                    "retrieval_encoding", "retrieval_budget",
                }
            )
        ],
        type_ignores=[],
    )
    exec(compile(retrieval_settings, "retrieval-settings", "exec"), ns)
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


def test_case_default_and_labels_never_reach_model_inputs(state):
    assert state["CASE_LIMIT"] == 50
    assert len(state["cases"]) == 2
    assert state["context_budget"] == {"unit": "chars", "limit": 256_000}
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
    exec(cells["data"].replace("CASE_LIMIT = 50", "CASE_LIMIT = 1"), state)
    assert len(state["cases"]) == 1
    assert len(state["queries"]) == 2


def test_full_corpus_option_keeps_every_case(state, cells):
    state["CASE_LIMIT"] = None
    exec(cells["data"].replace("CASE_LIMIT = 50", "CASE_LIMIT = None"), state)
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
            assert options["context_budget"] is state["retrieval_budget"]
            assert options["concurrency"] == state["RETRIEVAL_CONCURRENCY"]
            assert options["rpm"] == state["RETRIEVAL_RPM"]
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
    assert report["context_budget"] == {"unit": "tokens", "limit": 5000}
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
async def test_index_failure_is_retained_in_memory_and_blocks_continuation(state, cells):
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
    assert state["failures"][0]["error_type"] == "TimeoutError"
    assert not (state["OUTPUT_DIR"] / "last_extraction.json").exists()


@pytest.mark.asyncio
async def test_failure_display_is_bounded_and_keeps_full_details(state, cells, capsys):
    class Pipeline:
        async def index(self, cases):
            return {
                "extraction": {"failed_cases": [
                    {
                        "id": f"CASE-{i}", "error_type": "NotFoundError",
                        "error_message": "full detail " * 500,
                        "error_details": {"status_code": 404, "code": "model_not_found"},
                    } for i in range(20)
                ]},
                "embedding": {"failed_cases": []},
            }

    capsys.readouterr()
    state["pipeline"] = Pipeline()
    with pytest.raises(RuntimeError, match=r"failures\[0\]"):
        await execute(cells["index"], state)
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 6
    assert "CASE-4" in output and "CASE-5" not in output
    assert len(output) < 2000
    assert len(state["failures"]) == 20
    assert len(state["failures"][0]["error_message"]) > 200


def test_notebook_has_no_legacy_synthetic_source_or_agent_directive_overrides(cells):
    joined = "\n".join(cells.values())
    assert "synthetic_data" not in joined
    assert "Completion rule:" not in joined
    assert "INSTRUCTIONS +" not in joined
    assert "RunConfig(tracing_disabled=True)" in joined
    assert "graph=None" in joined
    assert "'suppress_response_errors': True" in cells["pipeline"]


@pytest.mark.asyncio
async def test_notebook_filter_demonstrates_custom_state_formatter_and_budget(state, cells):
    state["question"] = state["queries"][0]
    state["context_budget"] = {"unit": "chars", "limit": 1234}

    class Retriever:
        async def retrieve(self, queries, **options):
            assert queries == [state["question"]["query_0"]]
            assert options["context_budget"] is state["retrieval_budget"]
            assert options["context_budget"]["limit"] == 5000
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


@pytest.mark.asyncio
async def test_main_retrieval_uses_independent_5000_token_budget(state, cells):
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    extraction_budget = state["context_budget"]

    class Retriever:
        async def retrieve(self, queries, **options):
            budget = options["context_budget"]
            assert budget is not extraction_budget
            assert budget["unit"] == "tokens" and budget["limit"] == 5000
            text = 'お誕生日おめでとう\n{"request_id": "abc-123"}'
            assert budget["count_tokens"](text) == len(encoding.encode_ordinary(text))
            return {"results": [{"error": None, "candidates": [], "used_chars": 0}]}

    state["pipeline"] = Retriever()
    await execute(cells["retrieve"], state)
    assert state["context_budget"] is extraction_budget
    assert extraction_budget == {"unit": "chars", "limit": 256_000}


def test_extraction_cell_defines_budget_once_without_source_level_coupling(state, cells):
    state.update(
        worker=object(), reviewer=object(), client=object(),
        OpenAIEmbeddings=lambda **kwargs: object(),
        LocalPipeline=lambda output_dir, **kwargs: SimpleNamespace(**kwargs),
    )
    exec(cells["pipeline"], state)
    assert state["extraction"]["batch_budget"] is state["context_budget"]
    assert state["extraction"]["batch_budget"] is not state["retrieval_budget"]
    assert state["extraction"]["query_budget"] is not state["context_budget"]
    assert "context_budget" not in state["embedding"]
    assert "o200k_base" in cells["pipeline"]
    assert "count_tokens" in cells["pipeline"]
    assert not any(
        isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "context_budget"
            for target in node.targets
        )
        for name in ("retrieve", "filter", "evaluate")
        for node in ast.walk(ast.parse(cells[name]))
    )


async def test_evaluation_serializes_token_budget_without_callback(state, cells):
    budget = state["retrieval_budget"]

    class Retriever:
        async def retrieve(self, queries, **options):
            assert options["context_budget"] is budget
            return {"results": [{"error": None, "candidates": []}]}

    async def close():
        pass

    state.update(
        reopened=Retriever(),
        client=SimpleNamespace(close=close), display=lambda *args: None, stored_cases=[],
    )
    await execute(cells["evaluate"], state)
    report = json.loads((state["OUTPUT_DIR"] / "case_hits.json").read_text())
    assert report["context_budget"] == {"unit": "tokens", "limit": 5000}


def test_settings_are_local_to_their_stage_not_shared_globals(cells):
    setup = ast.parse(cells["setup"])
    setup_names = {
        target.id for node in setup.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
    }
    assert not setup_names & {
        "CASE_LIMIT", "MODEL", "EMBEDDING_MODEL", "CONCURRENCY", "RPM",
        "TOP_K", "context_budget", "retrieval_budget", "BUILD_GRAPH", "RUN_NAME", "OUTPUT_DIR",
    }
    configurations = {
        node.targets[0].id: node.value
        for node in ast.parse(cells["pipeline"]).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    for name in ("extraction", "embedding"):
        config = configurations[name]
        values = {key.value: value for key, value in zip(config.keys, config.values)}
        assert isinstance(values["concurrency"], ast.Constant)
        assert isinstance(values["rpm"], ast.Constant)
    assert "BUILD_GRAPH = False" in cells["graph"]
    assert "CASE_LIMIT = 50" in cells["data"]
    assert "worker_model =" in cells["agents"]
    assert "reviewer_model =" in cells["agents"]


def test_notebook_uses_package_handoff_tool_separate_from_output(cells):
    assert "write_handoff_note" in cells["setup"]
    assert "tools = [query_case_sql, edit_state]" in cells["agents"]
    assert "instructions=WORKER_INSTRUCTIONS, tools=[*tools, write_handoff_note]" in cells["agents"]
    assert "instructions=REVIEWER_INSTRUCTIONS, tools=tools" in cells["agents"]
