"""Core and adapter imports must not pull in unrelated agent frameworks."""

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


def test_dependencies_include_openai_agents_and_optional_integrations():
    config = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    dependencies = config["project"]["dependencies"]
    assert any(dep.startswith("openai-agents") for dep in dependencies)
    assert any(dep.startswith("openai>=") for dep in dependencies)
    assert not any("langgraph" in dep or "langchain" in dep for dep in dependencies)
    extras = config["project"]["optional-dependencies"]
    assert {
        "azure",
        "voyage",
        "dev",
    } == extras.keys()
    assert any(dep.startswith("azure-identity") for dep in extras["azure"])
    assert not any(dep.startswith("azure-identity") for dep in dependencies)
    assert not any(dep.startswith("tiktoken") for dep in dependencies)
    assert any(dep.startswith("tiktoken") for dep in extras["dev"])
    assert config["project"]["name"] == "raft"


@pytest.mark.parametrize(
    "module,allowed",
    [
        ("raft", []),
        ("raft.embedding", []),
        ("raft.embedding.openai", ["openai"]),
        ("raft.embedding.voyage", ["voyageai", "langchain_core"]),
        ("raft.extraction._agent", ["openai", "agents"]),
        ("raft.tools", ["openai", "agents"]),
    ],
)
def test_import_is_independent_of_other_sdks(module, allowed):
    for dependency in allowed:
        try:
            available = importlib.util.find_spec(dependency)
        except ModuleNotFoundError:
            available = None
        if available is None:
            pytest.skip(f"Optional dependency {dependency} is not installed")
    code = """
import importlib
import importlib.abc
import sys

roots = {"tiktoken", "voyageai", "openai", "agents", "langgraph", "langchain_core", "claude_agent_sdk",
         "agent_framework", "google.adk", "google.genai", "crewai"}
blocked = roots - set(sys.argv[2:])

class NoOtherSDKs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == root or fullname.startswith(root + ".") for root in blocked):
            raise ModuleNotFoundError("Unrelated optional SDK imported: " + fullname, name=fullname)

sys.meta_path.insert(0, NoOtherSDKs())
importlib.import_module(sys.argv[1])
assert not any(name == root or name.startswith(root + ".")
               for name in sys.modules for root in blocked)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, module, *allowed], text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
