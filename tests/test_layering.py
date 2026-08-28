"""The architectural boundaries, enforced.

Migration plan §2: *"One structural rule worth enforcing with a test:
``graph/build.py`` is the only module that imports ``langgraph``, and
``execution/`` is the only package that imports ``ib_async``. Both are one-line
grep-style tests, and both protect a decision that is otherwise easy to erode a
commit at a time."*

These read the AST rather than grepping, so a comment mentioning ``langgraph``
does not fail the build and ``import langgraph.graph as g`` does not sneak past.

If one of these fails, the fix is almost never to add an exemption here. Each
rule protects something that was decided deliberately and written down:

* **langgraph** -- architecture §1 and §15.4. LangGraph is an orchestrator. If a
  node body imports it, the next thing that happens is a ``ToolNode``.
* **ib_async** -- architecture §0. ``decide`` and ``execute`` are separate
  processes with separate event loops. ``ib_async`` installs its own loop policy
  and needs ``nest_asyncio``; mixing that with a dozen concurrent LLM HTTP calls
  is a debugging tarpit.
* **provider imports** -- architecture §2. Everything reaches a fetch through
  ``providers/registry.py``, which is what keeps the tool-promotion path open.
* **no tool-calling** -- architecture §2 and §15.4.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "research_desk"

#: The single module allowed to import langgraph.
LANGGRAPH_OWNER = SRC / "graph" / "build.py"

#: The single package allowed to import ib_async.
IB_OWNER = SRC / "execution"


def _modules() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    """Top-level package names imported by a module, from its AST."""
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # `level > 0` is a relative import -- never an external package.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(SRC)))
def test_only_build_imports_langgraph(path: Path) -> None:
    if path == LANGGRAPH_OWNER:
        return
    offenders = {r for r in _imported_roots(path) if r in {"langgraph", "langgraph_sdk"}}
    assert not offenders, (
        f"{path.relative_to(SRC)} imports {offenders}. Only graph/build.py may. "
        "Node bodies stay framework-free so that an upgrade is a plumbing change "
        "rather than a rewrite (architecture §1, §15.5)."
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(SRC)))
def test_only_execution_imports_ib_async(path: Path) -> None:
    if IB_OWNER in path.parents:
        return
    assert "ib_async" not in _imported_roots(path), (
        f"{path.relative_to(SRC)} imports ib_async, but only execution/ may. "
        "`decide` must have no IB connection and no ib_async in its dependency "
        "tree at all (architecture §0)."
    )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_langchain_chat_models(path: Path) -> None:
    """Our router owns model calls, so Ollama's format=<json_schema> stays reachable."""
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("langchain"):
            names = {alias.name for alias in node.names}
            banned = {n for n in names if "ChatModel" in n or n.startswith("Chat")}
            assert not banned, (
                f"{path.relative_to(SRC)} imports {banned} from {node.module}. "
                "LangGraph is an orchestrator only -- a node body calls OUR "
                "client (architecture §1, constraint 1)."
            )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_tool_calling_primitives(path: Path) -> None:
    """No ToolNode, no ReAct loop, and no interrupt()-based human gate.

    The last one matters most. LangGraph's human-in-the-loop primitives will
    tempt someone to collapse `decide` and `execute` into one process, which
    reintroduces exactly the event-loop tangle the §0 split exists to avoid.
    """
    banned = {"ToolNode", "create_react_agent", "interrupt", "ToolExecutor"}
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("langgraph"):
            found = {a.name for a in node.names} & banned
            assert not found, (
                f"{path.relative_to(SRC)} imports {found}. There is no "
                "tool-calling in this design, and the human gate is a file read "
                "in a separate process (architecture §2, §1 constraint 3)."
            )


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(SRC)))
def test_providers_reached_only_through_the_registry(path: Path) -> None:
    """Nothing imports a provider module directly.

    This is what makes the prefetch functions promotable to real tools later as
    a config change rather than a refactor (architecture §2).
    """
    if path.parent.name == "providers":
        return
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        parts = node.module.split(".")
        if "providers" in parts:
            tail = parts[parts.index("providers") + 1 :]
            assert not tail or tail[0] == "registry", (
                f"{path.relative_to(SRC)} imports providers.{'.'.join(tail)} "
                "directly. Every fetch goes through providers/registry.py."
            )


def test_the_owners_actually_exist() -> None:
    """Guard against the rules passing vacuously after a rename."""
    assert LANGGRAPH_OWNER.exists(), f"{LANGGRAPH_OWNER} is gone; update this test."
    assert IB_OWNER.is_dir(), f"{IB_OWNER} is gone; update this test."
    assert "langgraph" in _imported_roots(LANGGRAPH_OWNER), (
        "graph/build.py no longer imports langgraph, so the rule above proves "
        "nothing. Either LangGraph was dropped -- in which case say so in the "
        "architecture doc -- or this test needs a new owner."
    )
