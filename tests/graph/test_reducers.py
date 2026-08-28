"""The reducer trap, caught at Stage 0 instead of Stage 4.

Architecture §4: *"``llm_calls``, ``errors``, ``violations``, and each
``DebateTranscript.turns`` are appended to by nodes that run in parallel;
annotate them with an add-reducer or the four analysts will silently overwrite
each other's ``llm_calls``. This is the one place LangGraph's state model needs
care, and it is worth a test that runs the fan-out and asserts four records
land."*

The reason to write it now, three stages before the fan-out ships, is that the
failure is **silent**. Nothing raises. Three of four records simply are not
there, and the first symptom is a cost report at Stage 5 that looks pleasingly
low, or a Stage 8 ablation drawing a conclusion from a quarter of the evidence.
"""

from __future__ import annotations

from datetime import date

import pytest

from research_desk.config import Settings
from research_desk.context import NodeContext
from research_desk.graph.build import build_fanout_graph, build_linear_graph
from research_desk.graph.nodes.toy import fanout_probe
from research_desk.models.state import DecisionState, LLMCallRecord, NodeError


@pytest.fixture
def ctx() -> NodeContext:
    return NodeContext(settings=Settings(), as_of=date(2026, 8, 27))


def _state(ctx: NodeContext) -> DecisionState:
    return DecisionState(run_id=ctx.run_id, symbol="SPY", as_of=ctx.as_of)


async def _passthrough(state: DecisionState, ctx: NodeContext) -> dict:
    return {}


async def test_four_parallel_branches_all_survive(ctx: NodeContext) -> None:
    """The headline case: four analysts, four records, none lost."""
    branches = [(f"analyst_{i}", fanout_probe(f"analyst_{i}")) for i in range(4)]

    graph = build_fanout_graph(
        ("prefetch", _passthrough),
        branches,
        ("collect", _passthrough),
        ctx,
    )
    result = DecisionState.model_validate(await graph.ainvoke(_state(ctx)))

    assert sorted(result.notes) == ["analyst_0", "analyst_1", "analyst_2", "analyst_3"], (
        "Parallel branches overwrote each other. DecisionState.notes has lost "
        "its add-reducer annotation -- and if notes lost it, llm_calls and "
        "errors have too."
    )


async def test_llm_call_records_accumulate_across_branches(ctx: NodeContext) -> None:
    """The field this actually matters for: cost and replay would both be wrong."""

    def recorder(node: str):
        async def probe(state: DecisionState, _ctx: NodeContext) -> dict:
            return {
                "llm_calls": [
                    LLMCallRecord(
                        node=node,
                        profile="quick",
                        provider="stub",
                        model="stub",
                        prompt_hash=node,
                        raw_response="{}",
                        usd=0.01,
                    )
                ]
            }

        return probe

    branches = [(f"n{i}", recorder(f"n{i}")) for i in range(4)]
    graph = build_fanout_graph(("start", _passthrough), branches, ("end", _passthrough), ctx)
    result = DecisionState.model_validate(await graph.ainvoke(_state(ctx)))

    assert len(result.llm_calls) == 4
    assert {r.node for r in result.llm_calls} == {"n0", "n1", "n2", "n3"}
    # If records were lost, spend would read low and the Stage 5 budget check
    # would pass a run that actually blew through it.
    assert round(sum(r.usd for r in result.llm_calls), 4) == 0.04


async def test_errors_accumulate_rather_than_replace(ctx: NodeContext) -> None:
    """Failure direction is HOLD, which requires knowing about every failure."""

    def failer(node: str):
        async def probe(state: DecisionState, _ctx: NodeContext) -> dict:
            return {"errors": [NodeError(node=node, kind="timeout", message="stub")]}

        return probe

    branches = [(f"n{i}", failer(f"n{i}")) for i in range(3)]
    graph = build_fanout_graph(("start", _passthrough), branches, ("end", _passthrough), ctx)
    result = DecisionState.model_validate(await graph.ainvoke(_state(ctx)))

    assert len(result.errors) == 3
    reason = result.degraded_reason()
    assert reason is not None and reason.count("timeout") == 3


async def test_linear_graph_threads_state_in_order(ctx: NodeContext) -> None:
    """Sequential nodes still see upstream patches -- the ordinary case."""
    graph = build_linear_graph(
        [("first", fanout_probe("first")), ("second", fanout_probe("second"))],
        ctx,
    )
    result = DecisionState.model_validate(await graph.ainvoke(_state(ctx)))
    assert result.notes == ["first", "second"]
