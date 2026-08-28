"""The Stage 0 exit gate, minus the eyeball.

The gate is *"a two-node toy LangGraph runs and both nodes appear as spans in
Langfuse."* The span half needs a running Langfuse and a human looking at it, so
`make toy-graph` owns that. What is testable in CI is everything underneath: the
node signature, the partial-state patch, the tracing wrapper staying invisible,
and the run degrading rather than failing when Langfuse is absent.
"""

from __future__ import annotations

from datetime import date

import pytest

from research_desk.config import Settings
from research_desk.context import NodeContext
from research_desk.graph.build import build_linear_graph
from research_desk.graph.nodes.toy import toy_prefetch, toy_summarise
from research_desk.models.modes import RunMode
from research_desk.models.state import DecisionState

TOY_NODES = [("toy_prefetch", toy_prefetch), ("toy_summarise", toy_summarise)]


@pytest.fixture
def ctx() -> NodeContext:
    return NodeContext(settings=Settings(), as_of=date(2026, 8, 27))


async def test_two_node_graph_runs_end_to_end(ctx: NodeContext) -> None:
    graph = build_linear_graph(TOY_NODES, ctx)
    result = DecisionState.model_validate(
        await graph.ainvoke(
            DecisionState(run_id=ctx.run_id, symbol="SPY", as_of=ctx.as_of)
        )
    )

    assert len(result.notes) == 2
    assert "prefetch: SPY as_of=2026-08-27" in result.notes[0]
    # The second node saw the first node's patch, which is the whole point of
    # threading state rather than passing arguments.
    assert "1 note(s) seen upstream" in result.notes[1]


async def test_raw_response_is_recorded_from_the_first_call(ctx: NodeContext) -> None:
    """`replay_llm` cannot be backfilled, so the raw text lands from call one."""
    graph = build_linear_graph(TOY_NODES, ctx)
    result = DecisionState.model_validate(
        await graph.ainvoke(DecisionState(run_id=ctx.run_id, symbol="AAPL", as_of=ctx.as_of))
    )

    assert len(result.llm_calls) == 1
    record = result.llm_calls[0]
    assert record.node == "toy_summarise"
    assert record.raw_response, "raw_response must never be empty -- it IS the replay substrate"
    assert "AAPL" in record.raw_response
    assert record.attempt == 1


async def test_runs_untraced_when_langfuse_is_not_configured(ctx: NodeContext) -> None:
    """Tracing is a bonus, never load-bearing.

    Losing 8 minutes of local inference to a telemetry error would be absurd, so
    the absence of Langfuse must be invisible to the graph.
    """
    assert not ctx.settings.tracing_enabled
    graph = build_linear_graph(TOY_NODES, ctx)
    result = await graph.ainvoke(DecisionState(run_id=ctx.run_id, symbol="SPY", as_of=ctx.as_of))
    assert DecisionState.model_validate(result).notes


async def test_nodes_are_plain_callables_outside_a_graph(ctx: NodeContext) -> None:
    """A node reads state, reads ctx, returns a patch. No graph required.

    This is the property that makes nodes unit-testable and replayable, and it
    is why the mandatory signature is worth defending.
    """
    state = DecisionState(run_id=ctx.run_id, symbol="MSFT", as_of=ctx.as_of)

    patch = await toy_prefetch(state, ctx)
    assert isinstance(patch, dict)
    assert set(patch) <= set(DecisionState.model_fields), "a patch may only name real fields"
    # A patch is a patch: the node must not have mutated the state it was given.
    assert state.notes == []


async def test_checkpointer_is_opt_in(ctx: NodeContext) -> None:
    """Crash-resume only, and off unless asked for (architecture §1).

    A thread_id is required once a checkpointer is attached, which is a decent
    proxy for "the checkpointer is really wired up".
    """
    graph = build_linear_graph(TOY_NODES, ctx, checkpoint=True)
    state = DecisionState(run_id=ctx.run_id, symbol="SPY", as_of=ctx.as_of)

    result = await graph.ainvoke(state, config={"configurable": {"thread_id": ctx.run_id}})
    assert DecisionState.model_validate(result).notes


async def test_empty_node_list_is_rejected(ctx: NodeContext) -> None:
    with pytest.raises(ValueError):
        build_linear_graph([], ctx)


def test_replay_modes_cannot_trade() -> None:
    """The property that safety.assert_can_trade reads."""
    assert RunMode.LIVE.can_trade
    assert not RunMode.REPLAY.can_trade
    assert not RunMode.REPLAY_LLM.can_trade
    assert not RunMode.REPLAY.allows_network
    assert RunMode.REPLAY_LLM.uses_recorded_llm
