"""Graph assembly. **This is the only module in the repo that imports langgraph.**

That rule is enforced by ``tests/test_layering.py`` and it exists because the
value of LangGraph here is narrow and easy to over-draw on. It buys durable
node-level resume and free span boundaries. It is *not* the place to put agent
loops, tool selection, or the human gate.

Three things stay out, permanently (architecture §1, §15.4):

* ``langchain_core`` chat models -- our own router owns model calls, so that
  Ollama's ``format=<json_schema>`` stays directly reachable.
* ``ToolNode`` and ReAct loops -- there is no tool-calling in this design.
* ``interrupt()`` -- the human gate is a file read in a *separate process*. A
  LangGraph interrupt would collapse ``decide`` and ``execute`` into one event
  loop, which is the exact tangle the §0 split exists to avoid.

The checkpointer is for crash-resume within a single live run and nothing else.
``DecisionState`` JSONL remains the source of truth for replay and audit.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from ..context import NodeContext
from ..models.state import DecisionState, NodePatch
from ..obs.tracing import traced_node

logger = logging.getLogger(__name__)

#: A node body, in the signature architecture §1 makes mandatory.
NodeFn = Callable[[Any, NodeContext], Awaitable[NodePatch]]

#: ``(name, fn)``.
NodeSpec = tuple[str, NodeFn]


def _bind(name: str, fn: NodeFn, ctx: NodeContext) -> Callable[[Any], Awaitable[NodePatch]]:
    """Turn ``node(state, ctx)`` into the one-arg callable LangGraph wants.

    Tracing is applied here rather than in the node module so that node bodies
    import neither langgraph nor langfuse.
    """
    traced = traced_node(name)(fn)

    async def run(state: Any) -> NodePatch:
        return await traced(state, ctx)

    run.__name__ = name
    return run


def _saver(checkpoint: bool) -> InMemorySaver | None:
    """Crash-resume only.

    ``InMemorySaver`` is right for Stage 0: it makes resume work within a
    process without adding a database. A durable saver becomes worth it once a
    run is long enough that losing it to a crash hurts -- which, at ~30 s per
    Ollama call, is Stage 4.
    """
    return InMemorySaver() if checkpoint else None


def build_linear_graph(
    nodes: Sequence[NodeSpec],
    ctx: NodeContext,
    *,
    state_type: type = DecisionState,
    checkpoint: bool = False,
) -> Any:
    """Compile ``START -> n0 -> n1 -> ... -> END``."""
    if not nodes:
        raise ValueError("A graph needs at least one node.")

    graph = StateGraph(state_type)
    for name, fn in nodes:
        graph.add_node(name, _bind(name, fn, ctx))

    graph.add_edge(START, nodes[0][0])
    for (before, _), (after, _) in zip(nodes, nodes[1:]):
        graph.add_edge(before, after)
    graph.add_edge(nodes[-1][0], END)

    return graph.compile(checkpointer=_saver(checkpoint))


def build_fanout_graph(
    before: NodeSpec,
    parallel: Sequence[NodeSpec],
    after: NodeSpec,
    ctx: NodeContext,
    *,
    state_type: type = DecisionState,
    checkpoint: bool = False,
) -> Any:
    """Compile ``START -> before -> {parallel...} -> after -> END``.

    This is the shape Stage 4 needs for the four-analyst fan-out, and it is here
    at Stage 0 because of the trap it carries: **parallel nodes appending to the
    same list silently overwrite each other unless the field has an add-reducer**
    (architecture §4). Nothing raises; you just lose three of four records.

    ``tests/graph/test_reducers.py`` runs this shape and asserts every branch's
    record survives, so the trap is caught now rather than at Stage 4.
    """
    if not parallel:
        raise ValueError("A fan-out needs at least one parallel node.")

    graph = StateGraph(state_type)
    for name, fn in (before, *parallel, after):
        graph.add_node(name, _bind(name, fn, ctx))

    graph.add_edge(START, before[0])
    for name, _ in parallel:
        graph.add_edge(before[0], name)
        graph.add_edge(name, after[0])
    graph.add_edge(after[0], END)

    return graph.compile(checkpointer=_saver(checkpoint))


def build_decision_graph(ctx: NodeContext, *, checkpoint: bool = False) -> Any:
    """The Stage 3 vertical slice: ``prefetch -> market_analyst -> trader -> persist``.

    Three nodes plus a writer, one of them not an LLM. The plan is emphatic
    about why this shape comes first:

    > *"This is the highest-value de-risking step in the plan and it is worth
    > protecting from the temptation to build stage 4 first. If market analyst
    > -> trader does not produce something sane, more agents will not fix it;
    > they will produce a more confident version of the same nonsense."*

    Stage 4 inserts the three remaining analysts as a parallel fan-out between
    prefetch and the researchers, which is why ``build_fanout_graph`` already
    exists and is already tested for the reducer bug.
    """
    from .nodes.market_analyst import market_analyst
    from .nodes.persist import persist
    from .nodes.prefetch import prefetch
    from .nodes.trader import trader

    return build_linear_graph(
        [
            ("prefetch", prefetch),
            ("market_analyst", market_analyst),
            ("trader", trader),
            ("persist", persist),
        ],
        ctx,
        checkpoint=checkpoint,
    )
