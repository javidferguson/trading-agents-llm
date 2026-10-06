"""Node 2 -- the market analyst. **Intent-blind on purpose (§8).**

> *"Do not tell the fundamentals analyst 'we are bullish on AI
> infrastructure' before it reads the 10-K. The entire value of a bear
> researcher evaporates if every upstream report was primed with your thesis.
> Intent enters at the Trader node and no earlier. This deserves a code comment
> so nobody 'helpfully' fixes it later."*

This is that comment. The prompt built here contains the rendered snapshot and
nothing else -- no portfolio intent, no themes, no target weights, no mention
of what the desk already holds. ``tests/test_intent_blindness.py`` fails the
build if intent text reaches an analyst prompt.
"""

from __future__ import annotations

import logging

from ...context import NodeContext
from ...llm.structured import structured
from ...metrics.render import render_snapshot
from ...models.state import AnalystReport, DecisionState, NodeError, NodePatch
from ...prompts import load_prompt

logger = logging.getLogger(__name__)

NODE = "market_analyst"


async def market_analyst(state: DecisionState, ctx: NodeContext) -> NodePatch:
    """Read the snapshot and produce one ``AnalystReport``."""
    router = ctx.extras.get("router")
    if router is None:
        return {"errors": [NodeError(node=NODE, kind="misconfigured",
                                     message="no LLM router on the context")]}
    budget = ctx.extras.get("budget")
    if budget is not None and (spent := budget.exhausted()):
        return {
            "errors": [NodeError(node=NODE, kind="budget",
                                 message=f"budget exhausted before this node: {spent}")],
            "notes": [f"{NODE}: skipped, budget exhausted ({spent})"],
        }

    if state.snapshot is None:
        return {
            "errors": [NodeError(node=NODE, kind="no_snapshot",
                                 message="prefetch produced no snapshot")],
            "analyst_reports": {
                "market": AnalystReport.degraded(
                    "no market snapshot was available", kind="market")
            },
        }

    result = await structured(
        router, NODE, AnalystReport,
        [
            {"role": "system", "content": load_prompt(NODE)},
            # The ONLY thing after the system prompt is facts. See the module
            # docstring before adding anything else here.
            {"role": "user", "content": render_snapshot(state.snapshot)},
        ],
        degraded_fields={"kind": "market"},
    )

    if budget is not None:
        budget.record(calls=len(result.records), usd=result.usd)

    patch: NodePatch = {
        "analyst_reports": {"market": result.value},
        "llm_calls": result.records,
        "notes": [
            f"{NODE}: {result.value.stance} @ {result.value.confidence:.2f}"
            + (" (degraded)" if result.degraded else "")
        ],
    }
    if result.degraded:
        patch["errors"] = [NodeError(
            node=NODE, kind="degraded",
            message=result.value.degraded_reason or "unknown",
        )]
    return patch
