"""Node 9 -- the trader. **The first node that sees portfolio intent (§8).**

It receives the analyst's report, the same facts, and the intent. At Stage 6 it
also receives the drift table from ``intent_engine.compute_gaps()``, which is
the biggest reliability win in the design: it turns an open-ended "how much
should we buy" into a bounded "close this gap or don't", which small models
handle far better. Until then the intent arrives as prose.
"""

from __future__ import annotations

import logging

from ...config import load_yaml
from ...context import NodeContext
from ...llm.structured import structured
from ...metrics.render import render_intent, render_snapshot
from ...models.state import DecisionState, NodeError, NodePatch, TraderProposal
from ...prompts import load_prompt

logger = logging.getLogger(__name__)

NODE = "trader"


def _render_report(state: DecisionState) -> str:
    report = state.analyst_reports.get("market")
    if report is None:
        return "ANALYST REPORT: none was produced.\n"

    lines = [
        "ANALYST REPORT (market)",
        f"  stance     {report.stance}",
        f"  confidence {report.confidence}",
        f"  summary    {' '.join(report.summary.split())}",
        "  key points:",
    ]
    lines += [f"    - {' '.join(point.split())}" for point in report.key_points]
    if report.data_gaps:
        lines.append("  the analyst could not see:")
        lines += [f"    - {gap}" for gap in report.data_gaps]
    if report.parse_failed:
        # Said plainly rather than implied: a degraded report is not a neutral
        # read, and a trader that treats it as one is reasoning from nothing.
        lines.append(
            "  NOTE: this report is DEGRADED -- the analyst produced no usable "
            "output. Treat it as absent, not as neutral evidence."
        )
    return "\n".join(lines) + "\n"


async def trader(state: DecisionState, ctx: NodeContext) -> NodePatch:
    """Decide BUY, SELL or HOLD and produce a ``TraderProposal``."""
    router = ctx.extras.get("router")
    if router is None:
        return {"errors": [NodeError(node=NODE, kind="misconfigured",
                                     message="no LLM router on the context")]}

    facts = render_snapshot(state.snapshot) if state.snapshot is not None else \
        "No market facts were available for this symbol.\n"

    result = await structured(
        router, NODE, TraderProposal,
        [
            {"role": "system", "content": load_prompt(NODE)},
            {"role": "user", "content": (
                f"{_render_report(state)}\n"
                f"{facts}\n"
                f"{render_intent(load_yaml('portfolio-intent.yaml'))}\n"
                f"Decide on {state.symbol}."
            )},
        ],
    )

    patch: NodePatch = {
        "trader_proposal": result.value,
        "llm_calls": result.records,
        "notes": [
            f"{NODE}: {result.value.action} "
            f"{result.value.target_weight_pct:.1f}% @ "
            f"conviction {result.value.conviction:.2f}"
            + (" (degraded)" if result.degraded else "")
        ],
    }
    if result.degraded:
        patch["errors"] = [NodeError(
            node=NODE, kind="degraded",
            message=result.value.degraded_reason or "unknown",
        )]
    return patch
