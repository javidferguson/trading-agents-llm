"""Node 15 -- write ``DecisionState`` JSONL and ``proposal.json``. No LLM (§3).

Two artefacts, two jobs, and §1 is explicit that they must not be merged:

* **``DecisionState`` JSONL is the source of truth for replay and audit.** The
  LangGraph checkpointer exists only for crash-resume within a live run.
  *"Two mechanisms, two jobs. Do not merge them, and do not let a future
  refactor 'simplify' by deleting the JSONL."*
* **``proposal.json`` is the file the §0 split communicates through.** It is
  what `execute` reads, and the only thing it reads. One decision per file,
  named by run.

Failure direction is HOLD (§5). If any node errored, or the trader degraded, the
decision written here is a HOLD carrying the reason -- never a trade.

**Stage 6 moved the promotion out of this node.** ``compliance`` (node 14) now
promotes the verdict, sizes it and may veto it, so this node writes what that
node decided rather than deciding again. The promotion code stays here for the
graphs that have no compliance node -- and only for those -- because two nodes
independently deriving a ``FinalDecision`` is two chances to derive different
ones, and the one that reached a human would be whichever ran last.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from ...config import load_yaml
from ...context import NodeContext
from ...models.state import (
    DecisionState,
    FinalDecision,
    NodePatch,
    TraderProposal,
    _final_from_verdict,
)

logger = logging.getLogger(__name__)

NODE = "persist"

#: Fallback if portfolio-intent.yaml does not say. Short on purpose: a stale
#: proposal must not be executable, and §9 refuses outright rather than
#: prompting.
DEFAULT_TTL_HOURS = 18


async def persist(state: DecisionState, ctx: NodeContext) -> NodePatch:
    """Write both artefacts, promoting the proposal only if nothing else has."""
    intent = load_yaml("portfolio-intent.yaml")
    cadence = intent.get("cadence") or {}
    risk = intent.get("risk") or {}
    ttl_hours = float(cadence.get("proposal_ttl_hours", DEFAULT_TTL_HOURS))

    proposal = state.trader_proposal
    degraded = bool(state.errors) or proposal is None or proposal.parse_failed

    if state.final_decision is not None:
        # The compliance node already promoted, sized and checked. Writing is
        # all that is left -- and re-deriving the decision here would discard
        # its veto, which is the one thing in this pipeline that must not be
        # overridable by a later node.
        paths = _write(state, state.final_decision, ctx)
        return {"notes": [
            f"{NODE}: {state.final_decision.action} -> {paths['proposal'].name}"
        ]}

    if proposal is None:
        # Nothing to promote. Build the safe decision directly rather than
        # writing nothing: a run that produced no file looks identical to a run
        # that never happened.
        proposal = TraderProposal.degraded(
            state.degraded_reason() or "no trader proposal was produced"
        )

    expires_at = datetime.now() + timedelta(hours=ttl_hours)
    stop_pct = risk.get("default_stop_pct")

    # The fund manager decides when there is one. At Stage 3 there was not, and
    # the trader's proposal was promoted directly -- that shim stays for the
    # `--slice` shape, but a real verdict always wins.
    if state.risk_verdict is not None:
        decision = _final_from_verdict(
            state.risk_verdict, state.symbol,
            expires_at=expires_at, stop_loss_pct=stop_pct, degraded=degraded,
        )
    else:
        decision = FinalDecision.from_proposal(
            proposal, state.symbol,
            expires_at=expires_at, stop_loss_pct=stop_pct, degraded=degraded,
        )

    if degraded and decision.action != "HOLD":
        # Belt and braces. from_proposal copies the action, and a degraded run
        # must never carry a trade to the gate (§5).
        decision = decision.model_copy(update={
            "action": "HOLD",
            "target_weight_pct": 0.0,
            "conviction": 0.0,
            "rationale": f"degraded: {state.degraded_reason() or 'unknown'}",
        })

    final_state = state.model_copy(update={"final_decision": decision})
    paths = _write(final_state, decision, ctx)

    return {
        "final_decision": decision,
        "notes": [f"{NODE}: {decision.action} -> {paths['proposal'].name}"],
    }


def _write(state: DecisionState, decision: FinalDecision, ctx: NodeContext) -> dict:
    settings = ctx.settings

    journal_dir = settings.journal_dir
    journal_dir.mkdir(parents=True, exist_ok=True)
    jsonl = journal_dir / f"decisions_{state.as_of:%Y%m%d}.jsonl"
    with jsonl.open("a") as handle:
        handle.write(json.dumps(state.model_dump(mode="json"), default=str) + "\n")

    proposals_dir = settings.proposals_dir
    proposals_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = proposals_dir / f"{state.symbol}_{state.run_id}.json"
    body: dict = {
        "run_id": state.run_id,
        "as_of": state.as_of.isoformat(),
        "mode": state.mode.value,
        "config_hash": state.config_hash,
        "prompt_pack_version": state.prompt_pack_version,
        "decision": decision.model_dump(mode="json"),
    }

    # §9's middle layer. `execute` reads the plan for the share count and reads
    # `decision` for the text a human approves; both are needed and neither is
    # derivable from the other. Absent on a graph with no compliance node,
    # which `execute` must therefore treat as "nothing sized", not "size it
    # yourself".
    if state.order_plan is not None:
        body["order_plan"] = state.order_plan.model_dump(mode="json")

    # Not the whole book -- the two facts that make the plan auditable without
    # reopening data/portfolio.yaml, which is overwritten on every re-mark.
    if state.portfolio is not None:
        body["book"] = {
            "as_of": state.portfolio.as_of.isoformat(),
            "equity": round(state.portfolio.equity, 2),
            "cash": round(state.portfolio.cash, 2),
            "position_count": state.portfolio.position_count,
            "source": state.portfolio.source,
        }

    proposal_path.write_text(json.dumps(body, indent=2, default=str))

    logger.info("wrote %s and appended %s", proposal_path, jsonl)
    return {"proposal": proposal_path, "jsonl": jsonl}
