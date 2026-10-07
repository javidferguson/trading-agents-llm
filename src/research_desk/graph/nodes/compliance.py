"""Node 14 -- ``compliance``. **Deterministic veto vs PortfolioIntent. No LLM (§3).**

The second key. Everything upstream of this node was advice, including the fund
manager's verdict; this node turns a weight into a share count and then decides
whether the order goes out at all.

> *"Any violation blocks the order regardless of what the fund manager decided.
> Two-key system: LLM proposes, Python vetoes. This is where trust comes from."*
> (§9)

Three things it does, in this order, because the order is the design:

1. **Promote** the fund manager's verdict (or the trader's proposal, when
   running a shorter graph) to a ``FinalDecision``. ``persist`` did this at
   Stages 3-5 because nothing else existed; it happens here now, so that the
   decision compliance checks is the same object ``persist`` writes.
2. **Size** it -- ``intent/sizing.py``, the minimum of four caps.
3. **Check** the post-trade state -- ``intent/compliance.py``, which may veto.

The node never raises. A failure here produces a HOLD carrying the reason (§5),
because the failure direction of a compliance layer must be "no trade" and a
crashed veto is indistinguishable from an absent one.

**It reaches no LLM and no broker.** The only I/O is the EDGAR filing history
behind the earnings estimate, which is cached, and a read of the decision
journal for the cadence count. That is what makes §9's gate -- *"an OrderPlan is
produced with zero IB contact"* -- true by construction rather than by luck.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from ...context import NodeContext
from ...intent import compliance as rules
from ...intent.earnings import estimate_next_report
from ...intent.engine import compute_gaps, load_intent, load_portfolio, sector_map
from ...intent.sizing import size_order
from ...models.state import (
    DecisionState,
    FinalDecision,
    NodeError,
    NodePatch,
    TraderProposal,
    _final_from_verdict,
)

logger = logging.getLogger(__name__)

NODE = "compliance"


def _promote(state: DecisionState, *, expires_at: datetime,
             stop_pct: float | None, degraded: bool) -> FinalDecision:
    """The fund manager's verdict when there is one, the trader's otherwise.

    The ``--slice`` and ``--research`` graphs stop before the risk committee, so
    the Stage 3 shim stays reachable. A real verdict always wins.
    """
    if state.risk_verdict is not None:
        return _final_from_verdict(
            state.risk_verdict, state.symbol,
            expires_at=expires_at, stop_loss_pct=stop_pct, degraded=degraded,
        )

    proposal = state.trader_proposal or TraderProposal.degraded(
        state.degraded_reason() or "no trader proposal was produced"
    )
    return FinalDecision.from_proposal(
        proposal, state.symbol,
        expires_at=expires_at, stop_loss_pct=stop_pct, degraded=degraded,
    )


def _hold(
    decision: FinalDecision, reason: str, *, degraded: bool
) -> FinalDecision:
    """Force a decision to HOLD, keeping everything that explains why.

    ``degraded`` is the distinction that matters to ``execute``, which reads
    the flag to tell *"that a HOLD was a failure rather than a judgement"*
    (§4). Those are the two reasons a HOLD arrives here and they are not the
    same thing:

    * a node failed, so there is no decision -> ``degraded=True``
    * **compliance vetoed, so the decision is "no" -> ``degraded=False``**

    The first live Stage 6 run got this wrong and reported a correct veto as
    "DEGRADED -- this HOLD is a failure, not a judgement", which is precisely
    backwards: a veto is the one place in this pipeline where the system is
    working exactly as designed. The plan's own words for it are *"this is
    where trust comes from"*, and a trustworthy veto does not file itself as a
    malfunction.
    """
    return decision.model_copy(update={
        "action": "HOLD",
        "target_weight_pct": 0.0,
        "conviction": 0.0,
        "rationale": reason,
        "degraded": degraded,
    })


async def compliance(state: DecisionState, ctx: NodeContext) -> NodePatch:
    """Size the decision, check it, and veto it if any rule fails."""
    try:
        intent = load_intent()
        portfolio = load_portfolio()
    except Exception as exc:  # noqa: BLE001 -- degrade to HOLD, never raise (§5)
        # No book, no sizing. A HOLD with the reason beats a traceback, and it
        # beats an unsized order reaching `execute` far more.
        logger.exception("%s: could not load intent or portfolio", NODE)
        return {
            "errors": [NodeError(node=NODE, kind="misconfigured", message=str(exc))],
            "notes": [f"{NODE}: HOLD -- {type(exc).__name__}: {exc}"],
        }

    ttl_hours = float(intent.cadence.proposal_ttl_hours)
    expires_at = datetime.now() + timedelta(hours=ttl_hours)
    degraded = bool(state.errors)

    decision = _promote(
        state, expires_at=expires_at,
        stop_pct=intent.risk.default_stop_pct, degraded=degraded,
    )
    if degraded and decision.action != "HOLD":
        # Belt and braces, as in persist: a degraded run must not carry a trade
        # into sizing, let alone to a human gate (§5).
        decision = _hold(
            decision, f"degraded: {state.degraded_reason() or 'unknown'}",
            degraded=True,
        )

    drift = state.drift or compute_gaps(intent, portfolio, as_of=ctx.as_of)

    # The fresh mark from this run's prefetch, when there is one. The book's
    # stored price is the fallback and sizing handles that itself.
    reference_price = None
    if state.snapshot is not None:
        reference_price = state.snapshot.trend.available().get("close")

    sized = size_order(
        decision, intent, portfolio, drift,
        as_of=ctx.as_of, reference_price=reference_price,
    )

    dollar_adv = None
    if state.snapshot is not None:
        dollar_adv = state.snapshot.risk.available().get("dollar_adv_20d")

    earnings = await estimate_next_report(
        ctx.extras.get("registry"), state.symbol, ctx.as_of
    )

    plan = rules.check(
        sized.plan, decision, intent, portfolio,
        as_of=ctx.as_of,
        sectors=sector_map(),
        dollar_adv=dollar_adv,
        earnings=earnings,
        prior_decisions_today=rules.decisions_today(
            ctx.settings.journal_dir, ctx.as_of,
            # Re-deciding this symbol supersedes its own earlier proposal. See
            # decisions_today -- counting runs instead vetoed the first live
            # Stage 6 run on three re-runs of the symbol it was deciding.
            exclude_symbol=state.symbol,
        ),
    )

    # A vetoed order is a HOLD in the artefact `execute` reads, not a BUY with
    # a blocked plan attached. Both objects go to the journal; only one of them
    # is what the next process acts on, and it must not be able to read a BUY
    # out of a run that Python refused.
    if plan.blocked:
        decision = _hold(
            decision,
            "blocked by compliance: " + "; ".join(
                v.message for v in plan.blocking
            ),
            # A veto is a judgement, not a failure. See _hold.
            degraded=degraded,
        )

    notes = [
        f"{NODE}: {plan.action} {plan.quantity:+d} shares"
        f" (target {sized.target_weight_pct:.2f}%, bound by "
        f"{sized.binding_cap or 'nothing'})"
        if plan.quantity else
        f"{NODE}: no order -- {plan.skip_reason or 'nothing to size'}"
    ]
    for violation in plan.violations:
        notes.append(f"{NODE}: {violation.render()}")

    return {
        "final_decision": decision,
        "order_plan": plan,
        "portfolio": portfolio,
        "drift": drift,
        "notes": notes,
    }


__all__ = ["compliance"]
