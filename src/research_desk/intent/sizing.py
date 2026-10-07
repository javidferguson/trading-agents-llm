"""Weight -> shares. Deterministic, and the **minimum of four caps** (§9).

    target_w        = min(requested, cap_intent, cap_position, cap_risk)
    delta_notional  = equity * target_w/100 - pf.position_value(symbol)

    requested       = the approved FinalDecision.target_weight_pct
    cap_intent      = this symbol's target from compute_gaps()
    cap_position    = intent.risk.max_position_pct
    cap_risk        = intent.risk.per_trade_risk_pct / stop_dist

Then reject dust (``< min_trade_usd``), clamp to ``max_order_shares`` and 90% of
buying power.

**This deviates from §9's formula, deliberately.** §9 writes the first term as
``conviction_w = target_weight_pct * conviction``, and that is how Stage 6
shipped. The first live run showed what it does: the drift table offered TSM
5.83%, the trader proposed 5.83% at 0.75, the fund manager **adjusted down to
4.5%** for overbought risk at 0.65, and sizing then multiplied to 2.93% and
ordered **three shares**. The reduction for risk was applied twice -- once as a
judgement, once by the formula -- and with debater confidence typically 0.6-0.8
the systematic effect is a book that converges to about two thirds of its stated
theme targets and never closes a gap.

**The deciding argument is not the double-damping. It is that ``conviction``
already has a different job, and sizing is not it.** ``prompts/trader.md`` tells
the model that conviction *"is not enthusiasm: it is the probability you would
put on being right"* and that it *"will be measured against realised
outcomes"*; §11 makes the calibration plot -- realized hit rate bucketed by
stated confidence -- *"cheap and usually the most damning diagnostic"*.

A number that is being *measured* must not also be a *control input*. Once
conviction sets position size there is pressure to inflate it -- from prompt
tuning, or from the model's own tendencies -- and the calibration plot then
measures a quantity that was gamed into existence. Goodhart, in the one place
this design can least afford it, because §11's go/no-go rests on that plot.
So conviction is measured and never acted on, and after this change it has
**zero numeric consumers** anywhere in the codebase.

Two supporting reasons. The model already expresses size through
``target_weight_pct`` -- two knobs for one quantity fight each other, which is
exactly what the TSM run was. And ``trader.md`` already described sizing as
*"from your weight, the stop distance and the hard limits, taking the minimum of
four caps"*, so removing the conviction term makes the code match what the model
was told rather than merely resemble it.

**The drift target is a cap, not advice.** ``cap_intent`` is the symbol's target
from ``compute_gaps()``, so an order cannot exceed the gap the drift table
actually shows. Without it channel 2 is advisory: the table says "you are 5pp
light in AVGO" and the model is still free to ask for 10%, which is precisely
the "inventing allocations" §8 says the drift table exists to stop. With it,
"close this gap or don't" is enforced in Python rather than requested in a
prompt.

**The caps apply only to orders that ADD exposure, and this is not a detail.**
A cap whose job is to limit risk must never be able to *increase* the size of a
risk-reducing trade -- and ``min()`` over a reduction does exactly that. A trim
to 5% in a symbol whose ``cap_intent`` is 3.12% would be capped to 3.12% and
sell more than the fund manager approved. So a reducing order takes its
requested target as given and is constrained only by zero and the share limits.

**No model does any of this arithmetic (§2).** A model that names a share count
is sizing in its head; ``TraderProposal`` and ``RiskVerdict`` carry a weight for
exactly that reason.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date

from ..models.intent import PortfolioIntent
from ..models.orders import OrderPlan
from ..models.portfolio import PortfolioSnapshot
from ..models.state import FinalDecision
from .engine import DriftTable

logger = logging.getLogger(__name__)

#: §9: *"clamp to ... 90% of buying power."* The 10% held back is not
#: conservatism for its own sake -- the reference price here is as stale as the
#: bars cache, and a marketable limit order fills at a worse price than the
#: mark it was sized against.
BUYING_POWER_FRACTION = 0.90


@dataclass(frozen=True)
class SizingResult:
    """The sized order, before compliance sees it."""

    plan: OrderPlan
    #: Every cap considered, in §9's order, so "why is this order so small" has
    #: a one-line answer.
    caps_pct: dict[str, float]
    binding_cap: str
    target_weight_pct: float
    #: Notional before the dust and share clamps, for the log.
    raw_delta_usd: float


def _cap_risk_pct(intent: PortfolioIntent, stop_pct: float | None) -> float:
    """``per_trade_risk_pct / stop_dist``, as a position weight in percent.

    With a 0.75% risk budget and an 8% stop, a 9.375% position loses exactly
    0.75% of equity when the stop fills. A tighter stop permits a *larger*
    position for the same risk, which is the point of expressing risk this way
    rather than as a weight.
    """
    distance = (stop_pct if stop_pct and stop_pct > 0 else intent.risk.default_stop_pct)
    return intent.risk.per_trade_risk_pct / (distance / 100.0)


def size_order(
    decision: FinalDecision,
    intent: PortfolioIntent,
    portfolio: PortfolioSnapshot,
    drift: DriftTable,
    *,
    as_of: date,
    reference_price: float | None = None,
) -> SizingResult:
    """Turn a ``FinalDecision`` into a share count. No broker, no model, no network."""
    symbol = decision.symbol.upper()
    equity = portfolio.equity
    current_value = portfolio.position_value(symbol)
    current_pct = portfolio.weight_pct(symbol)
    held = portfolio.get(symbol)

    # The mark. The fresh snapshot price is preferred -- it came from this run's
    # prefetch -- and the book's stored mark is the fallback, which is why a
    # position carries one at all.
    price = reference_price or (held.last_price if held else None)

    row = drift.row(symbol)
    cap_intent = row.target_weight_pct if row is not None else 0.0

    # `requested` is the approved weight itself, not a limit derived from it.
    # It replaced `conviction_w` -- see the module docstring. It belongs in the
    # min() because it is genuinely binding whenever the fund manager asked for
    # less than the caps allow, which is the common case: a cap set that could
    # never be the answer would make `binding_cap` misleading.
    caps: dict[str, float] = {
        "requested": decision.target_weight_pct,
        "cap_position": intent.risk.max_position_pct,
        "cap_risk": _cap_risk_pct(intent, decision.stop_loss_pct),
        "cap_intent": cap_intent,
    }

    def plan(**fields) -> OrderPlan:
        base = {
            "symbol": symbol, "as_of": as_of, "action": decision.action,
            "reference_price": price, "current_weight_pct": current_pct,
            "caps_pct": caps,
        }
        return OrderPlan(**{**base, **fields})

    # --- the cases that produce no order at all -----------------------------

    if decision.action == "HOLD":
        return SizingResult(
            plan=plan(target_weight_pct=current_pct,
                      skip_reason="the decision is HOLD; nothing is sized"),
            caps_pct=caps, binding_cap="", target_weight_pct=current_pct,
            raw_delta_usd=0.0,
        )

    if price is None or price <= 0:
        return SizingResult(
            plan=plan(target_weight_pct=0.0, skip_reason=(
                f"no usable reference price for {symbol} -- neither this run's "
                "snapshot nor the book carries a mark, and a share count cannot "
                "be computed without one")),
            caps_pct=caps, binding_cap="", target_weight_pct=0.0,
            raw_delta_usd=0.0,
        )

    if equity <= 0:
        return SizingResult(
            plan=plan(target_weight_pct=0.0, skip_reason=(
                f"book equity is {equity:,.2f}; there is nothing to size against")),
            caps_pct=caps, binding_cap="", target_weight_pct=0.0,
            raw_delta_usd=0.0,
        )

    # --- the target weight --------------------------------------------------

    reducing = decision.action == "SELL"

    if reducing:
        # Requested, floored at zero. Long-only (§9 "sizing assumes it"), so a
        # SELL cannot cross into a short, and the caps above are deliberately
        # not applied -- see the module docstring.
        target_pct = max(decision.target_weight_pct, 0.0)
        binding = "requested (reducing: risk caps do not apply)"
        if not intent.universe.allow_shorts and target_pct > current_pct:
            return SizingResult(
                plan=plan(target_weight_pct=current_pct, skip_reason=(
                    f"a SELL cannot raise the target from {current_pct:.2f}% to "
                    f"{target_pct:.2f}%. If the intent is to add, the action is BUY")),
                caps_pct=caps, binding_cap="", target_weight_pct=current_pct,
                raw_delta_usd=0.0,
            )
        if held is None:
            return SizingResult(
                plan=plan(target_weight_pct=0.0, skip_reason=(
                    f"{symbol} is not held, and shorts are "
                    f"{'permitted but unsupported by sizing' if intent.universe.allow_shorts else 'not permitted'}"
                    " -- there is nothing to sell")),
                caps_pct=caps, binding_cap="", target_weight_pct=0.0,
                raw_delta_usd=0.0,
            )
    else:
        binding = min(caps, key=lambda name: caps[name])
        target_pct = caps[binding]

    raw_delta = equity * target_pct / 100.0 - current_value

    # A BUY whose caps put the target below the current weight is not a sell:
    # the fund manager asked to add and the caps said there is no room. Saying
    # so is the useful answer; quietly trimming would execute a trade nobody
    # approved.
    if not reducing and raw_delta < 0:
        return SizingResult(
            plan=plan(target_weight_pct=target_pct, skip_reason=(
                f"a BUY was approved but {symbol} is already at "
                f"{current_pct:.2f}% against a capped target of {target_pct:.2f}% "
                f"(bound by {binding}) -- there is no room to add"),
                binding_cap=binding),
            caps_pct=caps, binding_cap=binding, target_weight_pct=target_pct,
            raw_delta_usd=raw_delta,
        )

    # --- shares, and the clamps --------------------------------------------

    # Toward zero, always. Rounding up would put the position above the cap the
    # minimum above was computed to respect.
    shares = int(math.copysign(math.floor(abs(raw_delta) / price), raw_delta))

    closing_out = reducing and target_pct == 0.0 and held is not None
    if closing_out:
        # Exactly flat, rather than one share short of it because of rounding.
        shares = -int(held.quantity)

    if shares == 0:
        return SizingResult(
            plan=plan(target_weight_pct=target_pct, binding_cap=binding,
                      skip_reason=(
                          f"the {abs(raw_delta):,.0f} USD gap is less than one "
                          f"share at {price:,.2f}")),
            caps_pct=caps, binding_cap=binding, target_weight_pct=target_pct,
            raw_delta_usd=raw_delta,
        )

    notional = abs(shares) * price

    # Dust. A full close is exempt: refusing to exit a 300 USD position because
    # 300 < min_trade_usd would strand it in the book permanently, occupying a
    # max_positions slot it does not deserve.
    if notional < intent.risk.min_trade_usd and not closing_out:
        return SizingResult(
            plan=plan(target_weight_pct=target_pct, binding_cap=binding,
                      skip_reason=(
                          f"{notional:,.0f} USD is below min_trade_usd of "
                          f"{intent.risk.min_trade_usd:,.0f}")),
            caps_pct=caps, binding_cap=binding, target_weight_pct=target_pct,
            raw_delta_usd=raw_delta,
        )

    clamps: list[str] = []

    if abs(shares) > intent.risk.max_order_shares:
        shares = int(math.copysign(intent.risk.max_order_shares, shares))
        clamps.append(f"max_order_shares {intent.risk.max_order_shares}")

    if shares > 0:
        affordable = portfolio.cash * BUYING_POWER_FRACTION
        if shares * price > affordable:
            shares = int(affordable // price)
            clamps.append(
                f"{BUYING_POWER_FRACTION:.0%} of {portfolio.cash:,.0f} USD cash"
            )
            if shares == 0:
                return SizingResult(
                    plan=plan(target_weight_pct=target_pct, binding_cap=binding,
                              skip_reason=(
                                  f"{BUYING_POWER_FRACTION:.0%} of "
                                  f"{portfolio.cash:,.0f} USD cash does not cover "
                                  f"one share at {price:,.2f}")),
                    caps_pct=caps, binding_cap=binding,
                    target_weight_pct=target_pct, raw_delta_usd=raw_delta,
                )

    notional = abs(shares) * price
    if clamps:
        binding = f"{binding} then clamped by {' and '.join(clamps)}"

    return SizingResult(
        plan=plan(
            quantity=shares,
            estimated_notional=round(notional, 2),
            target_weight_pct=round(
                100.0 * (current_value + shares * price) / equity, 6
            ),
            binding_cap=binding,
        ),
        caps_pct=caps, binding_cap=binding, target_weight_pct=target_pct,
        raw_delta_usd=raw_delta,
    )


__all__ = ["BUYING_POWER_FRACTION", "SizingResult", "size_order"]
