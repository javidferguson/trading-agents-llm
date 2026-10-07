"""The Python veto. Channel 4, and **the second key** (§9).

> *"Compliance -- post-trade-state checks against max_position_pct,
> max_sector_pct, min_cash_pct, max_gross_exposure, max_positions, universe
> membership, earnings blackout, max_decisions_per_day, ADV participation. **Any
> violation blocks the order regardless of what the fund manager decided.**
> Two-key system: LLM proposes, Python vetoes. This is where trust comes from."*

Two properties make that sentence true rather than aspirational.

**Checks run on the POST-TRADE state, not the current one.** A book at 9% in a
symbol with a 10% cap passes every check; it is the book *after* the order that
would breach it. Checking the current state would pass every order ever
proposed, which is the failure mode that makes a compliance layer decorative.
``post_trade()`` applies the order to a copy of the book and the checks read
that.

**Nothing here consults a model, and no model output can switch a check off.**
The only inputs are the sized plan, the intent, the book, and three facts
measured elsewhere (ADV, the earnings estimate, today's decision count). The
fund manager's ``decision`` field is read for *reporting* and is never a
condition.

**One check warns rather than blocks**, and exactly one: the earnings blackout,
because with no earnings calendar wired up its input is an estimate from EDGAR
filing cadence. ``intent/earnings.py`` explains the choice at length. Every
other rule reads a number that is actually known, so every other rule blocks.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from ..models.intent import PortfolioIntent
from ..models.orders import OrderPlan, Violation
from ..models.portfolio import PortfolioSnapshot, Position
from ..models.state import FinalDecision
from .earnings import EarningsEstimate, blackout_reason

logger = logging.getLogger(__name__)

#: Float slack on every percentage comparison. Sizing computes a target to the
#: cap and floating point then puts the result a femtopercent over it; blocking
#: an order for that would be a correct check reporting a false violation.
EPSILON_PCT = 1e-6


def post_trade(
    portfolio: PortfolioSnapshot, symbol: str, shares: int, price: float
) -> PortfolioSnapshot:
    """The book as it would be if this order filled at ``price``.

    Cash moves against the trade and the position moves with it. Commission is
    not modelled: ``ib.whatIfOrder()`` returns the real figure at Stage 7 and a
    guess here would make the cash check wrong in a way that looks precise.
    """
    key = symbol.strip().upper()
    if shares == 0:
        return portfolio

    positions = [p for p in portfolio.positions if p.symbol != key]
    held = portfolio.get(key)
    quantity = (held.quantity if held else 0.0) + shares

    if abs(quantity) > 1e-9:
        if held is not None and shares > 0:
            # Weighted average cost, the way a broker reports it. Only an ADD
            # changes the basis; a partial sale leaves it alone.
            basis = (held.quantity * held.avg_cost + shares * price) / quantity
        elif held is not None:
            basis = held.avg_cost
        else:
            basis = price
        positions.append(Position(
            symbol=key, quantity=quantity, avg_cost=max(basis, 0.0),
            last_price=price,
            marked_on=held.marked_on if held else portfolio.as_of,
        ))

    return portfolio.model_copy(update={
        "positions": positions,
        "cash": portfolio.cash - shares * price,
    })


def decisions_today(
    journal_dir: Path, as_of: date, *, exclude_symbol: str | None = None
) -> int:
    """How many **distinct symbols** already have an actionable decision for ``as_of``.

    Two deliberate choices, both found by running this against a real journal.

    **A HOLD does not consume the budget.** ``max_decisions_per_day`` exists to
    stop the desk churning the book, and a day of HOLDs has churned nothing.
    Reading ten HOLDs and then being refused the one trade you wanted would be
    the cadence limit working exactly backwards.

    **Distinct symbols, not runs, and never the symbol being decided now.**
    This counted runs first, and the first live Stage 6 run was vetoed with "8
    actionable decisions already written" -- of which *three were re-runs of
    the symbol under decision*. Counting runs makes iterating on one name burn
    the day's budget, which collides head-on with the plan's third stopping
    rule: *"Anywhere, if `decide` stops being fun to iterate on. A 3-8 minute
    local run that you dread is a project that quietly ends."* A re-run
    supersedes its own earlier proposal rather than adding a decision to the
    day, and what the limit is actually protecting is how many *positions* the
    desk touches.

    Returns 0 and logs when the journal cannot be read. A missing journal means
    no decisions have been written, which is the honest answer; an unreadable
    one is worth a line in the log but must not take down the run.
    """
    path = journal_dir / f"decisions_{as_of:%Y%m%d}.jsonl"
    if not path.exists():
        return 0

    skip = (exclude_symbol or "").strip().upper()
    symbols: set[str] = set()
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A truncated last line is normal if a run was killed
                # mid-write. Skip it rather than discarding the whole count.
                continue
            final = row.get("final_decision") or {}
            if final.get("action") not in {"BUY", "SELL"}:
                continue
            symbol = str(final.get("symbol") or row.get("symbol") or "").upper()
            if symbol and symbol != skip:
                symbols.add(symbol)
    except OSError as exc:
        logger.warning("could not read %s for the cadence check: %s", path, exc)
        return 0
    return len(symbols)


def _sector_exposure_pct(
    book: PortfolioSnapshot, sectors: dict[str, str], sector: str
) -> float:
    """Gross exposure to one sector, as a percent of equity.

    Absolute values, like ``gross_value``: a long and a short in the same
    sector are two positions to unwind, not zero exposure.
    """
    total = sum(
        abs(p.market_value) for p in book.positions
        if sectors.get(p.symbol) == sector
    )
    equity = book.equity
    return 0.0 if equity <= 0 else 100.0 * total / equity


def check(
    plan: OrderPlan,
    decision: FinalDecision,
    intent: PortfolioIntent,
    portfolio: PortfolioSnapshot,
    *,
    as_of: date,
    sectors: dict[str, str] | None = None,
    dollar_adv: float | None = None,
    earnings: EarningsEstimate | None = None,
    prior_decisions_today: int = 0,
) -> OrderPlan:
    """Check ``plan`` and return it with violations attached.

    A blocking violation zeroes the quantity before the plan is returned --
    ``OrderPlan`` refuses to hold shares alongside a block, so the invariant is
    enforced by the schema rather than by the caller remembering.
    """
    sectors = sectors or {}
    violations: list[Violation] = []
    risk = intent.risk
    symbol = plan.symbol

    def add(rule: str, message: str, *, limit: float | None = None,
            actual: float | None = None, severity: str = "block") -> None:
        violations.append(Violation(
            rule=rule, severity=severity, message=message,  # type: ignore[arg-type]
            limit=limit, actual=actual,
        ))

    # --- checks that apply to every decision, order or not ------------------
    #
    # Recorded even for a HOLD. "No order was placed" and "no order was placed
    # because the book is three weeks stale" are different facts, and the
    # second one is the one that needs fixing.

    stale = portfolio.staleness_reason(as_of)
    if stale is not None:
        add("portfolio_stale", stale,
            limit=float(portfolio.stale_after_days),
            actual=float(portfolio.staleness_days(as_of)))

    if portfolio.equity <= 0:
        add("book_unusable",
            f"book equity is {portfolio.equity:,.2f}; no weight-based limit can "
            "be evaluated against it",
            actual=round(portfolio.equity, 2))

    if not intent.universe.admits(symbol):
        add("universe_membership",
            f"{symbol} is not in universe.include, or is listed under "
            "universe.exclude. The desk may not trade it regardless of the "
            "quality of the case for it")

    if earnings is not None:
        reason = blackout_reason(
            earnings, as_of,
            days_before=risk.earnings_blackout_days_before,
            days_after=risk.earnings_blackout_days_after,
        )
        if reason is not None:
            # The one non-blocking rule in the file. A CONFIRMED date blocks; an
            # estimate from filing cadence warns. See intent/earnings.py.
            severity = "block" if earnings.is_confirmed else "warn"
            add("earnings_blackout", reason, severity=severity)

    # --- checks that only mean something when shares are changing hands -----

    if plan.quantity == 0:
        return plan.model_copy(update={"violations": violations})

    price = plan.reference_price or 0.0
    shares = plan.quantity

    if shares < 0 and not intent.universe.allow_shorts:
        held = portfolio.get(symbol)
        resulting = (held.quantity if held else 0.0) + shares
        if resulting < -1e-9:
            add("no_shorts",
                f"selling {abs(shares)} shares would leave a position of "
                f"{resulting:,.0f}, and universe.allow_shorts is false",
                actual=resulting)

    if prior_decisions_today >= intent.cadence.max_decisions_per_day:
        add("max_decisions_per_day",
            f"{prior_decisions_today} OTHER symbol(s) already have an actionable "
            f"decision for {as_of.isoformat()}, against a cadence limit of "
            f"{intent.cadence.max_decisions_per_day}. Re-running this symbol does "
            "not count against it; a fourth name does",
            limit=float(intent.cadence.max_decisions_per_day),
            actual=float(prior_decisions_today))

    if dollar_adv is not None and dollar_adv > 0:
        participation = 100.0 * abs(shares) * price / dollar_adv
        if participation > risk.max_adv_participation_pct + EPSILON_PCT:
            add("adv_participation",
                f"{abs(shares)} shares is {participation:.2f}% of the 20-day "
                f"dollar ADV of {dollar_adv:,.0f} USD. A fill you cannot "
                "actually get makes the equity curve fiction",
                limit=risk.max_adv_participation_pct,
                actual=round(participation, 4))
    elif dollar_adv is None:
        add("adv_participation",
            "20-day dollar ADV is unavailable for this symbol, so the "
            "participation limit could not be evaluated",
            severity="warn")

    book = post_trade(portfolio, symbol, shares, price)

    weight = abs(book.weight_pct(symbol))
    if weight > risk.max_position_pct + EPSILON_PCT:
        add("max_position_pct",
            f"{symbol} would be {weight:.2f}% of equity after this order",
            limit=risk.max_position_pct, actual=round(weight, 4))

    sector = sectors.get(symbol)
    if sector:
        exposure = _sector_exposure_pct(book, sectors, sector)
        if exposure > risk.max_sector_pct + EPSILON_PCT:
            add("max_sector_pct",
                f"{sector} would be {exposure:.2f}% of equity after this order",
                limit=risk.max_sector_pct, actual=round(exposure, 4))

    cash_pct = book.cash_pct
    if cash_pct < risk.min_cash_pct - EPSILON_PCT:
        add("min_cash_pct",
            f"cash would fall to {cash_pct:.2f}% of equity "
            f"({book.cash:,.0f} USD) after this order",
            limit=risk.min_cash_pct, actual=round(cash_pct, 4))

    gross_cap = risk.effective_gross_cap_pct(
        allow_shorts=intent.universe.allow_shorts
    )
    gross = book.gross_exposure_pct
    if gross > gross_cap + EPSILON_PCT:
        # Names which limit bound, because the long-only cash floor usually
        # implies a tighter ceiling than max_gross_exposure_pct declares -- see
        # RiskLimits.effective_gross_cap_pct.
        source = (
            "max_gross_exposure_pct"
            if gross_cap == risk.max_gross_exposure_pct
            else f"the {risk.min_cash_pct:g}% cash floor, which caps gross at "
                 f"{gross_cap:g}% in a long-only book"
        )
        add("max_gross_exposure_pct",
            f"gross exposure would be {gross:.2f}% of equity after this order, "
            f"against {source}",
            limit=gross_cap, actual=round(gross, 4))

    if book.position_count > risk.max_positions:
        add("max_positions",
            f"the book would hold {book.position_count} positions after this "
            f"order",
            limit=float(risk.max_positions), actual=float(book.position_count))

    blocking = [v for v in violations if v.blocks]
    if blocking:
        return plan.model_copy(update={
            "violations": violations,
            "quantity": 0,
            "estimated_notional": 0.0,
            "skip_reason": (
                "blocked by " + ", ".join(v.rule for v in blocking)
                + " -- the fund manager approved this order and Python vetoed "
                "it (§9)"
            ),
        })

    return plan.model_copy(update={"violations": violations})


__all__ = ["EPSILON_PCT", "check", "decisions_today", "post_trade"]
