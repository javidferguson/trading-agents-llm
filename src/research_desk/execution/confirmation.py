"""The human confirmation gate. **There is deliberately no way to turn it off.**

Carried from the ORB+GEX engine (``git show 2005f11:confirmation.py``). The
structure survives unchanged -- a preflight type, ``render_decision``, a CLI
gate, a reject-all gate -- and only the renderer was rewritten, because the
options engine described a strike and an expiration and this one describes a
share count and a weight.

Three choices carried verbatim, each for a reason that cost something:

* **Approval requires typing the ticker symbol, not "y".** A yes/no prompt is
  answered by muscle memory; a symbol has to be read first.
* **A closed stdin declines.** The ``execute`` container runs with
  ``stdin_open``, but a cron, a pipe or a detached shell must produce a refusal
  rather than proceeding unattended.
* **The prompt shows the numbers that will actually be sent**, including the
  ``whatIfOrder`` margin and commission estimate, so there is no gap between
  what was approved and what goes out.

> *"The original ORB bot had no gate at all, and the options scanner's gate was
> INVERTED -- setting ``require_confirmation: false`` still prompted and then
> returned True regardless of the answer."*

That is why ``config.load_config`` refuses a config that sets the flag false,
why ``models.intent.Execution`` refuses to construct one in code, and why there
is no parameter here that disables the prompt.

**What this file adds that the options engine had nowhere to put: the dissent.**
§4 marks ``dissent`` and ``invalidation`` REQUIRED on ``FinalDecision``
specifically because *"both surface in the confirmation prompt"* -- the human
sees the strongest surviving argument against the trade immediately before
approving it. Until now those fields were written to a file nobody read at the
moment of decision.

**And the expiry refusal.** ``portfolio-intent.yaml``: *"A proposal generated
after Tuesday's close must not be executable on Thursday -- render_decision()
refuses outright rather than prompting."* An expired proposal raises before any
prompt is drawn, rather than appearing as a warning inside one that a tired
person can approve anyway.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from ..models.orders import OrderPlan
from ..models.state import FinalDecision

logger = logging.getLogger(__name__)

#: Width of the banner rules. Wide enough for the rationale to wrap sensibly in
#: a default terminal and narrow enough to survive a `docker compose run`.
RULE = 72


class ExpiredProposalError(RuntimeError):
    """The proposal's TTL has passed, so it cannot be executed.

    A distinct exception rather than a ``False`` return, because "the human
    said no" and "this was never offered to the human" are different outcomes
    and `execute` reports them differently.
    """


class StaleBookError(RuntimeError):
    """The book the order was sized against disagrees with the broker.

    Raised instead of prompting: the share count on screen was computed from a
    number now known to be wrong, so there is nothing safe to approve.
    """


@dataclass(frozen=True)
class Preflight:
    """What ``ib.whatIfOrder()`` said. Free, and places nothing.

    §9 requires this to run *before the human sees the prompt*, so the margin
    impact and commission are part of what is being approved rather than a
    surprise afterwards.
    """

    status: str | None = None
    init_margin_change: str | None = None
    maint_margin_change: str | None = None
    equity_with_loan_change: str | None = None
    commission: float | None = None
    commission_currency: str | None = None
    warning: str | None = None

    @classmethod
    def from_order_state(cls, state: Any) -> "Preflight":
        """Build from ib_async's ``OrderState``.

        Every field is read defensively. IB returns these as strings, returns
        the sentinel ``1.7976931348623157e+308`` for "unset" on the commission
        fields, and omits them entirely for some contract types -- and a
        preflight that raises would block an order for a reporting quirk.
        """
        def text(name: str) -> str | None:
            value = getattr(state, name, None)
            if value in (None, ""):
                return None
            return str(value)

        commission = getattr(state, "commission", None)
        # IB's "unset double". Shown as n/a rather than as 1.8e308.
        if isinstance(commission, float) and commission > 1e307:
            commission = None

        return cls(
            status=text("status"),
            init_margin_change=text("initMarginChange"),
            maint_margin_change=text("maintMarginChange"),
            equity_with_loan_change=text("equityWithLoanChange"),
            commission=commission,
            commission_currency=text("commissionCurrency"),
            warning=text("warningText"),
        )

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.init_margin_change, self.maint_margin_change,
             self.commission is not None, self.warning)
        )


class ConfirmationGate(Protocol):
    def confirm(
        self,
        decision: FinalDecision,
        plan: OrderPlan,
        *,
        limit_price: float,
        stop_price: float | None = None,
        preflight: Preflight | None = None,
        quote_source: str | None = None,
        offset_bps: int | None = None,
        session_warning: str | None = None,
    ) -> bool:
        ...


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.2f}"


def _wrap(
    text: str,
    width: int = RULE - 4,
    indent: str = "    ",
    bullet: str | None = None,
) -> list[str]:
    """Wrap prose for the terminal without pulling in textwrap's defaults.

    The rationale, dissent and invalidation are model-written sentences of
    unpredictable length, and the whole point of showing them is that they get
    read -- so they must not run off the edge of the screen.

    ``bullet`` marks the FIRST line only and the rest hang beneath it. Passing
    the bullet as the indent instead repeats it on every line, which turns one
    warning into three that look like three.
    """
    words = " ".join((text or "").split()).split()
    first = f"{indent}{bullet}" if bullet else indent
    hanging = " " * len(first)
    if not words:
        return [f"{first}(none)"]

    lines: list[str] = []
    current = first
    for word in words:
        if len(current) + len(word) + 1 > width and current.strip():
            lines.append(current.rstrip())
            current = hanging
        current += word + " "
    if current.strip():
        lines.append(current.rstrip())
    return lines


def assert_not_expired(decision: FinalDecision, *, now: datetime | None = None) -> None:
    """Refuse an expired proposal before anything is rendered.

    Separate from ``render_decision`` so that `execute` can refuse early --
    there is no reason to connect to a broker for a proposal that cannot be
    acted on.
    """
    moment = now or datetime.now()
    if decision.expires_at <= moment:
        age = moment - decision.expires_at
        hours = age.total_seconds() / 3600
        raise ExpiredProposalError(
            f"This proposal for {decision.symbol} expired at "
            f"{decision.expires_at:%Y-%m-%d %H:%M} -- {hours:.1f} hours ago. "
            "A stale proposal is not executable: the prices and the book it was "
            "sized against have both moved. Re-run `desk decide`."
        )


def render_decision(
    decision: FinalDecision,
    plan: OrderPlan,
    *,
    limit_price: float,
    stop_price: float | None = None,
    preflight: Preflight | None = None,
    quote_source: str | None = None,
    offset_bps: int | None = None,
    session_warning: str | None = None,
    now: datetime | None = None,
) -> str:
    """Format the order for human review. Raises if the proposal has expired.

    ``offset_bps`` is the offset **actually used to price this order**, and it
    is a parameter rather than being read off ``decision`` because those two
    numbers can now differ. ``limit_offset_bps`` moved into
    ``portfolio-intent.yaml`` and is read at execute time, so
    ``FinalDecision.limit_offset_bps`` is the decide-time record -- a proposal
    written yesterday still says 10 while the order goes out at 100. Rendering
    the stale one would print a number that is not what is being sent, on the
    one screen whose entire job is to show what is being sent.
    """
    assert_not_expired(decision, now=now)

    side = "BUY" if plan.quantity > 0 else "SELL"
    shares = abs(plan.quantity)
    notional = shares * limit_price
    # Falls back to the proposal's value only when the caller does not say,
    # which keeps older call sites honest rather than silently printing 0.
    bps = decision.limit_offset_bps if offset_bps is None else offset_bps
    # Signed the way it is applied: a SELL reaches DOWN through the spread.
    reach = f"{'+' if side == 'BUY' else '-'}{bps} bps"

    lines = [
        "=" * RULE,
        f"ORDER PROPOSAL -- REVIEW BEFORE APPROVING      {decision.symbol}",
        "=" * RULE,
        f"  Action       : {side} {shares} share(s) of {decision.symbol}",
        f"  Order type   : marketable LIMIT at {limit_price:,.2f}"
        f"  ({reach})",
        f"  Notional     : {notional:,.2f} USD",
    ]

    if session_warning:
        # High up, not buried below the rationale: it changes whether this
        # order can do anything at all, which is more basic than whether it is
        # a good idea.
        lines.append(f"  >> {session_warning}")

    if stop_price is not None:
        lines.append(
            f"  Protective   : STOP at {stop_price:,.2f} "
            f"(-{decision.stop_loss_pct:.1f}%), attached to this entry"
        )
    elif side == "BUY":
        # Said explicitly rather than omitted: sizing's cap_risk assumed a stop.
        lines.append(
            "  Protective   : NONE -- the risk cap that sized this assumed one"
        )

    if quote_source:
        lines.append(f"  Priced from  : {quote_source}")

    lines += [
        "",
        f"  Weight       : {plan.current_weight_pct:.2f}% -> "
        f"{plan.target_weight_pct:.2f}% of equity"
        + (f"  (bound by {plan.binding_cap})" if plan.binding_cap else ""),
        f"  Conviction   : {decision.conviction:.2f}"
        f"      Horizon: {decision.horizon_days} days",
        f"  Expires      : {decision.expires_at:%Y-%m-%d %H:%M}",
    ]

    if decision.fund_manager_adjustment:
        lines.append("")
        lines.append("  FUND MANAGER ADJUSTED THE PROPOSAL")
        lines += _wrap(decision.fund_manager_adjustment)

    lines += ["", "  WHY", *_wrap(decision.rationale)]

    # The two fields §4 makes required, and the reason it does. The dissent is
    # the single most useful thing on this screen: it is the strongest argument
    # against what you are about to approve, written by the model that approved
    # it.
    lines += ["", "  STRONGEST ARGUMENT AGAINST THIS ORDER", *_wrap(decision.dissent)]
    lines += ["", "  WHAT WOULD PROVE IT WRONG", *_wrap(decision.invalidation)]

    if plan.warnings:
        lines.append("")
        lines.append("  COMPLIANCE WARNINGS (not blocking, but read them)")
        for violation in plan.warnings:
            lines += _wrap(violation.message, bullet="- ")

    if preflight is not None and not preflight.is_empty:
        lines += [
            "",
            "  BROKER PREFLIGHT (whatIf -- nothing has been submitted)",
            f"    Init margin change : {preflight.init_margin_change or 'n/a'}",
            f"    Maint margin change: {preflight.maint_margin_change or 'n/a'}",
            f"    Commission         : {_money(preflight.commission)} "
            f"{preflight.commission_currency or ''}".rstrip(),
        ]
        if preflight.warning:
            lines.append(f"    WARNING            : {preflight.warning}")

    if decision.degraded:
        lines += [
            "",
            "  >> THIS RUN DEGRADED. At least one node produced no usable",
            "     output, so this order rests on less than a full analysis. <<",
        ]
    elif decision.absent_analysts:
        # Not a warning about the order -- a statement about what was read
        # before it. It takes the place of the DEGRADED banner this case used
        # to trip: the run is actionable now, and the human approving it is
        # still owed the fact that an analyst was blind.
        lines += [
            "",
            f"  PARTIAL COVERAGE: {', '.join(decision.absent_analysts)}"
            " had no data for this symbol.",
            "  The decision stands on the remaining analysts. Absent evidence,",
            "  not neutral evidence.",
        ]

    lines.append("=" * RULE)
    return "\n".join(lines)


class CLIConfirmationGate:
    """Prompts on stdin. Requires the ticker symbol to be typed to approve."""

    def confirm(
        self,
        decision: FinalDecision,
        plan: OrderPlan,
        *,
        limit_price: float,
        stop_price: float | None = None,
        preflight: Preflight | None = None,
        quote_source: str | None = None,
        offset_bps: int | None = None,
        session_warning: str | None = None,
    ) -> bool:
        print(render_decision(
            decision, plan, limit_price=limit_price, stop_price=stop_price,
            preflight=preflight, quote_source=quote_source,
            offset_bps=offset_bps, session_warning=session_warning,
        ))

        expected = decision.symbol.upper()
        prompt = (
            f"\nType {expected} to place this order, "
            "or anything else to skip: "
        )

        try:
            answer = input(prompt).strip().upper()
        except (EOFError, KeyboardInterrupt):
            # A closed stdin -- a pipe, a cron, a detached container -- must
            # decline rather than proceed unattended.
            print()
            logger.warning("No interactive input available; declining the order.")
            return False

        if answer == expected:
            logger.info(
                "Order approved by user: %s %+d shares", decision.symbol, plan.quantity
            )
            return True

        logger.info(
            "Order declined by user (entered %r, expected %r)", answer, expected
        )
        return False


class RejectAllGate:
    """Declines everything. Used by ``--dry-run`` and by tests.

    It still *renders*, because the point of a dry run is to see exactly what
    would have been shown.
    """

    def confirm(
        self,
        decision: FinalDecision,
        plan: OrderPlan,
        *,
        limit_price: float,
        stop_price: float | None = None,
        preflight: Preflight | None = None,
        quote_source: str | None = None,
        offset_bps: int | None = None,
        session_warning: str | None = None,
    ) -> bool:
        print(render_decision(
            decision, plan, limit_price=limit_price, stop_price=stop_price,
            preflight=preflight, quote_source=quote_source,
            offset_bps=offset_bps, session_warning=session_warning,
        ))
        print("\nDRY RUN -- declining automatically. Nothing was sent.")
        logger.info("RejectAllGate: declining %s", decision.symbol)
        return False


# --------------------------------------------------------------------------- #
# The batch review. One screen, nothing sent.
# --------------------------------------------------------------------------- #

#: What has to be typed to move from the batch review into the per-order gates.
#: Its own word for the same reason ``REWRITE_WORD`` is: a ticker approves one
#: order, and nothing should let one muscle-memory answer stand in for both a
#: portfolio decision and a trade.
REVIEW_WORD = "REVIEW"


@dataclass(frozen=True)
class BatchRow:
    """One pending order as the review screen shows it.

    Deliberately not a ``FinalDecision``/``OrderPlan`` pair: the screen needs a
    notional estimate and the absent-analyst flag side by side, and building
    that here keeps the renderer from reaching back into the proposal files.
    """

    symbol: str
    action: str
    quantity: int
    notional: float
    absent_analysts: tuple[str, ...] = ()


@dataclass(frozen=True)
class BatchProjection:
    """The book before and after the whole batch, with the caps to read it against.

    Computed by folding ``compliance.post_trade`` over every order, which is
    the same function the per-order veto uses -- so the projection cannot
    disagree with the checks that follow it.
    """

    cash_before: float
    cash_after: float
    cash_pct_after: float
    cash_pct_floor: float
    positions_before: int
    positions_after: int
    positions_cap: int
    gross_before_pct: float
    gross_after_pct: float
    gross_cap_pct: float
    #: sector -> (before_pct, after_pct, cap_pct). Only sectors the batch moves.
    sectors: tuple[tuple[str, float, float, float], ...] = ()

    @property
    def breaches(self) -> tuple[str, ...]:
        """Caps the projected state would cross. **Reported, never enforced here.**

        The per-order ``rules.check`` against the running book is what blocks,
        and it must stay the only thing that does -- two places that can veto
        an order are two places that can disagree about why. This exists so the
        human can stop before typing the first ticker rather than discovering
        it at order four.
        """
        out: list[str] = []
        if self.cash_pct_after < self.cash_pct_floor:
            out.append("min_cash_pct")
        if self.positions_after > self.positions_cap:
            out.append("max_positions")
        if self.gross_after_pct > self.gross_cap_pct:
            out.append("max_gross_exposure_pct")
        out += [f"max_sector_pct ({name})"
                for name, _, after, cap in self.sectors if after > cap]
        return tuple(out)


def render_batch(rows: Sequence[BatchRow], projection: BatchProjection) -> str:
    """The batch review screen. **Approves nothing.**

    This is the only place the portfolio-level effect of a set of orders is
    visible. Every order in the batch was sized and vetoed *in isolation*, so a
    set of individually-compliant orders can still move the book somewhere
    nobody chose -- five names in one theme, or a cash balance no single order
    would have breached. A sequence of per-order prompts cannot show that, by
    construction, because each one only knows about itself.
    """
    total = sum(r.notional for r in rows)
    lines = [
        "",
        "=" * RULE,
        f"BATCH REVIEW -- {len(rows)} order(s), NOTHING HAS BEEN SENT",
        "=" * RULE,
    ]
    for row in rows:
        flag = f"   partial:{','.join(row.absent_analysts)}" if row.absent_analysts else ""
        lines.append(
            f"  {row.symbol:<6} {row.action:<4} {row.quantity:+5d}"
            f"   ~{row.notional:>12,.0f} USD{flag}"
        )

    def row(label: str, before: str, after: str, cap: str) -> str:
        """One aggregate line. Columns are shared so the caps read down."""
        return f"    {label:<20} {before:>12} -> {after:<10} {cap}"

    lines += [
        "",
        "  AGGREGATE EFFECT ON THE BOOK",
        f"    {'total notional':<20} {total:>12,.0f} USD",
        row("cash", f"{projection.cash_before:,.0f}",
            f"{projection.cash_after:,.0f}",
            f"= {projection.cash_pct_after:.1f}% "
            f"(floor {projection.cash_pct_floor:.1f}%)"),
        row("positions", str(projection.positions_before),
            str(projection.positions_after),
            f"(max {projection.positions_cap})"),
        row("gross exposure", f"{projection.gross_before_pct:.1f}%",
            f"{projection.gross_after_pct:.1f}%",
            f"(cap {projection.gross_cap_pct:.1f}%)"),
    ]
    for name, before, after, cap in projection.sectors:
        lines.append(row(name[:20], f"{before:.1f}%", f"{after:.1f}%",
                         f"(cap {cap:.1f}%)"))

    if projection.breaches:
        lines += [
            "",
            "  >> THE PROJECTED STATE WOULD CROSS: "
            + ", ".join(projection.breaches),
            "     Compliance is re-checked per order against the book as it",
            "     actually is, so later orders will be REFUSED. Stop here and",
            "     re-run `desk decide` against a current book instead. <<",
        ]

    lines += [
        "",
        "  Orders are offered in the sequence above, each with its own live",
        "  price and its own prompt. When a cap binds, the earlier orders take",
        "  the headroom. You can decline any one of them.",
        "=" * RULE,
    ]
    return "\n".join(lines)


def confirm_batch_review(rows: Sequence[BatchRow], projection: BatchProjection) -> bool:
    """Show the batch, then ask only to *proceed to the order gates*.

    **This is not an approval of any order**, and the prompt says so. Each
    order still has to be approved by typing its ticker, through the same
    ``CLIConfirmationGate`` a single-order run uses. The alternative -- one
    token approving fifteen orders -- would make the ticker stand for a set
    rather than a trade, which is the property the gate exists to hold.

    A closed stdin stops the batch, like every other prompt in this module.
    """
    print(render_batch(rows, projection))
    prompt = (
        f"\nType {REVIEW_WORD} to go through these {len(rows)} order(s) one at "
        "a time, or anything else to stop: "
    )
    try:
        answer = input(prompt).strip().upper()
    except (EOFError, KeyboardInterrupt):
        print()
        logger.warning("No interactive input available; placing nothing.")
        return False

    if answer == REVIEW_WORD:
        logger.info("Batch review accepted; %d order(s) to review", len(rows))
        return True
    logger.info("Batch stopped at the review screen (entered %r)", answer)
    return False


# --------------------------------------------------------------------------- #
# An order that did not settle. Its own narrow gate.
# --------------------------------------------------------------------------- #

#: What has to be typed to pull a working order. Its own word for the same
#: reason ``REWRITE_WORD`` and ``REVIEW_WORD`` are theirs: the ticker approves
#: an order, and nothing should let one muscle-memory answer both place a trade
#: and retract one.
CANCEL_WORD = "CANCEL"


def render_unsettled(
    symbol: str,
    *,
    order_id: int,
    action: str,
    ordered: int,
    filled: int,
    limit_price: float,
    status: str,
    stop_price: float | None = None,
    timeout_s: float = 0.0,
) -> str:
    """What is still outstanding after the fill wait gave up.

    This exists because the batch used to print "STOPPED" and exit, which read
    as though nothing were outstanding. On 2026-10-09 MRVL order 53 was left
    working with a protective stop attached to it -- so if it filled after the
    process was gone, a stop would arm against a position the book had never
    heard of. Saying so is the minimum; offering to pull it is the point.
    """
    lines = [
        "",
        "=" * RULE,
        f"ORDER {order_id} IS STILL WORKING -- {symbol}",
        "=" * RULE,
        f"  {symbol:<6} {action} {ordered:+d} @ LMT {limit_price:,.2f}"
        f"   filled {filled} of {abs(ordered)}",
        f"  Status: {status}"
        + (f", unchanged for {timeout_s:.0f}s" if timeout_s else ""),
    ]
    if stop_price is not None:
        lines += [
            "",
            f"  A protective stop at {stop_price:,.2f} is attached. If this",
            "  entry fills after this run exits, that stop arms against a",
            "  position the book does not know about.",
        ]
    lines += [
        "",
        "  It is a DAY order, so it works until it fills or the session ends.",
        "  Leaving it is a legitimate choice -- it may simply be about to fill.",
        "=" * RULE,
    ]
    return "\n".join(lines)


def confirm_cancel(symbol: str, order_id: int) -> bool:
    """Ask whether to pull a working order and its stop.

    A narrow gate of its own, like ``confirm_rewrite``. What is being approved
    is a **retraction**, not a trade, and the two should not share a word.

    A closed stdin **leaves the order working** -- the opposite default from
    the order gate, and deliberately so. The order gate declines because doing
    nothing is the safe outcome when nobody is watching. Here the order already
    exists and was already approved by a human; cancelling it unattended would
    be the system reversing a decision on its own.
    """
    prompt = (
        f"\nType {CANCEL_WORD} to pull order {order_id} ({symbol}) and its "
        "stop, or anything else to leave it working: "
    )
    try:
        answer = input(prompt).strip().upper()
    except (EOFError, KeyboardInterrupt):
        print()
        logger.warning(
            "No interactive input available; leaving order %s working.", order_id
        )
        return False

    if answer == CANCEL_WORD:
        logger.info("Cancel approved for %s order %s", symbol, order_id)
        return True
    logger.info(
        "Cancel declined for %s order %s (entered %r)", symbol, order_id, answer
    )
    return False


#: What has to be typed to overwrite the book from the broker. A different word
#: from the ticker on purpose: it is a different action with a different
#: consequence, and reusing the ticker would let one muscle-memory answer do
#: both.
REWRITE_WORD = "REWRITE"


def confirm_rewrite(path: Any) -> bool:
    """Ask before overwriting ``data/portfolio.yaml`` from the broker.

    Its own narrow gate rather than a flag on the order prompt. The order gate
    was declined by this point -- the book disagreed with the account -- and
    what is being approved now is a *file write*, not a trade.
    """
    prompt = (
        f"\nType {REWRITE_WORD} to overwrite {path} from the broker, "
        "or anything else to leave it alone: "
    )
    try:
        answer = input(prompt).strip().upper()
    except (EOFError, KeyboardInterrupt):
        print()
        logger.warning("No interactive input available; leaving the book alone.")
        return False

    if answer == REWRITE_WORD:
        logger.info("Book rewrite approved for %s", path)
        return True
    logger.info("Book rewrite declined (entered %r)", answer)
    return False


__all__ = [
    "CANCEL_WORD",
    "BatchProjection",
    "BatchRow",
    "CLIConfirmationGate",
    "ConfirmationGate",
    "ExpiredProposalError",
    "Preflight",
    "REVIEW_WORD",
    "REWRITE_WORD",
    "RULE",
    "RejectAllGate",
    "StaleBookError",
    "assert_not_expired",
    "confirm_batch_review",
    "confirm_cancel",
    "confirm_rewrite",
    "render_batch",
    "render_decision",
    "render_unsettled",
]
