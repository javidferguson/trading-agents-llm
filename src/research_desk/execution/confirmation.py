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
    now: datetime | None = None,
) -> str:
    """Format the order for human review. Raises if the proposal has expired."""
    assert_not_expired(decision, now=now)

    side = "BUY" if plan.quantity > 0 else "SELL"
    shares = abs(plan.quantity)
    notional = shares * limit_price

    lines = [
        "=" * RULE,
        f"ORDER PROPOSAL -- REVIEW BEFORE APPROVING      {decision.symbol}",
        "=" * RULE,
        f"  Action       : {side} {shares} share(s) of {decision.symbol}",
        f"  Order type   : marketable LIMIT at {limit_price:,.2f}"
        f"  (+{decision.limit_offset_bps} bps)",
        f"  Notional     : {notional:,.2f} USD",
    ]

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
    ) -> bool:
        print(render_decision(
            decision, plan, limit_price=limit_price, stop_price=stop_price,
            preflight=preflight, quote_source=quote_source,
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
    ) -> bool:
        print(render_decision(
            decision, plan, limit_price=limit_price, stop_price=stop_price,
            preflight=preflight, quote_source=quote_source,
        ))
        print("\nDRY RUN -- declining automatically. Nothing was sent.")
        logger.info("RejectAllGate: declining %s", decision.symbol)
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
    "CLIConfirmationGate",
    "ConfirmationGate",
    "ExpiredProposalError",
    "Preflight",
    "REWRITE_WORD",
    "RULE",
    "RejectAllGate",
    "StaleBookError",
    "assert_not_expired",
    "confirm_rewrite",
    "render_decision",
]
