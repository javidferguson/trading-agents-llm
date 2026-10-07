"""``OrderPlan`` and ``Violation`` -- the middle layer of §9's three.

    FinalDecision (broker-agnostic) -> OrderPlan (sizing + compliance) -> ib_async

``FinalDecision`` names a *weight*; ``OrderPlan`` names a *share count*. The
split is what lets the first two layers run with no broker at all, and it is
why nothing in this module imports or mentions a contract, an exchange or a
``ib_async`` type. ``execute`` turns an ``OrderPlan`` into an order at Stage 7.

**Two keys, and the second one is Python.** The fund manager proposes; sizing
turns the proposal into shares; compliance then checks the *post-trade* state
and may veto. §9: *"Any violation blocks the order regardless of what the fund
manager decided. Two-key system: LLM proposes, Python vetoes. This is where
trust comes from."*

So ``OrderPlan.blocked`` is authoritative and no model writes it.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

#: ``block`` stops the order. ``warn`` is recorded, surfaced in the confirmation
#: prompt, and does not stop it.
#:
#: There is exactly one reason a severity exists at all, and it is narrow: a
#: check whose *input* is an estimate rather than a fact. The earnings blackout
#: is the only one today -- with no earnings calendar wired up, the next report
#: date is inferred from EDGAR filing cadence and is good to a week or two, so
#: blocking on it would veto roughly a month of every quarter on a guess. A
#: CONFIRMED date blocks; an ESTIMATED one warns loudly and the human gate sees
#: it. Every other rule in compliance.py is ``block``, because every other rule
#: reads a number that is actually known.
Severity = Literal["block", "warn"]


class Violation(BaseModel):
    """One rule, and what it found. Named by rule so Stage 8 can count them."""

    #: Stable identifier, not prose: ``max_position_pct``, ``min_cash_pct``,
    #: ``earnings_blackout``. Counted and grouped at Stage 8, so it must not
    #: change wording when the message does.
    rule: str
    severity: Severity = "block"
    message: str
    #: The limit, and what the post-trade state would have been. Both optional
    #: because a membership check ("not in the universe") has no number.
    limit: float | None = None
    actual: float | None = None

    @property
    def blocks(self) -> bool:
        return self.severity == "block"

    def render(self) -> str:
        numbers = ""
        if self.limit is not None and self.actual is not None:
            numbers = f" (limit {self.limit:g}, would be {self.actual:g})"
        mark = "BLOCK" if self.blocks else "WARN "
        return f"{mark} {self.rule}: {self.message}{numbers}"


class OrderPlan(BaseModel):
    """A sized, checked instruction. The artefact ``execute`` acts on.

    A plan is produced for **every** decision, including HOLD and including a
    blocked one. A run that silently wrote nothing would be indistinguishable
    from a run that never happened, and the reason an order was not placed is
    the most useful thing in the file.
    """

    schema_version: int = 1
    symbol: str
    as_of: date
    action: Literal["BUY", "SELL", "HOLD"]

    #: Signed share count: positive buys, negative sells. Zero for HOLD, for a
    #: blocked order, and for a delta that did not clear ``min_trade_usd``.
    quantity: int = 0
    #: The mark sizing used. Not a limit price -- ``execute`` computes the
    #: marketable limit from ``limit_offset_bps`` against a fresh quote, since
    #: this number is as stale as the bars cache it came from.
    reference_price: float | None = None
    estimated_notional: float = 0.0

    #: Where the target weight came from, and which of §9's four caps bound it.
    target_weight_pct: float = 0.0
    current_weight_pct: float = 0.0
    binding_cap: str = ""
    #: Every cap considered, with its value. Kept because "why is this order so
    #: small" is the question this layer exists to answer.
    caps_pct: dict[str, float] = Field(default_factory=dict)

    violations: list[Violation] = Field(default_factory=list)
    #: Why no order is going out, in one line. Empty when one is.
    skip_reason: str = ""

    @property
    def blocking(self) -> list[Violation]:
        return [v for v in self.violations if v.blocks]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if not v.blocks]

    @property
    def blocked(self) -> bool:
        return bool(self.blocking)

    @property
    def actionable(self) -> bool:
        """True when ``execute`` has something to place."""
        return self.quantity != 0 and not self.blocked and self.action != "HOLD"

    @model_validator(mode="after")
    def _a_blocked_plan_carries_no_shares(self) -> "OrderPlan":
        """The invariant the whole layer exists to guarantee.

        A blocked plan with a non-zero quantity is the single most dangerous
        object this module could produce: ``execute`` reads ``quantity`` and a
        future refactor that forgets to re-check ``blocked`` would place it.
        Enforced in the schema so that no construction path can express it.
        """
        if self.blocked and self.quantity != 0:
            raise ValueError(
                f"{self.symbol}: plan is blocked by "
                f"{[v.rule for v in self.blocking]} but carries "
                f"{self.quantity} shares. A blocked order is zero shares."
            )
        if self.action == "HOLD" and self.quantity != 0:
            raise ValueError(
                f"{self.symbol}: action is HOLD but quantity is {self.quantity}."
            )
        return self

    def render(self) -> str:
        """Human-readable, for the CLI and for Stage 7's confirmation prompt."""
        lines = [f"ORDER PLAN  {self.symbol}  {self.action}"]
        if self.quantity:
            lines.append(
                f"  {self.quantity:+d} shares @ ~{self.reference_price:,.2f} "
                f"= {self.estimated_notional:,.0f} USD"
            )
        lines.append(
            f"  weight {self.current_weight_pct:.2f}% -> {self.target_weight_pct:.2f}%"
            + (f"  (bound by {self.binding_cap})" if self.binding_cap else "")
        )
        if self.caps_pct:
            caps = "  ".join(f"{k} {v:.2f}%" for k, v in self.caps_pct.items())
            lines.append(f"  caps: {caps}")
        for violation in self.violations:
            lines.append(f"  {violation.render()}")
        if self.skip_reason:
            lines.append(f"  NO ORDER: {self.skip_reason}")
        return "\n".join(lines)


__all__ = ["OrderPlan", "Severity", "Violation"]
