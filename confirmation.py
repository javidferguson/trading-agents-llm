"""The human confirmation gate.

Every order on this path requires explicit approval. The original ORB bot had no
gate at all, and the options scanner's gate was inverted -- setting
``require_confirmation: false`` still prompted and then returned True regardless
of the answer.

Two deliberate choices here:

* Approval requires typing the **ticker symbol**, not "y". A yes/no prompt is
  answered by muscle memory; a symbol has to be read first.
* The prompt shows the whole bracket and the whatIf margin/commission estimate,
  so the numbers being approved are the numbers that will be sent.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol

from ..models import TradeDecision

logger = logging.getLogger(__name__)


class PreflightReport(Protocol):
    """What ib.whatIfOrder() told us about the proposed order."""

    init_margin_change: Optional[str]
    maint_margin_change: Optional[str]
    commission: Optional[float]
    warning: Optional[str]


class ConfirmationGate(Protocol):
    def confirm(self, decision: TradeDecision, preflight: Optional[PreflightReport] = None) -> bool:
        ...


def render_decision(decision: TradeDecision, preflight: Optional[PreflightReport] = None) -> str:
    """Format the trade for human review."""
    side = "CALL" if decision.right.value == "C" else "PUT"
    lines = [
        "=" * 64,
        "TRADE PROPOSAL -- REVIEW BEFORE APPROVING",
        "=" * 64,
        f"  Symbol       : {decision.symbol}",
        f"  Instrument   : {decision.strike:.2f} {side}  exp {decision.expiration}",
        f"  Quantity     : {decision.quantity} contract(s)",
        "",
        f"  Signal       : {decision.signal.signal_type.value} "
        f"({decision.signal.strategy}) at {decision.signal.timestamp}",
        f"  Spot price   : {decision.spot_price:.2f}",
        f"  Peak GEX     : {decision.highest_gex_strike:.2f} "
        f"({'above' if decision.highest_gex_strike > decision.spot_price else 'below'} spot)",
        "",
        "  BRACKET",
        f"    Entry (LMT): ${decision.entry_price:.2f}",
        f"    Take profit: ${decision.take_profit_price:.2f}  "
        f"({(decision.take_profit_price / decision.entry_price - 1) * 100:+.1f}%)",
        f"    Stop loss  : ${decision.stop_loss_price:.2f}  "
        f"({(decision.stop_loss_price / decision.entry_price - 1) * 100:+.1f}%)",
        "",
        f"  Total debit  : ${decision.total_debit:,.2f}",
        f"  Max loss     : ${(decision.entry_price - decision.stop_loss_price) * decision.quantity * 100:,.2f}"
        "  (if the stop fills at its trigger)",
    ]

    if preflight is not None:
        lines += [
            "",
            "  BROKER PREFLIGHT (whatIf -- nothing submitted)",
            f"    Init margin : {preflight.init_margin_change or 'n/a'}",
            f"    Maint margin: {preflight.maint_margin_change or 'n/a'}",
            f"    Commission  : {preflight.commission if preflight.commission is not None else 'n/a'}",
        ]
        if preflight.warning:
            lines.append(f"    WARNING     : {preflight.warning}")

    lines.append("=" * 64)
    return "\n".join(lines)


class CLIConfirmationGate:
    """Prompts on stdin. Requires the ticker symbol to be typed to approve."""

    def confirm(self, decision: TradeDecision, preflight: Optional[PreflightReport] = None) -> bool:
        print(render_decision(decision, preflight))
        expected = decision.symbol.upper()
        prompt = f"\nType {expected} to place this order, or anything else to skip: "

        try:
            answer = input(prompt).strip().upper()
        except (EOFError, KeyboardInterrupt):
            # A closed stdin (non-interactive container) must decline, not proceed.
            print()
            logger.warning("No interactive input available; declining the trade.")
            return False

        if answer == expected:
            logger.info("Trade approved by user: %s %s", decision.symbol, decision.strike)
            return True

        logger.info("Trade declined by user (entered %r, expected %r)", answer, expected)
        return False


class RejectAllGate:
    """Declines everything. Used for dry runs and tests."""

    def confirm(self, decision: TradeDecision, preflight: Optional[PreflightReport] = None) -> bool:
        logger.info("RejectAllGate: declining %s %s", decision.symbol, decision.strike)
        return False
