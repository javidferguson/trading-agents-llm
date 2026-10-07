"""Positioning metrics from FINRA and Form 4. All arithmetic, no model (§2)."""

from __future__ import annotations

import logging
import statistics
from datetime import date
from typing import Any

from ..models.market import PositioningMetrics
from .indicators import _Block

logger = logging.getLogger(__name__)


def positioning_metrics(
    short_interest: list[dict[str, Any]] | None,
    short_volume: list[dict[str, Any]] | None,
    insiders: list[dict[str, Any]] | None,
    as_of: date,
    *,
    unavailable_reason: str | None = None,
) -> PositioningMetrics:
    """Combine the three positioning sources, explaining whatever is missing."""
    if unavailable_reason:
        return PositioningMetrics.unavailable(unavailable_reason)

    block = _Block(1)

    # --- short interest: semi-monthly, and always stale ---------------------
    if not short_interest:
        for name in ("short_interest_shares", "short_interest_change_pct",
                     "days_to_cover", "short_interest_age_days"):
            block.unavailable(
                name,
                "no short-interest record is public yet for this date "
                "(semi-monthly, disseminated ~8 business days after settlement)",
            )
    else:
        latest = short_interest[0]
        block.compute("short_interest_shares", 1, lambda: latest.get("short_shares"))
        block.compute("days_to_cover", 1, lambda: latest.get("days_to_cover"))
        block.compute("short_interest_change_pct", 1, lambda: latest.get("change_pct"))
        # How stale, in days. A model reasoning about positioning should know
        # it is looking at a two-to-three-week-old number.
        block.compute(
            "short_interest_age_days", 1,
            lambda: float((as_of - latest["settlement_date"]).days),
        )

    # --- daily short-sale volume -------------------------------------------
    if not short_volume:
        for name in ("short_volume_ratio", "short_volume_ratio_20d"):
            block.unavailable(name, "no published short-volume sessions on or before as_of")
    else:
        ratios = [r["short_ratio"] for r in short_volume if r.get("short_ratio") is not None]
        if ratios:
            block.compute("short_volume_ratio", 1, lambda: ratios[0])
            block.compute(
                "short_volume_ratio_20d", 1,
                lambda: statistics.mean(ratios),
                note=f"averaged over the {len(ratios)} published session(s) found",
            )
        else:
            for name in ("short_volume_ratio", "short_volume_ratio_20d"):
                block.unavailable(name, "sessions found but none reported total volume")

    # --- insiders: discretionary only --------------------------------------
    if insiders is None:
        for name in ("insider_buy_count", "insider_sell_count", "insider_net_usd",
                     "insider_nondiscretionary_count"):
            block.unavailable(name, "Form 4 filings not fetched")
    else:
        discretionary = [t for t in insiders if t.get("discretionary")]
        others = [t for t in insiders if not t.get("discretionary")]

        buys = [t for t in discretionary if t.get("acquired")]
        sells = [t for t in discretionary if not t.get("acquired")]

        block.compute("insider_buy_count", 1, lambda: float(len(buys)))
        block.compute("insider_sell_count", 1, lambda: float(len(sells)))
        block.compute("insider_nondiscretionary_count", 1, lambda: float(len(others)))

        def _net_usd() -> float | None:
            # Only transactions that reported a price. An exercise or a gift
            # has none, and treating that as zero would drag the total toward
            # nothing while looking like a real figure.
            priced = [t for t in discretionary if t.get("usd") is not None]
            if not priced:
                return None
            return sum(
                (t["usd"] if t.get("acquired") else -t["usd"]) for t in priced
            )
        block.compute(
            "insider_net_usd", 1, _net_usd,
            note="no discretionary Form 4 transaction in the window reported a price",
        )

    return block.build(PositioningMetrics)
