"""The earnings blackout, estimated from filing cadence. **Warns, never blocks.**

``risk.earnings_blackout_days_before/after`` is a compliance rule with no data
source. FOLLOWUPS.md said so before this stage began: *"Finnhub -- company news,
earnings calendar... No events block. The earnings-blackout compliance check
(§9) has no input, so Stage 6 will need this."* There is no ``FINNHUB_API_KEY``.

Rather than ship a veto that cannot fire, or a hand-maintained calendar that
goes stale the moment nobody updates it, this derives the next report date from
**the cadence of past periodic filings**, which EDGAR already has cached for the
universe. A company that filed 10-Qs 91, 92 and 90 days apart will file the next
one in about 91 days.

**That estimate is good to a week or two, and the design says so out loud.**
Blocking on it would veto roughly a month of every quarter on a guess -- and
worse, it would look like a real earnings veto in the log. So:

* a **confirmed** date (an earnings calendar, when one is wired up) → ``block``
* an **estimated** date inside the window → ``warn``, loudly, surfaced in the
  confirmation prompt a human reads
* **no usable history** (an ETF, a foreign private issuer) → ``warn`` saying
  the check could not be evaluated

This is the only check in ``compliance.py`` that does not block, and the reason
is narrow and worth restating: every other rule reads a number that is actually
known. Flipping this to ``block`` is one line and no schema change, the day a
real calendar exists.

**Why not just assume 91 days from the last filing?** Because fiscal calendars
are not uniform. AVGO's year ends in early November and it files its 10-K in
mid-December; a flat 91-day assumption puts its next report five weeks early.
The median of the observed intervals handles that, and handles the two-week
drift of a company that files on the second Tuesday rather than a fixed date.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

logger = logging.getLogger(__name__)

#: Which filings count as an earnings report for cadence purposes. 6-K is
#: excluded upstream in the provider -- see ``periodic_filings``.
#:
#: 20-F and 40-F are annual-only, so a symbol whose history is all 20-F gets a
#: ~365-day cadence. That is correct and useless: it says nothing about the
#: quarterly results an FPI announces on 6-K. ``_basis_for`` refuses to call
#: such an estimate usable rather than reporting a confident annual date.
QUARTERLY_FORMS = frozenset({"10-Q", "10-K"})

#: Minimum intervals needed before a median means anything. Two filings give
#: one interval, which is an observation rather than a cadence -- a single late
#: filing would set the whole estimate.
MIN_INTERVALS_FOR_MEDIAN = 2

#: Fallback when there are too few filings to measure. A US quarter, and it is
#: used only to produce *something* to warn about; ``basis`` stays
#: ``"estimated"`` either way and nothing blocks on it.
NOMINAL_QUARTER_DAYS = 91

#: Above this, the cadence is annual rather than quarterly and the symbol does
#: not report through EDGAR periodic filings at a useful frequency.
ANNUAL_CADENCE_DAYS = 200

Basis = Literal["confirmed", "estimated", "unknown"]


@dataclass(frozen=True)
class EarningsEstimate:
    """When the next report probably lands, and how much that "probably" is worth."""

    symbol: str
    next_report: date | None
    basis: Basis
    source: str
    note: str
    last_report: date | None = None
    interval_days: int | None = None
    observations: int = 0

    @property
    def is_confirmed(self) -> bool:
        return self.basis == "confirmed"

    @property
    def is_usable(self) -> bool:
        return self.next_report is not None and self.basis != "unknown"

    def window(self, *, days_before: int, days_after: int) -> tuple[date, date] | None:
        """The blackout window around the report, or ``None`` if unknown."""
        if self.next_report is None:
            return None
        return (
            self.next_report - timedelta(days=days_before),
            self.next_report + timedelta(days=days_after),
        )

    def describe(self) -> str:
        if self.next_report is None:
            return f"{self.symbol}: next report unknown ({self.note})"
        cadence = f", ~{self.interval_days}d cadence" if self.interval_days else ""
        return (
            f"{self.symbol}: next report {self.next_report.isoformat()} "
            f"({self.basis}, from {self.source}{cadence})"
        )


def _intervals(filed: list[date]) -> list[int]:
    """Days between consecutive filings, newest-first input."""
    return [(filed[i] - filed[i + 1]).days for i in range(len(filed) - 1)]


def estimate_from_filings(
    symbol: str, filings: list[dict[str, Any]], *, as_of: date
) -> EarningsEstimate:
    """Pure half of the estimate, so it is testable without a provider.

    ``filings`` is ``registry.periodic_filings()`` output: newest first, each
    with ``form`` and ``filed``.
    """
    quarterly = [
        row["filed"] for row in filings
        if row.get("form") in QUARTERLY_FORMS and isinstance(row.get("filed"), date)
    ]

    if not quarterly:
        annual = [r for r in filings if isinstance(r.get("filed"), date)]
        if annual:
            # The foreign-private-issuer case. There IS filing history; it is
            # just annual, so it cannot locate a quarterly report.
            return EarningsEstimate(
                symbol=symbol, next_report=None, basis="unknown",
                source="edgar", observations=len(annual),
                last_report=annual[0]["filed"],
                note=(
                    f"only annual filings ({annual[0]['form']}) -- a foreign "
                    "private issuer announces quarterly results on 6-K, which "
                    "is not a periodic report and cannot be dated from cadence"
                ),
            )
        return EarningsEstimate(
            symbol=symbol, next_report=None, basis="unknown", source="edgar",
            note="no periodic filings on EDGAR (an ETF has none)",
        )

    last = quarterly[0]
    gaps = _intervals(quarterly)

    if len(gaps) >= MIN_INTERVALS_FOR_MEDIAN:
        # Median, not mean: one late filing or one restatement would drag a
        # mean by weeks, and the whole point is to be robust to the single
        # irregular quarter rather than to model it.
        interval = int(round(statistics.median(gaps)))
        source = f"edgar filing cadence, median of {len(gaps)} intervals"
    else:
        interval = NOMINAL_QUARTER_DAYS
        source = (
            f"nominal {NOMINAL_QUARTER_DAYS}-day quarter -- only "
            f"{len(quarterly)} periodic filing(s) visible, too few to measure "
            "a cadence"
        )

    if interval > ANNUAL_CADENCE_DAYS:
        return EarningsEstimate(
            symbol=symbol, next_report=None, basis="unknown", source="edgar",
            last_report=last, interval_days=interval, observations=len(quarterly),
            note=(
                f"filing cadence is ~{interval} days, which is annual rather "
                "than quarterly -- it cannot locate the next earnings report"
            ),
        )

    projected = last + timedelta(days=interval)

    # A projection already in the past means the company is overdue by the
    # cadence -- i.e. the report is imminent or was filed and not yet visible
    # in this cache. Rolling forward is wrong (it would skip the very print we
    # care about), so the window stays where the cadence put it and the note
    # says the estimate is behind.
    overdue = (as_of - projected).days
    note = (
        f"projected from {last.isoformat()} + {interval}d. "
        + (
            f"ALREADY {overdue} DAY(S) PAST -- a report is overdue by this "
            "cadence, or was filed and is not yet in the cache"
            if overdue > 0 else
            "An estimate from filing cadence is typically good to within "
            "1-2 weeks, which is why it warns rather than blocks."
        )
    )

    return EarningsEstimate(
        symbol=symbol, next_report=projected, basis="estimated", source=source,
        last_report=last, interval_days=interval, observations=len(quarterly),
        note=note,
    )


async def estimate_next_report(
    registry: Any, symbol: str, as_of: date
) -> EarningsEstimate:
    """Fetch the filing history and estimate from it.

    Any provider failure is an ``unknown`` estimate rather than an exception:
    a blackout check that cannot reach EDGAR must not take down a decision
    run, and ``unknown`` already means "warn, do not block".
    """
    if registry is None:
        return EarningsEstimate(
            symbol=symbol, next_report=None, basis="unknown", source="none",
            note="no provider registry available to read filing history",
        )
    try:
        filings = await registry.periodic_filings(symbol, as_of)
    except Exception as exc:  # noqa: BLE001 -- degrade, never raise (§5)
        logger.warning("earnings estimate for %s failed: %s", symbol, exc)
        return EarningsEstimate(
            symbol=symbol, next_report=None, basis="unknown", source="edgar",
            note=f"could not read filing history: {type(exc).__name__}: {exc}",
        )
    return estimate_from_filings(symbol, filings, as_of=as_of)


def blackout_reason(
    estimate: EarningsEstimate,
    as_of: date,
    *,
    days_before: int,
    days_after: int,
) -> str | None:
    """Why ``as_of`` sits in ``symbol``'s blackout, or ``None`` if it does not.

    Returns a reason for an *unknown* estimate too -- "could not be evaluated"
    is a finding, not a pass. The caller decides severity; ``compliance.py``
    makes every one of these a ``warn`` until a confirmed calendar exists.
    """
    if not estimate.is_usable:
        return (
            f"earnings blackout could not be evaluated -- {estimate.note}"
        )

    window = estimate.window(days_before=days_before, days_after=days_after)
    if window is None:
        return None
    start, end = window
    if start <= as_of <= end:
        return (
            f"{as_of.isoformat()} is inside the earnings blackout "
            f"{start.isoformat()}..{end.isoformat()} around an "
            f"{estimate.basis} report date of {estimate.next_report.isoformat()} "
            f"({estimate.source})"
        )
    return None


__all__ = [
    "ANNUAL_CADENCE_DAYS",
    "Basis",
    "EarningsEstimate",
    "MIN_INTERVALS_FOR_MEDIAN",
    "NOMINAL_QUARTER_DAYS",
    "QUARTERLY_FORMS",
    "blackout_reason",
    "estimate_from_filings",
    "estimate_next_report",
]
