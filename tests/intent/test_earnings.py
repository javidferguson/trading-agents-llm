"""The earnings estimate. **It warns, and the tests say why that is correct.**

The design decision under test is not the arithmetic -- it is the severity. With
no earnings calendar wired up (FOLLOWUPS.md: *"The earnings-blackout compliance
check (§9) has no input, so Stage 6 will need this"*), the next report date is
projected from EDGAR filing cadence and is good to a week or two. Blocking on it
would veto roughly a month of every quarter on a guess, and would look like a
real earnings veto in the log.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from research_desk.intent.earnings import (
    NOMINAL_QUARTER_DAYS,
    blackout_reason,
    estimate_from_filings,
    estimate_next_report,
)

AS_OF = date(2026, 10, 6)


def filings(*rows: tuple[str, str]) -> list[dict]:
    """``(form, filingDate)`` pairs, newest first, as the provider returns them."""
    return [{"form": f, "filed": date.fromisoformat(d)} for f, d in rows]


def quarterly(last: date, interval: int, count: int = 6) -> list[dict]:
    return [
        {"form": "10-Q", "filed": last - timedelta(days=interval * i)}
        for i in range(count)
    ]


# --------------------------------------------------------------------------- #
# The cadence estimate
# --------------------------------------------------------------------------- #


def test_the_next_report_is_projected_from_the_median_interval() -> None:
    rows = quarterly(date(2026, 8, 26), 91)
    estimate = estimate_from_filings("NVDA", rows, as_of=AS_OF)
    assert estimate.basis == "estimated"
    assert estimate.interval_days == 91
    assert estimate.next_report == date(2026, 8, 26) + timedelta(days=91)


def test_the_median_ignores_one_irregular_quarter() -> None:
    """Why median and not mean: one late filing would drag a mean by weeks, and
    being robust to the single odd quarter is the entire job."""
    rows = filings(
        ("10-Q", "2026-08-26"), ("10-Q", "2026-05-27"), ("10-Q", "2026-02-25"),
        ("10-K", "2025-11-26"), ("10-Q", "2025-02-20"),  # a 9-month hole
    )
    estimate = estimate_from_filings("X", rows, as_of=AS_OF)
    assert estimate.interval_days == pytest.approx(91, abs=3)


def test_a_fiscal_calendar_that_is_not_a_flat_quarter_is_handled() -> None:
    """A flat 91-day assumption puts AVGO's next report five weeks early,
    because its year ends in November and the 10-K follows in mid-December."""
    rows = filings(
        ("10-Q", "2026-09-10"), ("10-Q", "2026-06-09"), ("10-Q", "2026-03-11"),
        ("10-K", "2025-12-18"), ("10-Q", "2025-09-10"), ("10-Q", "2025-06-11"),
    )
    estimate = estimate_from_filings("AVGO", rows, as_of=AS_OF)
    assert estimate.next_report is not None
    # Mid-December, not early November.
    assert estimate.next_report.month == 12


def test_too_few_filings_falls_back_to_a_nominal_quarter_and_says_so() -> None:
    """XOM has one visible 10-Q: EDGAR's recent index is capped, so a prolific
    filer's periodic reports get pushed out of it."""
    estimate = estimate_from_filings(
        "XOM", filings(("10-Q", "2026-08-03")), as_of=AS_OF
    )
    assert estimate.basis == "estimated"
    assert estimate.interval_days == NOMINAL_QUARTER_DAYS
    assert "too few to measure" in estimate.source


def test_an_overdue_projection_is_not_rolled_forward() -> None:
    """Rolling it forward would skip the very print the check is about."""
    rows = quarterly(date(2026, 5, 1), 91)
    estimate = estimate_from_filings("X", rows, as_of=AS_OF)
    assert estimate.next_report == date(2026, 5, 1) + timedelta(days=91)
    assert estimate.next_report < AS_OF
    assert "ALREADY" in estimate.note


# --------------------------------------------------------------------------- #
# The cases that must NOT produce a confident date
# --------------------------------------------------------------------------- #


def test_a_foreign_private_issuer_is_unknown_rather_than_annual() -> None:
    """TSM files a 20-F once a year and reports quarterly results on 6-K, which
    it also uses for press releases -- 664 of them against 13 20-Fs. An annual
    cadence cannot locate a quarterly report, and saying so beats a date."""
    rows = filings(("20-F", "2026-04-16"), ("20-F", "2025-04-17"),
                   ("20-F", "2024-04-18"))
    estimate = estimate_from_filings("TSM", rows, as_of=AS_OF)
    assert estimate.basis == "unknown"
    assert estimate.next_report is None
    assert "foreign private issuer" in estimate.note
    assert not estimate.is_usable


def test_an_etf_with_no_periodic_filings_is_unknown() -> None:
    estimate = estimate_from_filings("SPY", [], as_of=AS_OF)
    assert estimate.basis == "unknown"
    assert "no periodic filings" in estimate.note


def test_an_annual_only_quarterly_cadence_is_refused() -> None:
    rows = [{"form": "10-K", "filed": date(2026, 2, 1) - timedelta(days=365 * i)}
            for i in range(4)]
    estimate = estimate_from_filings("X", rows, as_of=AS_OF)
    assert estimate.basis == "unknown"
    assert "annual rather than quarterly" in estimate.note


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #


def test_the_blackout_window_spans_before_and_after_the_report() -> None:
    estimate = estimate_from_filings("X", quarterly(AS_OF - timedelta(days=91), 91),
                                     as_of=AS_OF)
    start, end = estimate.window(days_before=2, days_after=1)
    assert start == estimate.next_report - timedelta(days=2)
    assert end == estimate.next_report + timedelta(days=1)


def test_a_date_inside_the_window_produces_a_reason() -> None:
    rows = quarterly(AS_OF - timedelta(days=90), 91)  # next report tomorrow
    estimate = estimate_from_filings("X", rows, as_of=AS_OF)
    reason = blackout_reason(estimate, AS_OF, days_before=2, days_after=1)
    assert reason is not None
    assert "inside the earnings blackout" in reason
    # The basis is named in the reason, so a human reading the confirmation
    # prompt knows how much the date is worth.
    assert "estimated" in reason


def test_a_date_outside_the_window_produces_nothing() -> None:
    rows = quarterly(AS_OF - timedelta(days=10), 91)
    estimate = estimate_from_filings("X", rows, as_of=AS_OF)
    assert blackout_reason(estimate, AS_OF, days_before=2, days_after=1) is None


def test_an_unknown_estimate_is_a_finding_not_a_pass() -> None:
    estimate = estimate_from_filings("SPY", [], as_of=AS_OF)
    reason = blackout_reason(estimate, AS_OF, days_before=2, days_after=1)
    assert reason is not None
    assert "could not be evaluated" in reason


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #


async def test_a_provider_failure_degrades_to_unknown_rather_than_raising() -> None:
    """A blackout check that cannot reach EDGAR must not take down a run (§5)."""
    class Broken:
        async def periodic_filings(self, symbol, as_of):
            raise RuntimeError("EDGAR returned 403")

    estimate = await estimate_next_report(Broken(), "AAA", AS_OF)
    assert estimate.basis == "unknown"
    assert "403" in estimate.note


async def test_no_registry_degrades_to_unknown() -> None:
    estimate = await estimate_next_report(None, "AAA", AS_OF)
    assert estimate.basis == "unknown"


async def test_filings_are_point_in_time() -> None:
    """A filing that was not public on ``as_of`` cannot inform a decision made
    on ``as_of`` (§7.4). Filtered in the provider; asserted here on its contract.
    """
    class Fake:
        def __init__(self) -> None:
            self.asked: date | None = None

        async def periodic_filings(self, symbol, as_of):
            self.asked = as_of
            return quarterly(date(2026, 8, 26), 91)

    registry = Fake()
    await estimate_next_report(registry, "AAA", AS_OF)
    assert registry.asked == AS_OF
