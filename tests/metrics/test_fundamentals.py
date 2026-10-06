"""XBRL extraction and the §7.1 value/quality/growth metrics.

Built on hand-written fact documents rather than a recorded EDGAR payload:
Apple's is 3.8 MB, and a fixture that large hides which concept a test actually
depends on. The shapes here are the real ones -- instants vs durations,
restatements, concept aliases -- because those are what the code gets wrong.
"""

from __future__ import annotations

from datetime import date

import pytest

from research_desk.metrics import fundamentals as fun
from research_desk.models.market import CompanyFacts


def facts_from(concepts: dict[str, list[dict]], as_of: date = date(2026, 10, 6)) -> CompanyFacts:
    return CompanyFacts(
        symbol="TEST", cik="0000000001", entity_name="Test Co",
        facts={"us-gaap": {
            name: {"units": {"USD": rows}} for name, rows in concepts.items()
        }},
        as_of=as_of,
    )


def quarter(start: str, end: str, val: float, filed: str, form: str = "10-Q") -> dict:
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form}


def year(start: str, end: str, val: float, filed: str) -> dict:
    return {"start": start, "end": end, "val": val, "filed": filed, "form": "10-K"}


def point(end: str, val: float, filed: str, form: str = "10-Q") -> dict:
    return {"end": end, "val": val, "filed": filed, "form": form}


# --------------------------------------------------------------------------- #
# TTM
# --------------------------------------------------------------------------- #


def test_ttm_sums_the_last_four_quarters() -> None:
    f = facts_from({"Revenues": [
        quarter("2025-10-01", "2025-12-31", 100, "2026-01-30"),
        quarter("2026-01-01", "2026-03-31", 110, "2026-04-30"),
        quarter("2026-04-01", "2026-06-30", 120, "2026-07-30"),
        quarter("2026-07-01", "2026-09-30", 130, "2026-10-01"),
    ]})
    assert fun.ttm(f, fun.REVENUE) == pytest.approx(460)


def test_ttm_uses_only_the_most_recent_four() -> None:
    rows = [
        quarter(f"202{y}-01-01", f"202{y}-03-31", 10 * y, f"202{y}-04-30")
        for y in range(1, 7)
    ]
    f = facts_from({"Revenues": rows})
    # Four most recent by period end: years 3,4,5,6 -> 30+40+50+60
    assert fun.ttm(f, fun.REVENUE) == pytest.approx(180)


def test_ttm_falls_back_to_the_latest_annual() -> None:
    """A 10-K is stale, but stale beats absent -- and an earnings yield built
    on nothing is worse than one built on last year."""
    f = facts_from({"Revenues": [
        year("2024-01-01", "2024-12-31", 900, "2025-02-01"),
        year("2025-01-01", "2025-12-31", 1000, "2026-02-01"),
    ]})
    assert fun.ttm(f, fun.REVENUE) == pytest.approx(1000)


def test_a_restatement_does_not_double_count_a_quarter() -> None:
    """Amended filings republish the same period with a new value. Summing
    both would inflate a TTM by a whole quarter."""
    f = facts_from({"Revenues": [
        quarter("2026-01-01", "2026-03-31", 100, "2026-04-30"),
        quarter("2026-01-01", "2026-03-31", 105, "2026-06-15", form="10-Q/A"),
        quarter("2026-04-01", "2026-06-30", 110, "2026-07-30"),
        quarter("2026-07-01", "2026-09-30", 120, "2026-10-01"),
        quarter("2025-10-01", "2025-12-31", 90, "2026-01-30"),
    ]})
    # The restated 105 wins over the original 100, and only once.
    assert fun.ttm(f, fun.REVENUE) == pytest.approx(105 + 110 + 120 + 90)


def test_concept_aliases_are_tried_in_order() -> None:
    """Revenue is spelled two ways across filers, sometimes both in one doc."""
    f = facts_from({"Revenues": [year("2025-01-01", "2025-12-31", 500, "2026-02-01")]})
    assert fun.ttm(f, fun.REVENUE) == pytest.approx(500)

    preferred = facts_from({
        "Revenues": [year("2025-01-01", "2025-12-31", 500, "2026-02-01")],
        "RevenueFromContractWithCustomerExcludingAssessedTax":
            [year("2025-01-01", "2025-12-31", 520, "2026-02-01")],
    })
    # The alias list puts the contract-revenue concept first.
    assert fun.ttm(preferred, fun.REVENUE) == pytest.approx(520)


# --------------------------------------------------------------------------- #
# Instants vs durations -- the bug that reached real data
# --------------------------------------------------------------------------- #


def test_instants_are_not_found_by_the_duration_helper() -> None:
    """The regression. Total assets and share count are balance-sheet instants.

    Reaching for them with `annual()` finds nothing, which is how the
    Piotroski score first came back "missing assets" for Apple -- a company
    that very obviously reports assets.
    """
    f = facts_from({"Assets": [
        point("2025-09-27", 365_000, "2025-11-01", form="10-K"),
        point("2026-09-26", 380_000, "2026-11-01", form="10-K"),
    ]})
    assert fun.annual(f, fun.ASSETS) is None, "instants have no duration"
    assert fun.instant_years_back(f, fun.ASSETS, years=0) == pytest.approx(380_000)
    assert fun.instant_years_back(f, fun.ASSETS, years=1) == pytest.approx(365_000)


def test_a_drifting_fiscal_year_end_still_matches() -> None:
    """52/53-week calendars move the year end by up to a week, and ordinary
    fiscal ends drift by a few days."""
    f = facts_from({"Assets": [
        point("2025-09-27", 100, "2025-11-01"),
        point("2026-10-03", 110, "2026-11-01"),   # 371 days later
    ]})
    assert fun.instant_years_back(f, fun.ASSETS, years=1) == pytest.approx(100)


def test_a_gap_too_large_to_be_a_year_is_rejected() -> None:
    """Better to report the metric unavailable than to call a two-year change
    a one-year change."""
    f = facts_from({"Assets": [
        point("2022-12-31", 100, "2023-02-01"),
        point("2026-09-30", 110, "2026-11-01"),
    ]})
    assert fun.instant_years_back(f, fun.ASSETS, years=1) is None


# --------------------------------------------------------------------------- #
# Point-in-time
# --------------------------------------------------------------------------- #


def test_facts_filed_after_as_of_are_invisible() -> None:
    """Risk number one (§15.1). A fact filed in 2026 must not be visible to a
    replay of 2025 even though it DESCRIBES 2025."""
    from research_desk.providers.edgar import _visible

    raw = {"us-gaap": {"Revenues": {"units": {"USD": [
        year("2024-01-01", "2024-12-31", 900, "2025-02-01"),
        year("2025-01-01", "2025-12-31", 1000, "2026-02-01"),
    ]}}}}

    early = _visible(raw, date(2025, 6, 1))
    rows = early["us-gaap"]["Revenues"]["units"]["USD"]
    assert len(rows) == 1 and rows[0]["val"] == 900

    late = _visible(raw, date(2026, 6, 1))
    assert len(late["us-gaap"]["Revenues"]["units"]["USD"]) == 2


def test_a_concept_with_nothing_visible_yet_disappears_entirely() -> None:
    from research_desk.providers.edgar import _visible

    raw = {"us-gaap": {"Revenues": {"units": {"USD": [
        year("2025-01-01", "2025-12-31", 1000, "2026-02-01"),
    ]}}}}
    assert _visible(raw, date(2025, 1, 1)) == {}


# --------------------------------------------------------------------------- #
# Metric blocks
# --------------------------------------------------------------------------- #


def complete_facts() -> CompanyFacts:
    """A filer with everything, two years deep."""
    return facts_from({
        "Revenues": [
            year("2024-01-01", "2024-12-31", 1000, "2025-02-01"),
            year("2025-01-01", "2025-12-31", 1200, "2026-02-01"),
        ],
        "GrossProfit": [
            year("2024-01-01", "2024-12-31", 400, "2025-02-01"),
            year("2025-01-01", "2025-12-31", 540, "2026-02-01"),
        ],
        "OperatingIncomeLoss": [
            year("2024-01-01", "2024-12-31", 200, "2025-02-01"),
            year("2025-01-01", "2025-12-31", 300, "2026-02-01"),
        ],
        "NetIncomeLoss": [
            year("2024-01-01", "2024-12-31", 150, "2025-02-01"),
            year("2025-01-01", "2025-12-31", 240, "2026-02-01"),
        ],
        "NetCashProvidedByUsedInOperatingActivities": [
            year("2024-01-01", "2024-12-31", 180, "2025-02-01"),
            year("2025-01-01", "2025-12-31", 300, "2026-02-01"),
        ],
        "PaymentsToAcquirePropertyPlantAndEquipment": [
            year("2025-01-01", "2025-12-31", 60, "2026-02-01"),
        ],
        "Assets": [
            point("2024-12-31", 2000, "2025-02-01", form="10-K"),
            point("2025-12-31", 2200, "2026-02-01", form="10-K"),
        ],
        "AssetsCurrent": [
            point("2024-12-31", 800, "2025-02-01"),
            point("2025-12-31", 1000, "2026-02-01"),
        ],
        "LiabilitiesCurrent": [
            point("2024-12-31", 500, "2025-02-01"),
            point("2025-12-31", 500, "2026-02-01"),
        ],
        "StockholdersEquity": [
            point("2025-12-31", 1200, "2026-02-01"),
        ],
        "LongTermDebtNoncurrent": [
            point("2024-12-31", 400, "2025-02-01"),
            point("2025-12-31", 300, "2026-02-01"),
        ],
        "CashAndCashEquivalentsAtCarryingValue": [
            point("2025-12-31", 200, "2026-02-01"),
        ],
        "CommonStockSharesOutstanding": [
            point("2024-12-31", 110, "2025-02-01"),
            point("2025-12-31", 100, "2026-02-01"),
        ],
        "EarningsPerShareDiluted": [
            year("2024-01-01", "2024-12-31", 1.36, "2025-02-01"),
            year("2025-01-01", "2025-12-31", 2.40, "2026-02-01"),
        ],
        "InterestExpense": [
            year("2025-01-01", "2025-12-31", 20, "2026-02-01"),
        ],
    })


def test_value_metrics_are_yields_not_multiples() -> None:
    """Yields stay finite through negative earnings; multiples do not."""
    metrics = fun.value_metrics(complete_facts(), price=50.0)

    # 100 shares at 50 = 5000 market cap
    assert metrics.market_cap == pytest.approx(5000)
    assert metrics.earnings_yield_pct == pytest.approx(240 / 5000 * 100)
    # FCF = CFO 300 - capex 60
    assert metrics.fcf_yield_pct == pytest.approx(240 / 5000 * 100)
    assert metrics.book_to_price == pytest.approx(1200 / 5000)
    assert metrics.sales_to_price == pytest.approx(1200 / 5000)
    assert not metrics.gaps


def test_a_loss_making_filer_still_produces_a_finite_yield() -> None:
    """The whole reason §7.1 chose yields: a P/E of -48 is nonsense a small
    model will reason about anyway."""
    f = facts_from({
        "NetIncomeLoss": [year("2025-01-01", "2025-12-31", -100, "2026-02-01")],
        "CommonStockSharesOutstanding": [point("2025-12-31", 100, "2026-02-01")],
    })
    metrics = fun.value_metrics(f, price=10.0)
    assert metrics.earnings_yield_pct == pytest.approx(-10.0)


def test_quality_metrics() -> None:
    metrics = fun.quality_metrics(complete_facts())

    assert metrics.gross_profitability == pytest.approx(540 / 2200)
    # Negative accruals: cash flow exceeds reported earnings. Good.
    assert metrics.accruals == pytest.approx((240 - 300) / 2200)
    assert metrics.accruals < 0
    assert metrics.roe_pct == pytest.approx(240 / 1200 * 100)
    assert metrics.current_ratio == pytest.approx(1000 / 500)
    assert metrics.interest_coverage == pytest.approx(300 / 20)
    assert not metrics.gaps


def test_piotroski_scores_a_healthy_filer_highly() -> None:
    score, missing = fun.piotroski_f_score(complete_facts())
    assert not missing
    # Profitable, cash-backed, improving, deleveraging, buying back stock,
    # expanding margins and turning assets faster: all nine.
    assert score == 9


def test_piotroski_is_none_rather_than_partial() -> None:
    """A partial score silently means something different -- 4/9 computed from
    five available components is not a 4."""
    f = facts_from({
        "NetIncomeLoss": [year("2025-01-01", "2025-12-31", 100, "2026-02-01")],
    })
    score, missing = fun.piotroski_f_score(f)
    assert score is None
    assert missing


def test_growth_metrics() -> None:
    metrics = fun.growth_metrics(complete_facts())

    assert metrics.revenue_yoy_pct == pytest.approx(20.0)
    assert metrics.eps_yoy_pct == pytest.approx((2.40 - 1.36) / 1.36 * 100)
    assert metrics.gross_margin_pct == pytest.approx(540 / 1200 * 100)
    assert metrics.operating_margin_pct == pytest.approx(300 / 1200 * 100)
    # Margin expanded: 45% from 40%
    assert metrics.gross_margin_change_pct == pytest.approx(5.0)
    # Buyback: 110 -> 100 shares
    assert metrics.share_count_change_1y_pct == pytest.approx(-100 / 11)


def test_a_swing_out_of_a_loss_reads_positive() -> None:
    """abs() in the denominator. Without it, -100 -> +50 reads as -150%
    growth, which is the wrong sign on a genuine recovery."""
    f = facts_from({"Revenues": [
        year("2024-01-01", "2024-12-31", -100, "2025-02-01"),
        year("2025-01-01", "2025-12-31", 50, "2026-02-01"),
    ]})
    assert fun.growth_metrics(f).revenue_yoy_pct == pytest.approx(150.0)


# --------------------------------------------------------------------------- #
# The ETF case -- the Stage 2 exit gate
# --------------------------------------------------------------------------- #


def test_no_facts_produces_explained_gaps_not_an_exception() -> None:
    """SPY has a CIK but 404s on company facts, with an XML body. A naive path
    breaks twice; this one reports 22 explained gaps and exits 0."""
    for builder in (fun.value_metrics, fun.quality_metrics, fun.growth_metrics):
        metrics = builder(None) if builder is not fun.value_metrics else builder(None, 100.0)
        assert metrics.gaps
        for name, reason in metrics.gaps.items():
            assert "XBRL" in reason, f"{name} should say why"
        assert not metrics.available()


def test_a_bank_without_a_classified_balance_sheet_says_so() -> None:
    """Banks and insurers do not present current assets/liabilities. That is a
    reporting convention, not missing data, and the reason should say so."""
    f = facts_from({
        "Revenues": [year("2025-01-01", "2025-12-31", 1000, "2026-02-01")],
        "Assets": [point("2025-12-31", 5000, "2026-02-01")],
    })
    metrics = fun.quality_metrics(f)
    assert metrics.current_ratio is None
    assert "classified balance sheet" in metrics.gaps["current_ratio"]


def test_price_without_facts_is_still_unavailable_not_zero() -> None:
    assert fun.value_metrics(None, None).gaps
    assert fun.value_metrics(complete_facts(), None).market_cap is None


def test_provenance_names_the_statement_not_the_newest_filing() -> None:
    """Observed against real data: a naive "most recent filed" reported
    `424B2` for JPM and `2.01 SD` for Berkshire, implying the fundamentals
    came out of a prospectus. Filers tag facts in prospectuses, 8-Ks and
    disclosure filings, and those are often the newest thing in the document.
    """
    f = facts_from({
        "Revenues": [
            year("2025-01-01", "2025-12-31", 1000, "2026-02-01"),
            # A prospectus supplement, tagged later than the 10-K.
            {"start": "2025-01-01", "end": "2025-12-31", "val": 1000,
             "filed": "2026-09-30", "form": "424B2"},
        ],
    })
    filed, form = f.latest_filing
    assert form == "10-K", f"reported {form!r}, which is not a financial statement"
    assert filed == date(2026, 2, 1)


def test_an_amended_statement_still_counts_as_provenance() -> None:
    f = facts_from({"Revenues": [
        year("2025-01-01", "2025-12-31", 1000, "2026-02-01"),
        {"start": "2025-01-01", "end": "2025-12-31", "val": 1010,
         "filed": "2026-03-15", "form": "10-K/A"},
    ]})
    filed, form = f.latest_filing
    assert form == "10-K/A" and filed == date(2026, 3, 15)


# --------------------------------------------------------------------------- #
# Foreign private issuers -- found by a live Stage 4 run
# --------------------------------------------------------------------------- #


def facts_multi(concepts: dict[str, dict[str, list[dict]]]) -> CompanyFacts:
    """Facts with explicit taxonomy and unit nesting."""
    built: dict[str, dict] = {}
    for key, units in concepts.items():
        taxonomy, name = key.split("/", 1)
        built.setdefault(taxonomy, {})[name] = {"units": units}
    return CompanyFacts(symbol="TSM", cik="0001046179", entity_name="TSMC",
                        facts=built, as_of=date(2026, 10, 6))


def test_ifrs_concepts_are_found_for_a_20f_filer() -> None:
    """A 20-F filer uses IFRS exclusively.

    Measured: TSM's company facts carry 334 `ifrs-full` concepts and ZERO
    `us-gaap` ones, so a us-gaap-only lookup found nothing and the fundamentals
    analyst reported having no data at all -- for a company that files
    perfectly good financials.
    """
    f = facts_multi({"ifrs-full/Revenue": {"USD": [
        year("2025-01-01", "2025-12-31", 88_000_000_000, "2026-04-16"),
    ]}})
    assert fun.ttm(f, fun.REVENUE) == pytest.approx(88e9)


def test_a_concept_never_mixes_two_currencies() -> None:
    """The subtler half, and the more dangerous.

    TSM reports Revenue, Assets, Equity and ProfitLoss under BOTH 'TWD' and
    'USD'. Flattening every unit into one series interleaved two currencies
    and sorted by date, so whichever happened to be last won -- and
    book_to_price would divide a Taiwan-dollar equity by a US-dollar market
    cap, producing a number roughly 32x wrong that looks entirely plausible.
    """
    f = facts_multi({"ifrs-full/Revenue": {
        "TWD": [year("2025-01-01", "2025-12-31", 2_894_000_000_000, "2026-04-16")],
        "USD": [year("2025-01-01", "2025-12-31", 88_000_000_000, "2026-04-16")],
    }})

    assert f.unit_for("Revenue") == "USD", "USD is preferred; the price side is USD"
    revenue = fun.ttm(f, fun.REVENUE)
    assert revenue == pytest.approx(88e9)
    assert revenue != pytest.approx(2.894e12), "the TWD series must not leak in"


def test_a_single_currency_filer_gets_that_currency() -> None:
    """Not every foreign filer helpfully reports USD too."""
    f = facts_multi({"ifrs-full/Revenue": {
        "TWD": [year("2025-01-01", "2025-12-31", 2_894_000_000_000, "2026-04-16")],
    }})
    assert f.unit_for("Revenue") == "TWD"
    assert fun.ttm(f, fun.REVENUE) == pytest.approx(2.894e12)


def test_us_gaap_wins_when_a_filer_has_both_taxonomies() -> None:
    """Order matters: a domestic filer's us-gaap series is the canonical one."""
    f = facts_multi({
        "us-gaap/Revenues": {"USD": [year("2025-01-01", "2025-12-31", 100, "2026-02-01")]},
        "ifrs-full/Revenue": {"USD": [year("2025-01-01", "2025-12-31", 999, "2026-02-01")]},
    })
    assert fun.ttm(f, fun.REVENUE) == pytest.approx(100)
