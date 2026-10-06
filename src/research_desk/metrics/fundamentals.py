"""Value, quality and growth from EDGAR facts. All arithmetic, no model (§2).

XBRL is not a table. Three things make extraction harder than it looks, and
all three are handled here rather than pushed downstream:

**Concepts have aliases.** Revenue is ``Revenues`` for some filers and
``RevenueFromContractWithCustomerExcludingAssessedTax`` for others, sometimes
both in one document. Each metric names an ordered list and takes the first
that yields data.

**Flows need trailing twelve months, not the last annual figure.** A 10-K is
up to a year stale by the time you read it, and an earnings yield computed
from stale earnings is wrong in a way that looks fine. TTM sums the last four
quarters, falling back to the most recent annual when quarterly data is thin.

**Stocks and flows are different shapes.** Balance-sheet facts are instants
(``end`` only); income and cash-flow facts are durations (``start``..``end``).
Mixing them silently produces plausible nonsense.

Everything unavailable says why, per the Stage 2 gate.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from ..models.market import CompanyFacts, GrowthMetrics, QualityMetrics, ValueMetrics
from .indicators import _Block

logger = logging.getLogger(__name__)

# --- concept aliases, in preference order ---------------------------------- #

REVENUE = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
]
NET_INCOME = ["NetIncomeLoss", "ProfitLoss"]
GROSS_PROFIT = ["GrossProfit"]
OPERATING_INCOME = ["OperatingIncomeLoss"]
CFO = ["NetCashProvidedByUsedInOperatingActivities",
       "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
CAPEX = ["PaymentsToAcquirePropertyPlantAndEquipment",
         "PaymentsToAcquireProductiveAssets"]
ASSETS = ["Assets"]
ASSETS_CURRENT = ["AssetsCurrent"]
LIABILITIES_CURRENT = ["LiabilitiesCurrent"]
EQUITY = ["StockholdersEquity",
          "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
LONG_TERM_DEBT = ["LongTermDebtNoncurrent", "LongTermDebt"]
SHORT_TERM_DEBT = ["ShortTermBorrowings", "LongTermDebtCurrent"]
CASH = ["CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"]
INTEREST = ["InterestExpense", "InterestExpenseDebt",
            "InterestIncomeExpenseNet"]
DA = ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
      "DepreciationAndAmortization"]
EPS = ["EarningsPerShareDiluted", "EarningsPerShareBasic"]
SHARES = ["CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding",
          "WeightedAverageNumberOfSharesOutstandingBasic"]

#: A duration fact counts as a quarter or a year if its span is near these.
QUARTER_DAYS = (80, 100)
ANNUAL_DAYS = (330, 400)


def _span_days(row: dict[str, Any]) -> int | None:
    start, end = row.get("start"), row.get("end")
    if not start or not end:
        return None
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _dedup_latest_filed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per period, keeping the most recently filed.

    Restatements and amended filings (10-K/A) republish the same period with a
    different value. Taking both would double-count a quarter in a TTM sum.
    """
    best: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        key = (row.get("start"), row.get("end"))
        current = best.get(key)
        if current is None or (row.get("filed") or "") > (current.get("filed") or ""):
            best[key] = row
    return sorted(best.values(), key=lambda r: r.get("end") or "")


def _rows_for(facts: CompanyFacts, concepts: list[str]) -> list[dict[str, Any]]:
    for name in concepts:
        rows = facts.concept(name)
        if rows:
            return _dedup_latest_filed(rows)
    return []


def instant(facts: CompanyFacts, concepts: list[str], *, before: str | None = None) -> float | None:
    """Most recent balance-sheet value (a point in time, not a period)."""
    rows = [r for r in _rows_for(facts, concepts) if _span_days(r) is None or _span_days(r) < 5]
    if before:
        rows = [r for r in rows if (r.get("end") or "") < before]
    return float(rows[-1]["val"]) if rows else None


def instant_series(facts: CompanyFacts, concepts: list[str]) -> list[tuple[date, float]]:
    """Every balance-sheet observation, ``(as_at, value)``, oldest first."""
    rows = [
        r for r in _rows_for(facts, concepts)
        if (_span_days(r) is None or _span_days(r) < 5) and r.get("end")
    ]
    by_date: dict[date, float] = {}
    for row in rows:
        by_date[date.fromisoformat(row["end"])] = float(row["val"])
    return sorted(by_date.items())


def instant_years_back(
    facts: CompanyFacts,
    concepts: list[str],
    years: int = 1,
    tolerance_days: int = 70,
) -> float | None:
    """A balance-sheet value roughly ``years`` before the latest observation.

    Needed because **total assets and share count are instants, not annual
    durations.** Reaching for them with ``annual()`` finds nothing -- which is
    how the Piotroski score and the share-count change first came back
    unavailable against real Apple data, reporting "missing assets" for a
    company that obviously reports assets.

    The tolerance exists because fiscal year-ends drift by a few days and a
    52/53-week retail calendar drifts by a week.
    """
    series = instant_series(facts, concepts)
    if not series:
        return None
    if years == 0:
        return series[-1][1]

    latest_at, _ = series[-1]
    target = latest_at - timedelta(days=365 * years)
    nearest_at, nearest_val = min(series, key=lambda pair: abs((pair[0] - target).days))
    if abs((nearest_at - target).days) > tolerance_days:
        return None
    return nearest_val


def annual(facts: CompanyFacts, concepts: list[str], *, offset: int = 0) -> dict[str, Any] | None:
    """The ``offset``-th most recent full-year duration fact (0 = latest)."""
    years = [
        r for r in _rows_for(facts, concepts)
        if (_span_days(r) or 0) >= ANNUAL_DAYS[0] and (_span_days(r) or 0) <= ANNUAL_DAYS[1]
    ]
    if len(years) <= offset:
        return None
    return years[-(offset + 1)]


def ttm(facts: CompanyFacts, concepts: list[str]) -> float | None:
    """Trailing twelve months for a flow.

    Four most recent quarters when available; otherwise the latest annual
    figure. A 10-K is up to a year stale by the time anyone reads it, and an
    earnings yield built on stale earnings is wrong in a way that looks right.
    """
    rows = _rows_for(facts, concepts)
    quarters = [
        r for r in rows
        if QUARTER_DAYS[0] <= (_span_days(r) or 0) <= QUARTER_DAYS[1]
    ]
    if len(quarters) >= 4:
        return float(sum(r["val"] for r in quarters[-4:]))

    latest_annual = annual(facts, concepts)
    return float(latest_annual["val"]) if latest_annual else None


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


# --------------------------------------------------------------------------- #
# Metric blocks
# --------------------------------------------------------------------------- #


def value_metrics(facts: CompanyFacts | None, price: float | None,
                  reason: str = "no XBRL financials") -> ValueMetrics:
    """Value. Yields rather than multiples -- see the model docstring."""
    if facts is None:
        return ValueMetrics.unavailable(reason)
    if price is None:
        return ValueMetrics.unavailable("no price available to form a market cap")

    block = _Block(1)
    shares = instant(facts, SHARES)
    if not shares:
        return ValueMetrics.unavailable("share count not reported in XBRL")

    cap = block.compute("market_cap", 1, lambda: price * shares)
    if not cap:
        return ValueMetrics.unavailable("market cap not computable")

    block.compute("earnings_yield_pct", 1,
                  lambda: _safe_div(ttm(facts, NET_INCOME), cap) and
                  _safe_div(ttm(facts, NET_INCOME), cap) * 100.0,
                  note="net income not reported")

    def _fcf_yield() -> float | None:
        cfo, capex = ttm(facts, CFO), ttm(facts, CAPEX)
        if cfo is None:
            return None
        # CapEx is reported as a positive outflow in the cash-flow statement.
        free_cash = cfo - abs(capex or 0.0)
        return free_cash / cap * 100.0
    block.compute("fcf_yield_pct", 1, _fcf_yield, note="operating cash flow not reported")

    def _ebit_to_ev() -> float | None:
        ebit = ttm(facts, OPERATING_INCOME)
        if ebit is None:
            return None
        debt = (instant(facts, LONG_TERM_DEBT) or 0.0) + (instant(facts, SHORT_TERM_DEBT) or 0.0)
        enterprise = cap + debt - (instant(facts, CASH) or 0.0)
        return _safe_div(ebit, enterprise) and ebit / enterprise * 100.0
    block.compute("ebit_to_ev_pct", 1, _ebit_to_ev, note="operating income not reported")

    block.compute("book_to_price", 1, lambda: _safe_div(instant(facts, EQUITY), cap),
                  note="shareholders equity not reported")
    block.compute("sales_to_price", 1, lambda: _safe_div(ttm(facts, REVENUE), cap),
                  note="revenue not reported")
    return block.build(ValueMetrics)


def quality_metrics(facts: CompanyFacts | None,
                    reason: str = "no XBRL financials") -> QualityMetrics:
    if facts is None:
        return QualityMetrics.unavailable(reason)

    block = _Block(1)
    assets = instant(facts, ASSETS)

    block.compute("gross_profitability", 1,
                  lambda: _safe_div(ttm(facts, GROSS_PROFIT), assets),
                  note="gross profit or total assets not reported")

    def _accruals() -> float | None:
        income, cash = ttm(facts, NET_INCOME), ttm(facts, CFO)
        if income is None or cash is None:
            return None
        return _safe_div(income - cash, assets)
    block.compute("accruals", 1, _accruals,
                  note="net income, operating cash flow or assets not reported")

    block.compute("roe_pct", 1,
                  lambda: (_safe_div(ttm(facts, NET_INCOME), instant(facts, EQUITY)) or 0) * 100.0
                  if _safe_div(ttm(facts, NET_INCOME), instant(facts, EQUITY)) is not None else None,
                  note="net income or equity not reported")

    def _roic() -> float | None:
        # NOPAT over invested capital, with a flat 21% US statutory rate rather
        # than a per-filer effective rate: the effective rate swings wildly on
        # one-off items and would make the series less comparable, not more.
        ebit = ttm(facts, OPERATING_INCOME)
        equity = instant(facts, EQUITY)
        if ebit is None or equity is None:
            return None
        debt = (instant(facts, LONG_TERM_DEBT) or 0.0) + (instant(facts, SHORT_TERM_DEBT) or 0.0)
        invested = equity + debt - (instant(facts, CASH) or 0.0)
        return _safe_div(ebit * 0.79, invested) and ebit * 0.79 / invested * 100.0
    block.compute("roic_pct", 1, _roic, note="operating income or equity not reported")

    def _net_debt_to_ebitda() -> float | None:
        ebit, da = ttm(facts, OPERATING_INCOME), ttm(facts, DA)
        if ebit is None:
            return None
        ebitda = ebit + (da or 0.0)
        debt = (instant(facts, LONG_TERM_DEBT) or 0.0) + (instant(facts, SHORT_TERM_DEBT) or 0.0)
        return _safe_div(debt - (instant(facts, CASH) or 0.0), ebitda)
    block.compute("net_debt_to_ebitda", 1, _net_debt_to_ebitda,
                  note="operating income or debt not reported")

    block.compute("interest_coverage", 1,
                  lambda: _safe_div(ttm(facts, OPERATING_INCOME), abs(ttm(facts, INTEREST) or 0) or None),
                  note="interest expense not reported (common for net-cash filers)")

    block.compute("current_ratio", 1,
                  lambda: _safe_div(instant(facts, ASSETS_CURRENT),
                                    instant(facts, LIABILITIES_CURRENT)),
                  note="current assets or liabilities not reported "
                       "(banks and insurers do not present a classified balance sheet)")

    score, missing = piotroski_f_score(facts)
    if score is None:
        block.unavailable("piotroski_f_score",
                          f"needs two comparable years; missing {', '.join(missing)}")
    else:
        block.values["piotroski_f_score"] = float(score)

    return block.build(QualityMetrics)


def piotroski_f_score(facts: CompanyFacts) -> tuple[int | None, list[str]]:
    """The nine binary components, all computable from EDGAR alone.

    A single 0-9 integer is exactly the kind of input a small model uses well
    -- far better than nine ratios it has to weigh itself. Returns
    ``(score, missing_inputs)``; ``None`` when a year-over-year comparison is
    impossible, because a partial score silently means something different.
    """
    missing: list[str] = []

    def pair(concepts: list[str], label: str) -> tuple[float | None, float | None]:
        now, before = annual(facts, concepts, offset=0), annual(facts, concepts, offset=1)
        if now is None or before is None:
            missing.append(label)
            return None, None
        return float(now["val"]), float(before["val"])

    income_now, income_before = pair(NET_INCOME, "net income")
    cfo_now, _ = pair(CFO, "operating cash flow")
    revenue_now, revenue_before = pair(REVENUE, "revenue")
    gross_now, gross_before = pair(GROSS_PROFIT, "gross profit")

    # Instants, not durations -- see instant_years_back.
    assets_now = instant_years_back(facts, ASSETS, years=0)
    assets_before = instant_years_back(facts, ASSETS, years=1)
    if assets_now is None or assets_before is None:
        missing.append("assets")

    if None in (income_now, cfo_now, assets_now, assets_before):
        return None, sorted(set(missing))

    points = 0
    roa_now = _safe_div(income_now, assets_now)
    roa_before = _safe_div(income_before, assets_before)

    # Profitability
    if roa_now and roa_now > 0:
        points += 1
    if cfo_now > 0:
        points += 1
    if roa_now is not None and roa_before is not None and roa_now > roa_before:
        points += 1
    if cfo_now > income_now:      # accruals: cash beats reported earnings
        points += 1

    # Leverage, liquidity and funding
    debt_now = instant_years_back(facts, LONG_TERM_DEBT, years=0)
    debt_before = instant_years_back(facts, LONG_TERM_DEBT, years=1)
    if debt_now is not None and debt_before is not None:
        if _safe_div(debt_now, assets_now) < _safe_div(debt_before, assets_before):
            points += 1

    current_now = _safe_div(
        instant_years_back(facts, ASSETS_CURRENT, years=0),
        instant_years_back(facts, LIABILITIES_CURRENT, years=0),
    )
    current_before = _safe_div(
        instant_years_back(facts, ASSETS_CURRENT, years=1),
        instant_years_back(facts, LIABILITIES_CURRENT, years=1),
    )
    if current_now is not None and current_before is not None and current_now > current_before:
        points += 1

    shares_now = instant_years_back(facts, SHARES, years=0)
    shares_before = instant_years_back(facts, SHARES, years=1)
    if shares_now is not None and shares_before is not None and shares_now <= shares_before:
        points += 1

    # Operating efficiency
    if gross_now is not None and gross_before is not None and revenue_now and revenue_before:
        if _safe_div(gross_now, revenue_now) > _safe_div(gross_before, revenue_before):
            points += 1
    if revenue_now and revenue_before and assets_now and assets_before:
        if _safe_div(revenue_now, assets_now) > _safe_div(revenue_before, assets_before):
            points += 1

    return points, sorted(set(missing))


def growth_metrics(facts: CompanyFacts | None,
                   reason: str = "no XBRL financials") -> GrowthMetrics:
    if facts is None:
        return GrowthMetrics.unavailable(reason)

    block = _Block(1)

    def _yoy(concepts: list[str]) -> float | None:
        now, before = annual(facts, concepts, offset=0), annual(facts, concepts, offset=1)
        if now is None or before is None or not before["val"]:
            return None
        # abs() in the denominator so a swing out of a loss reads positive
        # rather than flipping sign.
        return (float(now["val"]) - float(before["val"])) / abs(float(before["val"])) * 100.0

    block.compute("revenue_yoy_pct", 1, lambda: _yoy(REVENUE),
                  note="needs two annual revenue figures")
    block.compute("eps_yoy_pct", 1, lambda: _yoy(EPS),
                  note="needs two annual EPS figures")

    def _cagr() -> float | None:
        now, old = annual(facts, REVENUE, offset=0), annual(facts, REVENUE, offset=3)
        if now is None or old is None or float(old["val"]) <= 0:
            return None
        return ((float(now["val"]) / float(old["val"])) ** (1 / 3) - 1) * 100.0
    block.compute("revenue_cagr_3y_pct", 1, _cagr, note="needs four annual revenue figures")

    revenue_ttm = ttm(facts, REVENUE)
    block.compute("gross_margin_pct", 1,
                  lambda: (_safe_div(ttm(facts, GROSS_PROFIT), revenue_ttm) or 0) * 100.0
                  if _safe_div(ttm(facts, GROSS_PROFIT), revenue_ttm) is not None else None,
                  note="gross profit or revenue not reported")
    block.compute("operating_margin_pct", 1,
                  lambda: (_safe_div(ttm(facts, OPERATING_INCOME), revenue_ttm) or 0) * 100.0
                  if _safe_div(ttm(facts, OPERATING_INCOME), revenue_ttm) is not None else None,
                  note="operating income or revenue not reported")

    def _margin_change(profit_concepts: list[str]) -> float | None:
        now_p, before_p = annual(facts, profit_concepts, offset=0), annual(facts, profit_concepts, offset=1)
        now_r, before_r = annual(facts, REVENUE, offset=0), annual(facts, REVENUE, offset=1)
        if None in (now_p, before_p, now_r, before_r):
            return None
        current = _safe_div(float(now_p["val"]), float(now_r["val"]))
        prior = _safe_div(float(before_p["val"]), float(before_r["val"]))
        if current is None or prior is None:
            return None
        return (current - prior) * 100.0
    block.compute("gross_margin_change_pct", 1, lambda: _margin_change(GROSS_PROFIT),
                  note="needs two annual gross-profit and revenue figures")
    block.compute("operating_margin_change_pct", 1, lambda: _margin_change(OPERATING_INCOME),
                  note="needs two annual operating-income and revenue figures")

    def _share_change() -> float | None:
        now = instant_years_back(facts, SHARES, years=0)
        before = instant_years_back(facts, SHARES, years=1)
        if now is None or before in (None, 0):
            return None
        return (now - before) / before * 100.0
    block.compute("share_count_change_1y_pct", 1, _share_change,
                  note="needs share counts a year apart")
    return block.build(GrowthMetrics)
