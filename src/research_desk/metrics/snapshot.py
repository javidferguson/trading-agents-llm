"""``build_market_snapshot()`` -- everything deterministic, before any model runs.

This is the §2 rule made concrete. Node 0 of the graph calls this; nodes 2-13
receive its rendered output and are asked only for judgment. No LLM is involved
here and none ever should be.

The Stage 2 exit gate reads off this function: *"`snapshot SYMBOL=SPY` prints a
complete ``MarketSnapshot`` with every §7.1 metric either populated or
explicitly ``None`` with a reason, no LLM."* The "with a reason" half is
enforced by ``MetricBlock``, which refuses to construct otherwise.
"""

from __future__ import annotations

import logging
from datetime import date

from ..models.market import (
    BarSeries,
    MarketSnapshot,
    MeanReversionMetrics,
    RelativeStrengthMetrics,
    RiskMetrics,
    TrendMetrics,
)
from ..providers.base import ProviderError
from . import fundamentals, indicators, positioning, sentiment

logger = logging.getLogger(__name__)


async def _optional_series(registry, symbol: str | None, as_of: date) -> BarSeries | None:
    """Fetch a supporting series, tolerating its absence.

    The benchmark and the sector ETF are *supporting* inputs: without them you
    lose relative strength and beta, which is a real loss but not a reason to
    refuse to produce a snapshot. The symbol's own bars are different -- see
    ``build_market_snapshot``.
    """
    if not symbol:
        return None
    try:
        return await registry.daily_bars(symbol, as_of)
    except ProviderError as exc:
        logger.info("supporting series %s unavailable: %s", symbol, str(exc).splitlines()[0])
        return None


async def build_market_snapshot(
    registry,
    symbol: str,
    as_of: date,
    *,
    benchmark_symbol: str | None = None,
    sector_symbol: str | None = None,
) -> MarketSnapshot:
    """Fetch and compute everything numeric for one symbol as of one date.

    Raises ``ProviderError`` when the symbol's own bars are missing. That is
    deliberately *not* degraded to an empty snapshot: a market report with no
    prices is not a weaker report, it is a different and useless object, and
    the error names the command that fixes it.
    """
    series = await registry.daily_bars(symbol, as_of)

    benchmark = await _optional_series(registry, benchmark_symbol, as_of)
    sector = await _optional_series(registry, sector_symbol, as_of)

    # Fundamentals are optional in the same way the benchmark is: an ETF has
    # none and that is a fact about the instrument, not a failure to fetch.
    facts = None
    facts_reason = "no XBRL financials (ETFs and trusts file none)"
    try:
        facts = await registry.company_facts(symbol, as_of)
    except ProviderError as exc:
        facts_reason = f"EDGAR unavailable: {str(exc).splitlines()[0]}"
        logger.info("%s: %s", symbol, facts_reason)

    # Positioning and sentiment are independently optional: FINRA needs no
    # key but can be down, and GDELT depends on a collector that may never
    # have run. Each failure becomes its own explained gap rather than
    # sinking the snapshot.
    async def _optional(coro, label: str):
        try:
            return await coro, None
        except ProviderError as exc:
            reason = f"{label} unavailable: {str(exc).splitlines()[0]}"
            logger.info("%s: %s", symbol, reason)
            return None, reason

    interest, interest_reason = await _optional(
        registry.short_interest(symbol, as_of), "FINRA short interest")
    volume, _ = await _optional(
        registry.short_volume(symbol, as_of), "FINRA short volume")
    insiders, _ = await _optional(
        registry.insider_transactions(symbol, as_of), "EDGAR Form 4")
    tone, _ = await _optional(registry.news_tone(symbol, as_of), "GDELT")

    last_close = series.bars[-1].close if len(series) else None
    filed, form = facts.latest_filing if facts else (None, None)

    snapshot = MarketSnapshot(
        symbol=series.symbol,
        as_of=as_of,
        bars_available=len(series),
        last_bar_day=series.last_day,
        sources={"bars": series.source},
        trend=indicators.trend_metrics(series),
        risk=indicators.risk_metrics(series, benchmark),
        mean_reversion=indicators.mean_reversion_metrics(series),
        relative_strength=indicators.relative_strength_metrics(
            series, benchmark, sector,
            benchmark_symbol=benchmark_symbol,
            sector_symbol=sector_symbol,
        ),
        value=fundamentals.value_metrics(facts, last_close, facts_reason),
        quality=fundamentals.quality_metrics(facts, facts_reason),
        growth=fundamentals.growth_metrics(facts, facts_reason),
        positioning=positioning.positioning_metrics(
            interest, volume, insiders, as_of,
            unavailable_reason=interest_reason if volume is None and insiders is None
            else None,
        ),
        sentiment=sentiment.sentiment_metrics(tone, as_of),
        fundamentals_asof=filed,
        fundamentals_form=form,
    )

    if benchmark is not None:
        snapshot.sources["benchmark"] = f"{benchmark.symbol}@{benchmark.source}"
    if sector is not None:
        snapshot.sources["sector"] = f"{sector.symbol}@{sector.source}"
    if facts is not None:
        snapshot.sources["fundamentals"] = f"edgar:CIK{facts.cik}"

    # A stale last bar is a quiet way to make a decision on old prices. Warn
    # rather than refuse: a weekend or a holiday is a legitimate three-day gap.
    if series.last_day and (as_of - series.last_day).days > 5:
        logger.warning(
            "%s: newest cached bar is %s, %d days before as_of=%s. Refresh with "
            "`make bars`.",
            symbol, series.last_day, (as_of - series.last_day).days, as_of,
        )

    return snapshot


def universe_context(universe: dict, symbol: str) -> tuple[str | None, str | None]:
    """``(benchmark, sector_etf)`` for a symbol, from universe.yaml.

    The benchmark is global; the sector ETF is per symbol and hand-maintained,
    because free sector classification is poor (§7.1).
    """
    benchmark = universe.get("benchmark")
    sector = None
    for entry in universe.get("symbols", []):
        if entry["symbol"].upper() == symbol.upper():
            sector = entry.get("sector_etf")
            break
    # The benchmark measured against itself is a column of zeros.
    if benchmark and benchmark.upper() == symbol.upper():
        benchmark = None
    return benchmark, sector
