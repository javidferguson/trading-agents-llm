"""Bars, metric blocks, and the snapshot every LLM node reads.

Two rules from the architecture shape everything here.

**§2 -- all arithmetic happens in Python.** These models carry numbers that
``metrics/indicators.py`` computed. No node ever recomputes one, and no prompt
ever contains a formula.

**§7.4 / the Stage 2 exit gate -- every metric is either populated or
explicitly ``None`` with a reason.** That is enforced structurally here rather
than left to discipline: ``MetricBlock`` refuses to construct if any field is
``None`` without an entry in ``gaps``. The failure mode this prevents is the
one §7.5 is emphatic about -- a silent zero or a neutral default is
indistinguishable from a real measurement, and it corrupts every ablation
downstream. A hole you can see is strictly better.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Bar(BaseModel):
    """One daily OHLCV bar, normalised.

    **The single conversion point into this type** (migration plan §1, carried
    as a pattern). Stooq, IB and possibly yfinance all produce daily bars with
    different field names, timezone handling and adjustment conventions. One
    normaliser, and ``source`` records which one served it -- because when the
    primary fails over, the numbers differ slightly and you will want to know
    which a decision was made on.
    """

    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    #: Which provider served this bar. Not decoration: see §7.4.
    source: str

    @model_validator(mode="after")
    def _ohlc_is_coherent(self) -> "Bar":
        """Catch a mangled parse at the boundary rather than six metrics later.

        A high below the low, or a close outside the range, means the columns
        were misread -- which is exactly the failure mode a CSV schema change
        produces, and it would otherwise surface as a quietly wrong ATR.
        """
        if self.high < self.low:
            raise ValueError(f"{self.day}: high {self.high} < low {self.low}")
        if not (self.low <= self.close <= self.high):
            raise ValueError(
                f"{self.day}: close {self.close} outside [{self.low}, {self.high}]"
            )
        if not (self.low <= self.open <= self.high):
            raise ValueError(
                f"{self.day}: open {self.open} outside [{self.low}, {self.high}]"
            )
        return self


class BarSeries(BaseModel):
    """A symbol's daily history, ordered oldest-first."""

    symbol: str
    source: str
    bars: list[Bar] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ordered_and_unique(self) -> "BarSeries":
        days = [b.day for b in self.bars]
        if days != sorted(days):
            raise ValueError(f"{self.symbol}: bars are not in ascending date order")
        if len(set(days)) != len(days):
            raise ValueError(f"{self.symbol}: duplicate bar dates")
        return self

    def as_of(self, when: date) -> "BarSeries":
        """Everything up to and including ``when``. The point-in-time guard.

        This lives on the series rather than in the caller because §2's
        tool-promotion path means a model may eventually request bars at an
        arbitrary date -- and that is exactly where look-ahead bias creeps back
        in. Truncating here makes it impossible to see tomorrow by accident.
        """
        return BarSeries(
            symbol=self.symbol,
            source=self.source,
            bars=[b for b in self.bars if b.day <= when],
        )

    @property
    def closes(self) -> list[float]:
        return [b.close for b in self.bars]

    @property
    def last_day(self) -> date | None:
        return self.bars[-1].day if self.bars else None

    def __len__(self) -> int:
        return len(self.bars)


class MetricBlock(BaseModel):
    """Base for a group of metrics. A ``None`` must say why.

    Subclasses declare ``float | None`` fields. Construction fails if any of
    them is ``None`` and ``gaps`` does not explain it. That inverts the usual
    default: instead of hoping whoever added a metric remembered to record why
    it was missing, the model will not exist until they have.
    """

    #: field name -> why it could not be computed. Reaches the prompt as an
    #: honest "not available, because ..." rather than as a fabricated number.
    gaps: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _every_none_is_explained(self) -> "MetricBlock":
        unexplained = [
            name
            for name, value in self
            if name != "gaps" and value is None and name not in self.gaps
        ]
        if unexplained:
            raise ValueError(
                f"{type(self).__name__}: {unexplained} are None with no reason in "
                "`gaps`. Architecture §7.5: never a silent None, never a neutral "
                "default -- a missing metric must be distinguishable from a "
                "measured one, or the Stage 8 ablations are meaningless."
            )
        return self

    @classmethod
    def unavailable(cls, reason: str) -> Any:
        """A block where nothing could be computed, and every field says why.

        Needed because the validator above is strict on purpose: an all-``None``
        block with an empty ``gaps`` is exactly the silent-hole case it exists
        to reject, so there is no usable "empty" default. Saying "not computed"
        for every field is both honest and constructible.
        """
        names = [name for name in cls.model_fields if name != "gaps"]
        return cls(gaps={name: reason for name in names})

    def available(self) -> dict[str, float]:
        """Only the metrics that were actually computed."""
        return {
            name: value
            for name, value in self
            if name != "gaps" and value is not None
        }


class TrendMetrics(MetricBlock):
    """Price and trend (§7.1), all from daily OHLCV."""

    close: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    price_vs_sma_20_pct: float | None = None
    price_vs_sma_50_pct: float | None = None
    price_vs_sma_200_pct: float | None = None
    #: +1 golden cross (50 above 200), -1 death cross, 0 if undetermined.
    cross_state: float | None = None
    #: 12-month return EXCLUDING the most recent month. The academically
    #: standard definition -- the excluded month matters, because recent-month
    #: reversal contaminates it.
    momentum_12_1_pct: float | None = None
    #: Opposes momentum at short horizon. Keeping both stops a model conflating
    #: them into one "trend" story.
    reversal_1m_pct: float | None = None
    pct_52w_range: float | None = None
    max_drawdown_1y_pct: float | None = None


class RiskMetrics(MetricBlock):
    """Risk and liquidity (§7.1)."""

    atr_14: float | None = None
    atr_14_pct_of_price: float | None = None
    realized_vol_20d_pct: float | None = None
    realized_vol_60d_pct: float | None = None
    #: 20d / 60d. Below 1 means vol is contracting.
    vol_ratio_20_60: float | None = None
    downside_deviation_pct: float | None = None
    beta_vs_spy: float | None = None
    dollar_adv_20d: float | None = None
    volume_zscore: float | None = None
    #: Amihud illiquidity: mean(|return| / dollar volume). Bigger is thinner.
    amihud_illiquidity: float | None = None


class MeanReversionMetrics(MetricBlock):
    """Mean reversion (§7.1)."""

    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bollinger_pct_b: float | None = None
    dist_from_vwap_20d_pct: float | None = None
    ema_12: float | None = None
    ema_26: float | None = None


class RelativeStrengthMetrics(MetricBlock):
    """Relative strength (§7.1).

    *"The largest gap in the earlier draft. Absolute return tells you almost
    nothing in a trending market."* Needs one extra series each -- the
    benchmark and the sector ETF -- not a peer universe.
    """

    vs_benchmark_1m_pct: float | None = None
    vs_benchmark_3m_pct: float | None = None
    vs_benchmark_12m_pct: float | None = None
    vs_sector_1m_pct: float | None = None
    vs_sector_3m_pct: float | None = None
    vs_sector_12m_pct: float | None = None
    benchmark_symbol: str | None = None
    sector_symbol: str | None = None

    @model_validator(mode="after")
    def _every_none_is_explained(self) -> "RelativeStrengthMetrics":
        # The two symbol fields are labels, not measurements, so they are
        # exempt from the gaps rule -- a missing sector ETF is already recorded
        # as a gap against the metrics that needed it.
        unexplained = [
            name
            for name, value in self
            if name not in {"gaps", "benchmark_symbol", "sector_symbol"}
            and value is None
            and name not in self.gaps
        ]
        if unexplained:
            raise ValueError(f"{type(self).__name__}: {unexplained} are None with no reason")
        return self


#: Every metric block on a snapshot, in display order. One list so adding a
#: block cannot be half-wired -- `all_gaps` and `metric_count` both read it.
BLOCK_NAMES = (
    "trend", "risk", "mean_reversion", "relative_strength",
    "value", "quality", "growth",
    "positioning", "sentiment",
)


class ValueMetrics(MetricBlock):
    """Value (§7.1), from EDGAR facts and the current price.

    **Yields, not multiples.** A P/E of -48 is nonsense a small model will
    reason about anyway; an earnings yield of -2% is a fact it handles. Yields
    also stay finite through negative earnings, which multiples do not -- the
    denominator never crosses zero.
    """

    market_cap: float | None = None
    earnings_yield_pct: float | None = None
    fcf_yield_pct: float | None = None
    #: EBIT/EV rather than EV/EBIT, for the same finiteness reason.
    ebit_to_ev_pct: float | None = None
    book_to_price: float | None = None
    sales_to_price: float | None = None


class QualityMetrics(MetricBlock):
    """Quality (§7.1). Two of these are the point of the whole block."""

    #: Novy-Marx. About as well-evidenced as value and nearly free to compute.
    gross_profitability: float | None = None
    #: (net income - operating cash flow) / assets. The best-documented
    #: *avoid* signal in the free-data universe, and it catches the thing news
    #: sentiment never will. Negative is good: earnings backed by cash.
    accruals: float | None = None
    roe_pct: float | None = None
    roic_pct: float | None = None
    net_debt_to_ebitda: float | None = None
    interest_coverage: float | None = None
    current_ratio: float | None = None
    #: 0-9, all nine components computable from EDGAR alone. A single small
    #: integer is exactly the kind of input a small model uses well.
    piotroski_f_score: float | None = None


class GrowthMetrics(MetricBlock):
    """Growth and dilution (§7.1)."""

    revenue_yoy_pct: float | None = None
    revenue_cagr_3y_pct: float | None = None
    eps_yoy_pct: float | None = None
    gross_margin_pct: float | None = None
    operating_margin_pct: float | None = None
    gross_margin_change_pct: float | None = None
    operating_margin_change_pct: float | None = None
    #: Buyback vs dilution. Free, point-in-time, and routinely ignored.
    #: Negative means the share count shrank.
    share_count_change_1y_pct: float | None = None


class PositioningMetrics(MetricBlock):
    """Revealed positioning (§7.1) -- what people did with money.

    *"This is what node 4 reads instead of Reddit."* Regulated disclosure, so
    barely manipulable, and every figure is dated.

    The insider fields count **only discretionary** transactions: open-market
    purchases and sales. An option exercise and the shares withheld to tax it
    are mechanical, and counting them as buying and selling says an executive
    bought hundreds of thousands of shares when nobody chose to buy anything.
    """

    short_interest_shares: float | None = None
    short_interest_change_pct: float | None = None
    days_to_cover: float | None = None
    #: Days between the settlement date and ``as_of``. Short interest is
    #: semi-monthly and published ~8 business days late, so it is ALWAYS stale;
    #: how stale is information a model should have.
    short_interest_age_days: float | None = None

    short_volume_ratio: float | None = None
    short_volume_ratio_20d: float | None = None

    insider_buy_count: float | None = None
    insider_sell_count: float | None = None
    insider_net_usd: float | None = None
    #: Non-discretionary filings in the window (exercises, awards, tax
    #: withholding). Surfaced so a reader can see the discretionary count is
    #: small because little was chosen, not because little was filed.
    insider_nondiscretionary_count: float | None = None


class SentimentMetrics(MetricBlock):
    """News tone and attention (§7.1), normalised.

    Three rules from §7.1 that matter more than the numbers:

    1. **Percentiles, never raw tone, and never an adjective.** A tone of
       ``-1.7`` means nothing to an 8B model; "12th percentile of this
       symbol's own last year" means something.
    2. **The sign is not the obvious one.** Extreme bullish sentiment and
       abnormal attention are, at short horizons, better documented as
       *contrarian*. Tell an 8B model "sentiment is very positive" and it will
       say BUY every time -- so the prompt gets the percentile and the
       direction of change, and the bull and bear researchers argue about what
       it means.
    3. **Expect a weak signal.** Tone is largely priced in for large caps. It
       earns its place as a tiebreaker and a risk flag -- an unusual spike in
       coverage *volume* is a better reason to pay attention than the tone of
       that coverage.
    """

    news_tone_1d: float | None = None
    news_tone_7d: float | None = None
    news_tone_30d: float | None = None
    #: Where the 7-day tone sits in this symbol's own trailing year. The thing
    #: that actually goes in a prompt.
    news_tone_percentile_1y: float | None = None
    article_count_7d: float | None = None
    coverage_volume_zscore_90d: float | None = None


class MarketSnapshot(BaseModel):
    """Everything deterministic, computed before any model runs.

    ``build_market_snapshot(symbol, as_of)`` produces this with no LLM
    involved. It is the §2 rule made concrete: fetch and compute in Python,
    then hand a model rendered facts and ask only for judgment.
    """

    schema_version: int = 1
    symbol: str
    as_of: date
    built_at: datetime = Field(default_factory=datetime.now)

    #: Which provider served each field group. §7.4: when the primary fails
    #: over the numbers differ slightly, and you want to know which one a
    #: decision was made on.
    sources: dict[str, str] = Field(default_factory=dict)

    bars_available: int = 0
    last_bar_day: date | None = None

    # Defaults are "nothing computed, and here is why" rather than an empty
    # block -- see MetricBlock.unavailable. A snapshot that failed early is
    # then still a valid, honest snapshot.
    trend: TrendMetrics = Field(
        default_factory=lambda: TrendMetrics.unavailable("not computed")
    )
    risk: RiskMetrics = Field(
        default_factory=lambda: RiskMetrics.unavailable("not computed")
    )
    mean_reversion: MeanReversionMetrics = Field(
        default_factory=lambda: MeanReversionMetrics.unavailable("not computed")
    )
    relative_strength: RelativeStrengthMetrics = Field(
        default_factory=lambda: RelativeStrengthMetrics.unavailable("not computed")
    )
    value: ValueMetrics = Field(
        default_factory=lambda: ValueMetrics.unavailable("not computed")
    )
    quality: QualityMetrics = Field(
        default_factory=lambda: QualityMetrics.unavailable("not computed")
    )
    growth: GrowthMetrics = Field(
        default_factory=lambda: GrowthMetrics.unavailable("not computed")
    )
    positioning: PositioningMetrics = Field(
        default_factory=lambda: PositioningMetrics.unavailable("not computed")
    )
    sentiment: SentimentMetrics = Field(
        default_factory=lambda: SentimentMetrics.unavailable("not computed")
    )

    #: The latest EDGAR filing this snapshot's fundamentals came from, and when
    #: it was filed. Without it you cannot tell a stale snapshot from a company
    #: that simply has not reported.
    fundamentals_asof: date | None = None
    fundamentals_form: str | None = None

    # Still to arrive (both gated on free API keys):
    #   events  -- Finnhub earnings calendar, surprises, recommendation trends
    #   macro   -- FRED series, and the regime tag they key (§7.2)

    def all_gaps(self) -> dict[str, str]:
        """Every unavailable metric and why, flattened.

        The Stage 2 exit gate is readable off this: a complete snapshot is one
        where every §7.1 metric is either in `available()` or in here.
        """
        out: dict[str, str] = {}
        for block_name in BLOCK_NAMES:
            block = getattr(self, block_name)
            for field_name, reason in block.gaps.items():
                out[f"{block_name}.{field_name}"] = reason
        return out

    def metric_count(self) -> tuple[int, int]:
        """``(populated, missing)`` across every block."""
        populated = sum(len(getattr(self, name).available()) for name in BLOCK_NAMES)
        return populated, len(self.all_gaps())


class CompanyFacts:
    """One company's XBRL facts, already filtered to what was public at ``as_of``.

    A data shape, which is why it lives here rather than in
    ``providers/edgar.py``: ``metrics/fundamentals.py`` needs the type to read
    it, and reaching into a provider module for that would break the §2 rule
    that every *fetch* goes through the registry. The provider builds these;
    nothing else imports the provider.
    """

    def __init__(self, symbol: str, cik: str, entity_name: str,
                 facts: dict[str, Any], as_of: date):
        self.symbol = symbol
        self.cik = cik
        self.entity_name = entity_name
        self.facts = facts
        self.as_of = as_of

    def concept(self, name: str, taxonomy: str = "us-gaap") -> list[dict[str, Any]]:
        """Every visible fact for one XBRL concept, oldest first."""
        node = (self.facts.get(taxonomy) or {}).get(name) or {}
        rows: list[dict[str, Any]] = []
        for unit_rows in (node.get("units") or {}).values():
            rows.extend(unit_rows)
        return sorted(rows, key=lambda r: (r.get("end") or "", r.get("filed") or ""))

    def has(self, name: str, taxonomy: str = "us-gaap") -> bool:
        return bool(self.concept(name, taxonomy))

    #: Forms that actually carry financial statements. A company also tags
    #: facts in prospectuses (424B2), 8-Ks and disclosure filings (SD), and
    #: those are usually the newest thing in the document -- so a naive "most
    #: recent filed" reports provenance like "424B2" or "2.01 SD" for JPM and
    #: Berkshire, implying the numbers came from a prospectus. Observed; this
    #: list is the fix.
    STATEMENT_FORMS = ("10-K", "10-Q", "20-F", "40-F", "6-K", "11-K")

    @property
    def latest_filing(self) -> tuple[date | None, str | None]:
        """``(filed, form)`` of the most recent financial-statement filing.

        Surfaced on the snapshot because without it you cannot tell a stale
        snapshot from a company that simply has not reported yet -- so it has
        to name the filing the NUMBERS came from, not merely the newest thing
        the filer tagged.
        """
        best_date: date | None = None
        best_form: str | None = None
        for concepts in self.facts.values():
            for node in concepts.values():
                for unit_rows in (node.get("units") or {}).values():
                    for row in unit_rows:
                        filed, form = row.get("filed"), row.get("form") or ""
                        if not filed:
                            continue
                        # Accept amendments too: "10-K/A" starts with "10-K".
                        if not form.startswith(self.STATEMENT_FORMS):
                            continue
                        parsed = date.fromisoformat(filed)
                        if best_date is None or parsed > best_date:
                            best_date, best_form = parsed, form
        return best_date, best_form
