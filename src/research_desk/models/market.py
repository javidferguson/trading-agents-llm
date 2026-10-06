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

    # Arriving with the rest of Stage 2:
    #   2b: value, quality, growth  (EDGAR)
    #   2c: events, positioning, sentiment, macro, regime

    def all_gaps(self) -> dict[str, str]:
        """Every unavailable metric and why, flattened.

        The Stage 2 exit gate is readable off this: a complete snapshot is one
        where every §7.1 metric is either in `available()` or in here.
        """
        out: dict[str, str] = {}
        for block_name in ("trend", "risk", "mean_reversion", "relative_strength"):
            block = getattr(self, block_name)
            for field_name, reason in block.gaps.items():
                out[f"{block_name}.{field_name}"] = reason
        return out

    def metric_count(self) -> tuple[int, int]:
        """``(populated, missing)`` across every block."""
        populated = sum(
            len(getattr(self, name).available())
            for name in ("trend", "risk", "mean_reversion", "relative_strength")
        )
        return populated, len(self.all_gaps())
