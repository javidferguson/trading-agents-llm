"""The single entry point for every fetch. Nothing imports a provider directly.

Architecture §2, the tool-promotion path:

> *"Every fetch lives in ``providers/registry.py`` and is reached through it.
> Nothing imports a provider module directly."*

The payoff is deferred but concrete. When a local model can be trusted to
select its own fetches -- or when a node routes to a hosted model that already
can -- the upgrade is: generate tool schemas from these signatures, hand them
to a tool-capable node, and keep the deterministic path as both the fallback
and the replay substrate. Nothing above needs rewriting. That only holds if
every call site goes through here, which is why the layering test guards it.

The registry also owns the two cross-cutting rules, so no individual provider
can forget them:

* **Rate limiting**, per provider, by the clock.
* **Replay discipline (§7.4).** In ``mode="replay"`` any provider that would
  reach the network is refused, and any provider that cannot do point-in-time
  raises. Look-ahead bias is the number one way agentic backtests produce
  fictional Sharpe ratios.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from ..config import Settings, load_yaml
from ..models.market import BarSeries
from ..models.modes import RunMode
from .base import NetworkForbiddenError, PointInTimeError, ProviderError, ProviderSpec
from .cache import Cache
from .edgar import CompanyFacts, EdgarProvider
from .ib import IBBarsProvider
from .limiter import RateLimiter

logger = logging.getLogger(__name__)

#: Fallback intervals when providers.yaml does not state one. Published limits
#: are in §7.3; these are the conservative reading of them.
DEFAULT_INTERVALS = {
    "edgar": 0.12,     # 10 req/s published; leave headroom
    "finnhub": 1.1,    # 60/min
    "gdelt": 5.0,      # stated in the body of its own 429
    "finra": 0.5,
    "fred": 0.5,
    "cboe": 1.0,
    "ib": 0.0,         # cache reader, no network
}


class ProviderRegistry:
    """Every fetch in the system goes through one of these methods."""

    def __init__(
        self,
        *,
        mode: RunMode,
        cache: Cache,
        limiter: RateLimiter,
        bars: Any,
        edgar: Any = None,
    ):
        self.mode = mode
        self.cache = cache
        self.limiter = limiter
        self._bars = bars
        self._edgar = edgar

    @classmethod
    def from_config(
        cls,
        settings: Settings,
        *,
        mode: RunMode | None = None,
        config: dict[str, Any] | None = None,
    ) -> "ProviderRegistry":
        body = config if config is not None else load_yaml("providers.yaml")
        declared = body.get("providers") or {}

        intervals = dict(DEFAULT_INTERVALS)
        for name, spec in declared.items():
            if "rate_limit_per_s" in spec and spec["rate_limit_per_s"]:
                intervals[name] = 1.0 / float(spec["rate_limit_per_s"])
            elif "rate_limit_per_min" in spec and spec["rate_limit_per_min"]:
                intervals[name] = 60.0 / float(spec["rate_limit_per_min"])

        cache = Cache(settings.cache_dir)
        return cls(
            mode=mode or settings.mode,
            cache=cache,
            limiter=RateLimiter(intervals),
            bars=IBBarsProvider(cache),
            edgar=EdgarProvider(cache, settings.sec_user_agent),
        )

    # ----------------------------------------------------------------- guards

    def _guard(self, spec: ProviderSpec) -> None:
        """Enforce §7.4 before any provider runs."""
        if self.mode.allows_network:
            return
        if not spec.supports_point_in_time:
            raise PointInTimeError(
                f"provider {spec.name!r} declares supports_point_in_time=False "
                f"and cannot be used in mode={self.mode.value}. It would return "
                "today's value for a past date, which makes the backtest "
                "fiction rather than merely wrong (§7.5, §15.1)."
            )
        if spec.reads_network:
            raise NetworkForbiddenError(
                f"provider {spec.name!r} would reach the network, but "
                f"mode={self.mode.value} permits the cache only (§7.4). "
                "Populate the cache in live mode first."
            )

    async def _rate_limited(self, spec: ProviderSpec) -> None:
        # A rate limit exists to protect a remote service. A cache reader has
        # no remote, and providers.yaml's `ib: rate_limit_per_s: 1` was written
        # for the real fetch -- applying it here would cost 2s per snapshot
        # (symbol + benchmark + sector) to throttle three file reads.
        if not spec.reads_network:
            return
        waited = await self.limiter.acquire(spec.name)
        if waited > 0.5:
            logger.debug("waited %.1fs for %s's rate limit", waited, spec.name)

    # --------------------------------------------------------------- fetches

    async def daily_bars(self, symbol: str, as_of: date) -> BarSeries:
        """Daily OHLCV bars for ``symbol``, up to and including ``as_of``.

        Served from the cache written by the execute side -- ``decide`` holds no
        IB connection (§0). Raises if the symbol has never been fetched.
        """
        spec = self._bars.spec
        self._guard(spec)
        await self._rate_limited(spec)
        return await self._bars.daily_bars(symbol, as_of)

    async def company_facts(self, symbol: str, as_of: date) -> CompanyFacts | None:
        """XBRL company facts for ``symbol`` as they were public on ``as_of``.

        ``None`` means the entity files no XBRL financials -- an answer, not a
        failure. An ETF has no income statement, and SPY in particular has a
        CIK but 404s on company facts.
        """
        if self._edgar is None:
            raise ProviderError("no EDGAR provider configured")
        spec = self._edgar.spec
        self._guard(spec)
        await self._rate_limited(spec)
        return await self._edgar.company_facts(symbol, as_of)

    # Arriving with Stage 2c, each one typed, each taking as_of:
    #   company_news(symbol, as_of, days)   -- Finnhub
    #       earnings_calendar(symbol, as_of)    -- Finnhub
    #       short_interest(symbol, as_of)       -- FINRA
    #       insider_transactions(symbol, as_of) -- EDGAR Form 4
    #       macro_series(series_id, as_of)      -- FRED
    #       news_tone(symbol, as_of)            -- GDELT cache
    #       put_call_ratio(as_of)               -- CBOE

    def describe(self) -> list[dict[str, Any]]:
        """What is wired up, for `desk providers`."""
        rows = []
        for provider in (p for p in (self._bars, self._edgar) if p is not None):
            spec: ProviderSpec = provider.spec
            rows.append({
                "name": spec.name,
                "role": spec.role,
                "point_in_time": spec.supports_point_in_time,
                "network": spec.reads_network,
                "min_interval_s": self.limiter.interval_for(spec.name),
            })
        return rows
