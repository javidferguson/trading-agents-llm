"""Fetch daily bars from IB and write them to the provider cache.

THE WRITING HALF OF THE BARS BRIDGE
===================================
This module imports ``ib_async`` and therefore lives in ``execution/`` -- the
only package permitted to (§0, enforced by ``tests/test_layering.py``). It is
run as its own process via ``scripts/fetch_bars.py`` / ``make bars``, never
imported by ``decide`` and never reachable from the ``desk`` CLI.

    execution/bars.py    <- here: ib_async, talks to the Gateway, WRITES cache
    providers/ib.py         reads the cache, imports nothing of the sort

``decide`` then has no IB connection and no price path that could reach the
network, which is stronger point-in-time discipline than a direct fetch would
give (see providers/ib.py).

THE TRAPS, ALL FROM §14 AND ALL MEASURED
========================================
* **``endDateTime`` must be timezone-aware.** A naive value is resolved by IB
  in a zone of its own choosing: measured, naive ``10:00`` returned the
  10:30-10:59 bars while an aware ``10:00 ET`` returned 09:30-09:59. For daily
  bars the stakes are a whole session.
* **The container's ``TIME_ZONE`` must match the exchange** or IB stamps bars
  in UTC. Set in docker-compose.yml; re-stated here because this is the code
  that suffers from it.
* **``assert_paper_account`` fires before anything else.** This process does
  not place orders, so the gate is not strictly required -- it runs anyway,
  because the one thing worse than a missing safety check is one that is
  present in some IB entry points and absent in others.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from ib_async import IB, Stock, util

from ..models.market import Bar, BarSeries
from .safety import assert_paper_account

logger = logging.getLogger(__name__)

#: The exchange whose session these bars belong to. Also the zone every
#: `endDateTime` is stamped in -- see the module docstring.
EXCHANGE_TZ = ZoneInfo("America/New_York")

#: Cache coordinates, and they must match providers/ib.py exactly. A mismatch
#: here is a silent permanent cache miss, so both sides name the same constants.
CACHE_PROVIDER = "ib"
CACHE_METHOD = "daily_bars"

#: 12-1 momentum needs 12 months, the 200-day SMA needs 200 sessions, and the
#: 52-week percentile needs a year. Five years leaves room for all of it plus
#: the own-history percentiles §7.1 asks for.
DEFAULT_DURATION = "5 Y"


def bar_from_ib(row: Any, source: str = CACHE_PROVIDER) -> Bar:
    """**The single conversion point into the internal ``Bar`` type.**

    Carried as a pattern from the ORB engine's ``bars/base.py::bar_from_ib``
    (migration plan §1). One normaliser, because this repo may yet see IB,
    yfinance and a vendor CSV all producing daily bars with different column
    names and adjustment conventions -- and because ``source`` has to be
    recorded per §7.4.
    """
    day = row.date
    if isinstance(day, datetime):
        day = day.date()
    elif isinstance(day, str):
        day = date.fromisoformat(day[:10])

    return Bar(
        day=day,
        open=float(row.open),
        high=float(row.high),
        low=float(row.low),
        close=float(row.close),
        # IB reports -1 for "no volume data"; that is absence, not zero, and
        # letting it through would corrupt dollar ADV and the volume z-score.
        volume=float(row.volume) if row.volume and row.volume > 0 else 0.0,
        source=source,
    )


def _end_of_day(when: date) -> datetime:
    """A timezone-aware end-of-session stamp. See the docstring's first trap."""
    return datetime.combine(when, time(23, 59, 59), tzinfo=EXCHANGE_TZ)


def ib_ticker(symbol: str, provider_symbols: dict[str, str] | None = None) -> str:
    """The spelling IB uses for a symbol.

    Share classes are the whole reason this exists. IB wants ``BRK B`` with a
    SPACE where everyone else writes ``BRK.B``, ``BRK-B`` or ``brk-b.us``, and
    passing the wrong one does not error -- ``qualifyContracts`` just returns
    nothing, which surfaces much later as an empty price series.
    ``config/universe.yaml`` records the per-provider spelling; this reads it.

    The fallback converts ``.`` to a space, which is IB's convention, so a new
    share-class ticker works even before someone adds a mapping for it.
    """
    mapped = (provider_symbols or {}).get("ib")
    if mapped:
        return mapped
    return symbol.upper().replace(".", " ")


async def fetch_series(
    ib: IB,
    symbol: str,
    *,
    end: date | None = None,
    duration: str = DEFAULT_DURATION,
    exchange: str = "SMART",
    currency: str = "USD",
    provider_symbols: dict[str, str] | None = None,
) -> BarSeries:
    """Daily TRADES bars for one symbol, newest bar on or before ``end``."""
    ticker = ib_ticker(symbol, provider_symbols)
    contract = Stock(ticker, exchange, currency)

    # qualifyContractsAsync returns a list that may contain None -- an unknown
    # or misspelled symbol does not raise, it comes back unresolved. Unpacking
    # it blindly turns that into an AttributeError sixty lines away.
    qualified = await ib.qualifyContractsAsync(contract)
    resolved = qualified[0] if qualified else None
    if resolved is None or not getattr(resolved, "conId", 0):
        raise RuntimeError(
            f"IB could not resolve {symbol!r} (sent as {ticker!r}) on "
            f"{exchange}/{currency}. Share classes are the usual cause: IB "
            f"writes BRK.B as 'BRK B' with a space. Add a provider_symbols.ib "
            f"entry for it in config/universe.yaml."
        )

    rows = await ib.reqHistoricalDataAsync(
        resolved,
        # Aware, always. A naive value here is the measured one-session bug.
        endDateTime=_end_of_day(end) if end else "",
        durationStr=duration,
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )
    if not rows:
        raise RuntimeError(
            f"IB returned no bars for {symbol} (as {ticker!r}). It resolved to "
            f"conId {resolved.conId} on "
            f"{resolved.primaryExchange or exchange}, so the contract is fine "
            "-- this is most likely missing historical-data permission, or a "
            "duration longer than IB will serve for this contract."
        )

    bars = [bar_from_ib(row) for row in rows]
    # Keep the canonical ticker, not IB's spelling: `providers/ib.py` looks the
    # cache up by the universe's symbol, and 'BRK B' there would be a
    # permanent, silent miss.
    return BarSeries(symbol=symbol.upper(), source=CACHE_PROVIDER, bars=bars)


async def fetch_and_cache(
    symbols: list[str],
    cache: Any,
    *,
    host: str,
    port: int,
    client_id: int,
    duration: str = DEFAULT_DURATION,
    end: date | None = None,
    provider_symbols: dict[str, dict[str, str]] | None = None,
    pace_s: float = 2.0,
) -> dict[str, int]:
    """Fetch each symbol and write it to the cache. Returns symbol -> bar count.

    ``pace_s`` respects IB's historical-data pacing: roughly 60 requests per
    10 minutes, and identical requests inside 15 seconds are refused. One
    request per symbol over a ~21-symbol universe is nowhere near the limit,
    but a 2 s gap costs 40 seconds total and removes the whole class of
    "pacing violation" errors, which arrive as an unhelpful generic message.
    """
    ib = IB()
    written: dict[str, int] = {}

    await ib.connectAsync(host, port, clientId=client_id)
    try:
        # Not strictly needed -- nothing here places an order -- but a safety
        # gate that is sometimes skipped is a safety gate nobody trusts.
        accounts = assert_paper_account(ib)
        logger.info("connected to %s:%s as %s", host, port, ", ".join(accounts))

        for index, symbol in enumerate(symbols):
            if index:
                await asyncio.sleep(pace_s)
            series = await fetch_series(
                ib, symbol, end=end, duration=duration,
                provider_symbols=(provider_symbols or {}).get(symbol.upper()),
            )
            cache.put(
                CACHE_PROVIDER,
                CACHE_METHOD,
                payload={
                    "symbol": series.symbol,
                    "source": series.source,
                    # TRADES + useRTH returns split/dividend-adjusted history.
                    # Recorded rather than assumed: adjustment rewrites older
                    # bars, which is the one way a stored series stops being
                    # point-in-time honest (providers/ib.py SPEC says so too).
                    "adjusted": True,
                    "duration": duration,
                    "bars": [b.model_dump(mode="json") for b in series.bars],
                },
                as_of=None,  # append-only series; sliced at read time
                symbol=series.symbol,
            )
            written[series.symbol] = len(series)
            logger.info(
                "%s: %d bars, %s .. %s",
                series.symbol, len(series),
                series.bars[0].day, series.bars[-1].day,
            )
    finally:
        ib.disconnect()

    return written
