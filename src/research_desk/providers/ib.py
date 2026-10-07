"""Daily bars from Interactive Brokers -- **read from the cache, never fetched here.**

THE §0 SPLIT, MADE CONCRETE
===========================
``decide`` must not import ``ib_async`` at all. Not "should avoid": the import
installs its own event loop policy and already needs ``nest_asyncio``, and
mixing that with a dozen concurrent LLM HTTP calls is the debugging tarpit §0
exists to avoid. ``tests/test_layering.py`` fails the build if it appears
outside ``execution/``.

So the bars bridge has two halves that never share a process:

    execution/bars.py   imports ib_async, talks to the Gateway, WRITES the cache
    providers/ib.py     imports nothing of the sort, READS the cache
    (this file)

That is the same shape as the decide/execute split itself -- two processes
communicating through a file -- applied one level down.

IT IS ALSO BETTER THAN A DIRECT FETCH
=====================================
Not merely a workaround. Because this path is cache-only *by construction*:

* §7.4's "in replay the cache is the only permitted source" is satisfied for
  bars in every mode, not just replay. There is no code path that could reach
  the network for a price, so look-ahead bias cannot enter through here.
* A research run does not need the Gateway up. You refresh bars when
  convenient and iterate on prompts for days afterwards -- which matters,
  because §4 of the migration plan lists "decide stops being fun to iterate
  on" as a legitimate reason to abandon the project.

WHY IB AND NOT STOOQ
====================
§7.3 named Stooq primary and IB the authoritative fallback. Stooq has since put
its CSV endpoint behind a JavaScript proof-of-work browser challenge and its
bulk archive behind a 401, so it is no longer usable programmatically without
defeating a bot check. IB is authoritative, already paid for, and carries no
terms-of-service question.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

from ..models.market import Bar, BarSeries
from .base import ProviderError, ProviderSpec
from .cache import Cache

logger = logging.getLogger(__name__)

SPEC = ProviderSpec(
    name="ib",
    #: Daily bars for days <= as_of do not change, so slicing a stored history
    #: is genuinely point-in-time. The one caveat is retroactive split and
    #: dividend adjustment, which rewrites older bars -- recorded in the
    #: artefact's `adjusted` flag by the fetcher rather than guessed at here.
    supports_point_in_time=True,
    min_interval_s=0.0,
    #: The whole point: this half never touches the network.
    reads_network=False,
    role="daily OHLCV, read from the cache written by execution/bars.py",
)

#: Cache coordinates. ``as_of`` is deliberately absent from the key: a price
#: history is append-only, so the honest unit is one long series sliced at read
#: time, not a separate copy per decision date.
METHOD = "daily_bars"


class IBBarsProvider:
    """Reads the bar artefacts that the execute side wrote."""

    spec = SPEC

    def __init__(self, cache: Cache):
        self._cache = cache

    async def daily_bars(self, symbol: str, as_of: date) -> BarSeries:
        """Daily OHLCV bars for ``symbol``, up to and including ``as_of``.

        Returns the symbol's full stored history truncated at ``as_of``. The
        truncation happens here rather than in the caller so that a future
        tool-calling model cannot see past its own decision date.

        Raises ``ProviderError`` if no bars have been fetched for this symbol,
        with the command that fixes it -- a cache miss here is an operational
        state (the fetcher has not run), not a bug.
        """
        entry = self._cache.get(SPEC.name, METHOD, as_of=None, symbol=symbol.upper())
        if entry is None:
            raise ProviderError(
                f"no cached bars for {symbol.upper()}. The execute side fetches "
                f"them; decide only reads them (§0).\n"
                f"    make bars SYMBOLS={symbol.upper()}\n"
                f"  (needs the IB Gateway up: make gateway-start)"
            )

        payload = entry.get("payload") or {}
        series = BarSeries(
            symbol=payload.get("symbol", symbol.upper()),
            source=payload.get("source", SPEC.name),
            bars=[Bar(**row) for row in payload.get("bars", [])],
        )

        sliced = series.as_of(as_of)
        if not len(sliced):
            raise ProviderError(
                f"{symbol.upper()}: {len(series)} bars cached but none on or "
                f"before {as_of}. Earliest cached bar is "
                f"{series.bars[0].day if series.bars else 'n/a'} -- refetch with "
                f"a longer history, or check as_of."
            )
        if len(sliced) < len(series):
            logger.debug(
                "%s: %d of %d bars are at or before %s",
                symbol, len(sliced), len(series), as_of,
            )
        return sliced

    def stored_symbols(self) -> list[str]:
        """Which symbols have bar artefacts. For `desk providers` and doctor."""
        found = []
        for path in self._cache.entries(SPEC.name):
            if not path.name.startswith(f"{METHOD}-"):
                continue
            try:
                entry = json.loads(path.read_text())
            except Exception:
                continue
            symbol = (entry.get("payload") or {}).get("symbol")
            if symbol:
                found.append(symbol)
        return sorted(set(found))
