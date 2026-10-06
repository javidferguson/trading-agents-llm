"""Per-provider rate limiting, by the clock rather than by trailing sleeps.

The pattern is lifted from ``scripts/gdelt_collect.py``, where it was arrived
at the hard way. A trailing ``sleep()`` after each call ignores how long the
request itself took and leaves the gap between two calls dependent on every
call site remembering to wait. Gating on a monotonic timestamp makes "never
closer than N seconds" true by construction, wherever the call comes from.

``time.monotonic`` and not ``time.time``: an NTP correction or a DST change can
make wall-clock deltas negative, and a negative delta lets a burst straight
through the gate.

The published limits this exists to respect (architecture §7.3):

    EDGAR     10 req/s, and a missing User-Agent 403s
    Finnhub   60 req/min
    GDELT     one request / 5 s, stated only in the body of its 429
    FRED      generous, one fetch per run anyway
"""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Minimum interval between calls, tracked per provider name."""

    def __init__(self, intervals: dict[str, float] | None = None):
        #: provider -> minimum seconds between requests
        self.intervals = dict(intervals or {})
        self._last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def interval_for(self, provider: str) -> float:
        return self.intervals.get(provider, 0.0)

    async def acquire(self, provider: str) -> float:
        """Block until this provider may be called again. Returns seconds waited.

        The per-provider lock matters once the four-analyst fan-out lands at
        Stage 4: without it, concurrent nodes each read a stale ``_last``,
        compute the same remaining wait, and fire simultaneously -- which is a
        burst that looks exactly like no rate limiting at all.
        """
        interval = self.interval_for(provider)
        if interval <= 0:
            return 0.0

        lock = self._locks.setdefault(provider, asyncio.Lock())
        async with lock:
            previous = self._last.get(provider)
            waited = 0.0
            if previous is not None:
                remaining = interval - (time.monotonic() - previous)
                if remaining > 0:
                    await asyncio.sleep(remaining)
                    waited = remaining
            self._last[provider] = time.monotonic()
            return waited
