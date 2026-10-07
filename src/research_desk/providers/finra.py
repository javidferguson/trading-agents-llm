"""FINRA short interest and daily short-sale volume. Keyless, and genuinely dated.

This is the data node 4 reads instead of Reddit (§7.6): **revealed
positioning** -- what people did with money -- rather than what they say.
Regulated disclosure, so it is barely manipulable, and every record carries a
date.

THE LOOK-AHEAD TRAP, MEASURED
=============================
§7.3 says "short interest carries a settlement date", which is true and not
sufficient. **The settlement date is not the date the number became public.**

Measured on 2026-10-06: the newest settlement FINRA would serve was
**2026-09-15**. The 2026-09-30 settlement had not been published six days
later, because FINRA disseminates roughly eight business days after
settlement. So filtering on ``settlementDate <= as_of`` makes up to twelve
days of future information visible to a replay -- precisely the §15.1 failure
("look-ahead bias will silently invalidate the backtest"), and precisely the
kind that looks like a working backtest.

Hence ``DISSEMINATION_LAG_DAYS``: a record counts as visible only once
``settlementDate + lag <= as_of``.

TWO SMALLER QUIRKS
==================
* **A missing daily file returns 403, not 404.** Verified: Monday 2026-10-05
  gives 200, Tuesday 2026-10-06 (not yet published) gives 403, and so does a
  Saturday. The CDN needs no auth, so 403 means "no such file" here -- but it
  does mean a genuine permission problem is indistinguishable from an
  unpublished date.
* **Sorting needs a partition filter.** The API rejects ``sortFields`` unless
  every partition key has an EQUAL filter, so results are range-filtered and
  sorted client-side.

Daily volume files hold **every symbol**, so they are cached per DATE rather
than per symbol: the first symbol in a run pays for twenty fetches and the
other fourteen are free.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import date, timedelta
from typing import Any

import httpx

from .base import ProviderError, ProviderSpec

logger = logging.getLogger(__name__)

SPEC = ProviderSpec(
    name="finra",
    supports_point_in_time=True,
    min_interval_s=0.5,
    reads_network=True,
    role="revealed positioning -- short interest and daily short-sale volume",
)

SHORT_INTEREST_URL = (
    "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
)
SHORT_VOLUME_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{stamp}.txt"

METHOD_INTEREST = "short_interest"
METHOD_VOLUME = "short_volume_day"

#: Calendar days between a settlement date and public dissemination.
#:
#: FINRA's published schedule is ~8 BUSINESS days, which is 11-12 calendar
#: days. 14 is deliberately conservative: being a couple of days late with a
#: semi-monthly figure costs almost nothing, while being two days early is
#: look-ahead bias that would flatter every backtest.
DISSEMINATION_LAG_DAYS = 14


class FinraProvider:
    """Short interest and short-sale volume, both filtered to what was public."""

    spec = SPEC

    def __init__(self, cache: Any, timeout_s: float = 45.0):
        self._cache = cache
        self._timeout = timeout_s

    # ------------------------------------------------------------- interest

    async def short_interest(self, symbol: str, as_of: date) -> list[dict[str, Any]]:
        """Semi-monthly short-interest records public on or before ``as_of``.

        Newest first. Each record carries ``settlement_date``,
        ``short_shares``, ``days_to_cover``, ``change_pct`` and the
        ``public_from`` date the dissemination lag implies.
        """
        rows = await self._short_interest_rows(symbol)

        visible = []
        for row in rows:
            settled = row.get("settlementDate")
            if not settled:
                continue
            try:
                settled_on = date.fromisoformat(settled)
            except ValueError:
                continue
            public_from = settled_on + timedelta(days=DISSEMINATION_LAG_DAYS)
            if public_from > as_of:
                continue
            visible.append({
                "settlement_date": settled_on,
                "public_from": public_from,
                "short_shares": _number(row.get("currentShortPositionQuantity")),
                "previous_short_shares": _number(row.get("previousShortPositionQuantity")),
                "days_to_cover": _number(row.get("daysToCoverQuantity")),
                "change_pct": _number(row.get("changePercent")),
                "average_daily_volume": _number(row.get("averageDailyVolumeQuantity")),
            })

        visible.sort(key=lambda r: r["settlement_date"], reverse=True)
        return visible

    async def _short_interest_rows(self, symbol: str) -> list[dict[str, Any]]:
        """Raw records for a symbol, cached whole and filtered at read time."""
        entry = self._cache.get(SPEC.name, METHOD_INTEREST, as_of=None, symbol=symbol.upper())
        if entry is not None:
            return (entry.get("payload") or {}).get("rows") or []

        # A wide range rather than "latest": the API refuses sortFields unless
        # every partition key carries an EQUAL filter, so ordering happens here.
        today = date.today()
        body = {
            "limit": 200,
            "compareFilters": [
                {"fieldName": "symbolCode", "fieldValue": symbol.upper(),
                 "compareType": "EQUAL"},
            ],
            "dateRangeFilters": [
                {"fieldName": "settlementDate",
                 "startDate": (today - timedelta(days=5 * 365)).isoformat(),
                 "endDate": today.isoformat()},
            ],
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    SHORT_INTEREST_URL,
                    json=body,
                    headers={"Content-Type": "application/json"},
                )
            except httpx.HTTPError as exc:
                raise ProviderError(f"FINRA unreachable: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"FINRA short interest returned {response.status_code} for "
                f"{symbol}: {response.text[:200]}"
            )

        rows = list(csv.DictReader(io.StringIO(response.text)))
        self._cache.put(SPEC.name, METHOD_INTEREST, payload={"rows": rows},
                        as_of=None, symbol=symbol.upper())
        return rows

    # --------------------------------------------------------------- volume

    async def short_volume(
        self, symbol: str, as_of: date, days: int = 20
    ) -> list[dict[str, Any]]:
        """Daily short-sale volume for ``symbol``, newest first.

        Walks back from ``as_of`` collecting published sessions. The daily
        files are **published one business day late** (verified: on Tuesday the
        newest file is Monday's), so ``as_of`` itself is normally absent -- that
        is correct, not a gap.
        """
        collected: list[dict[str, Any]] = []
        cursor = as_of
        # Calendar days scanned, not sessions: weekends and holidays have no
        # file at all, and 403 is how the CDN says so.
        scanned = 0
        while len(collected) < days and scanned < days * 3:
            scanned += 1
            rows = await self._volume_for_day(cursor)
            if rows is not None:
                row = rows.get(symbol.upper())
                if row:
                    collected.append({"day": cursor, **row})
            cursor -= timedelta(days=1)
        return collected

    async def _volume_for_day(self, day: date) -> dict[str, dict[str, float]] | None:
        """Every symbol's short volume for one session, or ``None`` if unpublished.

        Cached **per date, not per symbol**: one file carries the whole market,
        so a 15-symbol universe makes twenty fetches in total rather than three
        hundred.
        """
        stamp = day.strftime("%Y%m%d")
        entry = self._cache.get(SPEC.name, METHOD_VOLUME, as_of=None, day=stamp)
        if entry is not None:
            payload = entry.get("payload") or {}
            return None if payload.get("unpublished") else payload.get("symbols")

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(SHORT_VOLUME_URL.format(stamp=stamp))
            except httpx.HTTPError as exc:
                raise ProviderError(f"FINRA unreachable: {exc}") from exc

        # 403 is how this CDN reports a file that does not exist -- a weekend,
        # a holiday, or a session not yet published. Verified against Saturday
        # 2026-10-03 and same-day 2026-10-06. Cached so the miss is paid once.
        if response.status_code in (403, 404):
            self._cache.put(SPEC.name, METHOD_VOLUME, payload={"unpublished": True},
                            as_of=None, day=stamp)
            return None
        if response.status_code >= 400:
            raise ProviderError(
                f"FINRA short volume returned {response.status_code} for {stamp}"
            )

        symbols: dict[str, dict[str, float]] = {}
        for row in csv.DictReader(io.StringIO(response.text), delimiter="|"):
            ticker = (row.get("Symbol") or "").upper()
            if not ticker:
                continue
            total = _number(row.get("TotalVolume")) or 0.0
            short = _number(row.get("ShortVolume")) or 0.0
            symbols[ticker] = {
                "short_volume": short,
                "total_volume": total,
                "short_exempt_volume": _number(row.get("ShortExemptVolume")) or 0.0,
                "short_ratio": (short / total) if total > 0 else None,
            }

        self._cache.put(SPEC.name, METHOD_VOLUME, payload={"symbols": symbols},
                        as_of=None, day=stamp)
        return symbols


def _number(raw: Any) -> float | None:
    if raw in (None, "", "null"):
        return None
    try:
        return float(str(raw).replace(",", ""))
    except ValueError:
        return None
