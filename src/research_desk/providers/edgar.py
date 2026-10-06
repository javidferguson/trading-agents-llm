"""SEC EDGAR XBRL company facts. **The best free source in the stack** (§7.3).

Why it earns that description: every fact carries its ``filed`` date, so this
is *genuinely* point-in-time rather than approximately so. Filtering to
``filed <= as_of`` reproduces exactly what a reader could have known on that
date -- including the detail that Q3 numbers were not public until the 10-Q
landed weeks after the quarter ended. Almost nothing else free can do that.

THREE THINGS THAT WILL BITE, ALL MEASURED
=========================================
* **A missing ``User-Agent`` returns 403**, and the response says nothing about
  why. Verified: no header -> 403; ``User-Agent: name email`` -> 200. SEC asks
  for a real contact; it is read from ``SEC_EDGAR_USER_AGENT``.
* **An ETF 404s, with an XML body.** SPY *has* a CIK (0000884394) -- the
  architecture doc assumed it would not -- but
  ``/api/xbrl/companyfacts/CIK0000884394.json`` returns 404 carrying
  ``<?xml ...><Error><Code>NoSuchKey</Code>``. So a naive path breaks twice:
  once on the status, once parsing XML as JSON. That is the Stage 2 exit
  gate's ETF case, and it is handled as an *answer* ("this entity files no
  XBRL financials"), not an error -- an ETF genuinely has no income statement.
* **Tickers are spelled with a dash.** EDGAR writes BRK.B as ``BRK-B``, which
  is a third spelling after our canonical ``BRK.B`` and IB's ``BRK B``.

Rate limit is 10 req/s. The payloads are large -- Apple's is 3.8 MB -- so the
cache is doing real work here, not just being polite.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import httpx

from ..models.market import CompanyFacts
from .base import ProviderError, ProviderSpec
from .cache import Cache

logger = logging.getLogger(__name__)

SPEC = ProviderSpec(
    name="edgar",
    #: Every fact carries `filed`. This is the real thing, not an approximation.
    supports_point_in_time=True,
    min_interval_s=0.12,   # 10 req/s published; leave headroom
    reads_network=True,
    role="fundamentals -- XBRL company facts, genuinely point-in-time",
)

BASE_URL = "https://data.sec.gov"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

METHOD_FACTS = "company_facts"
METHOD_TICKERS = "ticker_map"


class MissingUserAgentError(ProviderError):
    """SEC_EDGAR_USER_AGENT is not set, and EDGAR will 403 without it."""


def _visible(facts: dict[str, Any], as_of: date) -> dict[str, Any]:
    """Drop every fact filed after ``as_of``.

    This is the entire point-in-time guard, and it lives here rather than in
    the caller because §2's tool-promotion path means a model may eventually
    request facts at an arbitrary date. Look-ahead bias is risk number one
    (§15.1), and a fact filed in 2026 must be invisible to a replay of 2024
    even though it *describes* 2024.
    """
    cutoff = as_of.isoformat()
    out: dict[str, Any] = {}
    for taxonomy, concepts in (facts or {}).items():
        kept_concepts: dict[str, Any] = {}
        for name, node in concepts.items():
            kept_units: dict[str, Any] = {}
            for unit, rows in (node.get("units") or {}).items():
                visible = [r for r in rows if (r.get("filed") or "") <= cutoff]
                if visible:
                    kept_units[unit] = visible
            if kept_units:
                kept_concepts[name] = {**node, "units": kept_units}
        if kept_concepts:
            out[taxonomy] = kept_concepts
    return out


class EdgarProvider:
    """Company facts, cached whole and sliced by filing date at read time."""

    spec = SPEC

    def __init__(self, cache: Cache, user_agent: str | None, timeout_s: float = 60.0):
        self._cache = cache
        self._user_agent = user_agent
        self._timeout = timeout_s
        self._tickers: dict[str, str] | None = None

    def _headers(self) -> dict[str, str]:
        if not self._user_agent:
            raise MissingUserAgentError(
                "SEC_EDGAR_USER_AGENT is not set. EDGAR returns 403 without a "
                "User-Agent naming a real contact, and says nothing about why.\n"
                "    SEC_EDGAR_USER_AGENT=\"Your Name you@example.com\"   # in .env"
            )
        return {"User-Agent": self._user_agent, "Accept-Encoding": "gzip, deflate"}

    async def _get_json(self, url: str) -> Any | None:
        """GET and parse. ``None`` on 404 -- which is a real answer here."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(url, headers=self._headers())
            except httpx.HTTPError as exc:
                raise ProviderError(f"EDGAR unreachable: {exc}") from exc

        if response.status_code == 404:
            return None
        if response.status_code == 403:
            raise ProviderError(
                f"EDGAR returned 403 for {url}. The User-Agent is the usual "
                f"cause -- currently {self._user_agent!r}. SEC wants a real "
                "name and email."
            )
        if response.status_code == 429:
            raise ProviderError(
                "EDGAR rate-limited us (429). The published limit is 10 req/s "
                "and the registry paces to it; something bypassed the limiter."
            )
        response.raise_for_status()

        try:
            return response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            # EDGAR answers some misses with an XML error document.
            raise ProviderError(
                f"EDGAR returned non-JSON for {url}: {response.text[:160]!r}"
            ) from exc

    async def cik_for(self, symbol: str) -> str | None:
        """Zero-padded 10-digit CIK, or ``None`` if EDGAR does not list it."""
        if self._tickers is None:
            entry = self._cache.get(SPEC.name, METHOD_TICKERS, as_of=None)
            if entry is None:
                payload = await self._get_json(TICKERS_URL)
                if payload is None:
                    raise ProviderError("EDGAR ticker map unavailable (404)")
                self._cache.put(SPEC.name, METHOD_TICKERS, payload=payload, as_of=None)
                entry = {"payload": payload}
            rows = (entry.get("payload") or {}).values()
            self._tickers = {
                str(r["ticker"]).upper(): f"{int(r['cik_str']):010d}" for r in rows
            }

        wanted = symbol.upper()
        # EDGAR writes share classes with a dash: BRK.B -> BRK-B. A third
        # spelling after our canonical dot and IB's space.
        return self._tickers.get(wanted) or self._tickers.get(wanted.replace(".", "-"))

    async def company_facts(self, symbol: str, as_of: date) -> CompanyFacts | None:
        """XBRL facts for ``symbol`` as they were public on ``as_of``.

        Returns ``None`` when the entity files no XBRL financials. That is an
        answer, not a failure: an ETF has no income statement, and SPY in
        particular has a CIK but 404s on company facts.
        """
        cik = await self.cik_for(symbol)
        if cik is None:
            return None

        # Cached whole (as_of=None) and sliced at read time, like bars: the
        # document is append-only, and a 3.8 MB copy per decision date would
        # be absurd.
        entry = self._cache.get(SPEC.name, METHOD_FACTS, as_of=None, cik=cik)
        if entry is None:
            payload = await self._get_json(f"{BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json")
            if payload is None:
                # Record the miss so an ETF is not re-fetched on every run.
                self._cache.put(SPEC.name, METHOD_FACTS,
                                payload={"no_xbrl": True, "cik": cik},
                                as_of=None, cik=cik)
                return None
            self._cache.put(SPEC.name, METHOD_FACTS, payload=payload, as_of=None, cik=cik)
            entry = {"payload": payload}

        payload = entry.get("payload") or {}
        if payload.get("no_xbrl"):
            return None

        visible = _visible(payload.get("facts") or {}, as_of)
        if not visible:
            return None

        return CompanyFacts(
            symbol=symbol.upper(),
            cik=cik,
            entity_name=payload.get("entityName", symbol.upper()),
            facts=visible,
            as_of=as_of,
        )
