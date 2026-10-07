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
from datetime import date, timedelta
from typing import Any
from xml.etree import ElementTree

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
METHOD_SUBMISSIONS = "submissions"
METHOD_FORM4 = "form4"

#: Form 4 transaction codes, and which ones carry signal.
#:
#: ONLY P AND S ARE DISCRETIONARY. This is the single most important thing
#: about Form 4 and it is easy to get wrong: a real Apple filing in this
#: cache reports an "M" of 374,541 shares acquired (an option exercise) and
#: an "F" of 199,038 disposed (shares withheld to pay the tax on it). Counted
#: naively that reads as an executive buying 374k shares, which is the
#: opposite of informative -- nobody chose to buy anything. Awards (A) and
#: gifts (G) are likewise not decisions about price.
DISCRETIONARY_CODES = frozenset({"P", "S"})
CODE_MEANINGS = {
    "P": "open-market purchase",
    "S": "open-market sale",
    "M": "exercise or conversion of a derivative",
    "F": "shares withheld to cover tax",
    "A": "grant or award",
    "G": "gift",
    "C": "conversion",
    "D": "disposition to the issuer",
}


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


#: Forms that carry a periodic financial report, and nothing else.
#:
#: 10-Q and 10-K are the domestic quarterly and annual. 20-F and 40-F are the
#: annual reports of a foreign private issuer and a Canadian MJDS filer. **6-K
#: is deliberately absent** -- see ``periodic_filings``.
PERIODIC_FORMS: tuple[str, ...] = ("10-Q", "10-K", "20-F", "40-F")


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

    async def insider_transactions(
        self, symbol: str, as_of: date, days: int = 90, max_filings: int = 40
    ) -> list[dict[str, Any]]:
        """Form 4 transactions filed in the ``days`` before ``as_of``.

        Point-in-time by ``filingDate``, which for Form 4 is genuinely close to
        public availability -- insiders must file within two business days.

        Each record carries its transaction ``code`` and a ``discretionary``
        flag. **Read that flag.** Only open-market purchases and sales (P, S)
        are decisions about price; an option exercise (M) and the shares
        withheld to tax it (F) are mechanical, and counting them as buying and
        selling is actively misleading -- see DISCRETIONARY_CODES.

        ``max_filings`` bounds the work: a megacap files hundreds of Form 4s a
        year, each one a separate document fetch.
        """
        cik = await self.cik_for(symbol)
        if cik is None:
            return []

        recent = await self._submissions(cik)
        window_start = as_of - timedelta(days=days)

        wanted: list[tuple[str, str, date]] = []
        for accession, form, filed in zip(
            recent.get("accessionNumber") or [],
            recent.get("form") or [],
            recent.get("filingDate") or [],
        ):
            if form != "4" or not filed:
                continue
            filed_on = date.fromisoformat(filed)
            if not (window_start <= filed_on <= as_of):
                continue
            wanted.append((accession, form, filed_on))
            if len(wanted) >= max_filings:
                break

        transactions: list[dict[str, Any]] = []
        for accession, _form, filed_on in wanted:
            document = await self._form4(cik, accession)
            if document is None:
                continue
            for tx in _parse_form4(document):
                transactions.append({**tx, "filed": filed_on, "accession": accession})
        return transactions

    async def periodic_filings(
        self, symbol: str, as_of: date, forms: tuple[str, ...] = PERIODIC_FORMS
    ) -> list[dict[str, Any]]:
        """Periodic report filings visible on ``as_of``, newest first.

        The input to the earnings-blackout estimate. EDGAR has no forward
        calendar -- it records what has been filed, not what is coming -- so
        the cadence of past 10-Qs is the only free signal for when the next
        report lands. ``intent/earnings.py`` turns these dates into an
        estimate and is explicit that it is one.

        Point-in-time by ``filingDate``, like every other method here: a
        filing that was not public on ``as_of`` cannot inform a decision made
        on ``as_of`` (§7.4).

        **Foreign private issuers return almost nothing useful, by design.**
        TSM files a 20-F once a year and reports quarterly results on 6-K,
        which it also uses for press releases, dividend notices and board
        changes -- 664 of them in this cache against 13 20-Fs. Including 6-K
        would put the median interval at a few days and produce a confident
        estimate that is nonsense, so it is excluded and the caller is left
        with an annual cadence it can correctly refuse to trust.
        """
        cik = await self.cik_for(symbol)
        if cik is None:
            return []

        recent = await self._submissions(cik)
        wanted = set(forms)
        out: list[dict[str, Any]] = []
        for form, filed, reported, accession in zip(
            recent.get("form") or [],
            recent.get("filingDate") or [],
            recent.get("reportDate") or [],
            recent.get("accessionNumber") or [],
        ):
            if form not in wanted or not filed:
                continue
            filed_on = date.fromisoformat(filed)
            if filed_on > as_of:
                continue
            out.append({
                "form": form,
                "filed": filed_on,
                # The period the report covers, which is 4-6 weeks before it is
                # filed. Carried because it distinguishes a late filing from a
                # shifted fiscal calendar.
                "period_end": date.fromisoformat(reported) if reported else None,
                "accession": accession,
            })

        out.sort(key=lambda row: row["filed"], reverse=True)
        return out

    async def _submissions(self, cik: str) -> dict[str, Any]:
        """The filing index. Cached whole; filtered by date at read time."""
        entry = self._cache.get(SPEC.name, METHOD_SUBMISSIONS, as_of=None, cik=cik)
        if entry is None:
            payload = await self._get_json(f"{BASE_URL}/submissions/CIK{cik}.json")
            if payload is None:
                return {}
            self._cache.put(SPEC.name, METHOD_SUBMISSIONS, payload=payload,
                            as_of=None, cik=cik)
            entry = {"payload": payload}
        return ((entry.get("payload") or {}).get("filings") or {}).get("recent") or {}

    async def _form4(self, cik: str, accession: str) -> str | None:
        """One Form 4 XML document, cached by accession number.

        Fetched from the raw ``form4.xml``, not the ``xslF345X06/`` path the
        submissions index names -- that one is the XSL-rendered HTML, which
        returns 200 with ``text/html`` and parses as XML not at all.
        """
        entry = self._cache.get(SPEC.name, METHOD_FORM4, as_of=None, accession=accession)
        if entry is not None:
            payload = entry.get("payload") or {}
            return None if payload.get("missing") else payload.get("xml")

        bare = accession.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{bare}/form4.xml"
        )
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(url, headers=self._headers())
            except httpx.HTTPError as exc:
                raise ProviderError(f"EDGAR unreachable: {exc}") from exc

        if response.status_code >= 400:
            self._cache.put(SPEC.name, METHOD_FORM4, payload={"missing": True},
                            as_of=None, accession=accession)
            return None

        self._cache.put(SPEC.name, METHOD_FORM4, payload={"xml": response.text},
                        as_of=None, accession=accession)
        return response.text


def _parse_form4(xml: str) -> list[dict[str, Any]]:
    """Pull non-derivative transactions out of one Form 4.

    Derivative transactions are skipped: they are options and RSUs, which move
    on vesting schedules rather than on a view about price.
    """
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        logger.debug("unparseable Form 4", exc_info=True)
        return []

    relationship = root.find(".//reportingOwner/reportingOwnerRelationship")
    roles = (
        {child.tag: (child.text or "").strip() for child in relationship}
        if relationship is not None
        else {}
    )

    def text(node: Any, path: str) -> str | None:
        found = node.find(path)
        return found.text.strip() if found is not None and found.text else None

    out: list[dict[str, Any]] = []
    for tx in root.findall(".//nonDerivativeTransaction"):
        code = text(tx, "transactionCoding/transactionCode")
        shares = text(tx, "transactionAmounts/transactionShares/value")
        price = text(tx, "transactionAmounts/transactionPricePerShare/value")
        direction = text(tx, "transactionAmounts/transactionAcquiredDisposedCode/value")
        raw_date = text(tx, "transactionDate/value")

        try:
            share_count = float(shares) if shares else None
            unit_price = float(price) if price else None
        except ValueError:
            continue

        out.append({
            "date": date.fromisoformat(raw_date) if raw_date else None,
            "code": code,
            "code_meaning": CODE_MEANINGS.get(code or "", "other"),
            "discretionary": code in DISCRETIONARY_CODES,
            "acquired": direction == "A",
            "shares": share_count,
            "price": unit_price,
            # None rather than 0 when no price is reported: an exercise or a
            # gift has no transaction price, and a zero would quietly drag a
            # net-dollar total toward nothing.
            "usd": (share_count * unit_price)
            if share_count is not None and unit_price is not None
            else None,
            "is_officer": roles.get("isOfficer") == "true",
            "is_director": roles.get("isDirector") == "true",
            "is_ten_percent_owner": roles.get("isTenPercentOwner") == "true",
            "officer_title": roles.get("officerTitle") or None,
        })
    return out
