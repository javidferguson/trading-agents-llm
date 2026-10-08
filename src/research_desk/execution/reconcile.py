"""Compare the book the order was sized against to the account itself.

``execute`` is the **only** process that can see the real positions -- that is
the §0 split working as designed -- and the share count on the proposal was
computed from ``data/portfolio.yaml``. If the file disagrees with the account,
the number is wrong, and approving it means approving arithmetic over a bad
input.

So a mismatch is a **refusal, not a warning in the prompt**. The order gate is
never drawn: there is nothing safe to approve on a screen whose share count is
known to be derived from a stale figure. What the human gets instead is the
per-symbol diff, a REJECTED banner, and one narrow question -- rewrite the book
from the broker? -- followed by "re-run `desk decide`".

Rewriting is also the thing Stage 6 promised and never delivered:
the book's own header says *"At Stage 7, `execute` overwrites
this file from the live paper account and `source:` becomes `ib`."* This is
that.

**Why a tolerance at all.** Share counts are integers and must match exactly --
a position is either 105 shares or it is not. Cash and equity are floats that
move with every dividend, interest accrual and commission, so comparing them
exactly would reject every run. Only the positions gate the order; cash is
reported for information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from ib_async import IB

from ..models.portfolio import PortfolioSnapshot, Position

logger = logging.getLogger(__name__)

#: Account-summary tags worth pulling. NetLiquidation is IB's equity figure and
#: the one ``PortfolioSnapshot.equity`` is meant to reproduce.
ACCOUNT_TAGS = "NetLiquidation,TotalCashValue,BuyingPower"


@dataclass(frozen=True)
class Holding:
    """One position as the broker reports it."""

    symbol: str
    quantity: float
    avg_cost: float
    market_price: float | None = None


@dataclass(frozen=True)
class Row:
    """One line of the reconciliation table."""

    symbol: str
    book_qty: float
    broker_qty: float

    @property
    def matches(self) -> bool:
        # Exact: a share count is an integer and a difference of one share is a
        # real difference, not noise.
        return abs(self.book_qty - self.broker_qty) < 1e-9

    @property
    def state(self) -> str:
        if self.matches:
            return "ok"
        if self.book_qty == 0:
            return "NOT IN BOOK"
        if self.broker_qty == 0:
            return "NOT AT BROKER"
        return "QUANTITY DIFFERS"


@dataclass(frozen=True)
class Reconciliation:
    """The whole comparison, and whether it is safe to trade on."""

    rows: tuple[Row, ...] = ()
    book_equity: float = 0.0
    broker_equity: float | None = None
    book_cash: float = 0.0
    broker_cash: float | None = None
    broker_account: str = ""

    @property
    def mismatched(self) -> tuple[Row, ...]:
        return tuple(r for r in self.rows if not r.matches)

    @property
    def matches(self) -> bool:
        return not self.mismatched

    @property
    def equity_drift_pct(self) -> float | None:
        if not self.broker_equity or self.book_equity <= 0:
            return None
        return 100.0 * (self.broker_equity - self.book_equity) / self.book_equity


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


async def holdings_from_ib(ib: IB) -> dict[str, Holding]:
    """Current positions, keyed by symbol.

    Uses ``ib.positions()`` rather than ``portfolio()``: positions are reported
    per account without needing a market-data subscription, and the market
    price is not needed to decide whether the share counts agree.
    """
    out: dict[str, Holding] = {}
    for item in ib.positions():
        contract = item.contract
        # Stocks only. An option or future in the account is not something this
        # desk put there, and silently folding one into an equity book would
        # make the weights nonsense.
        if getattr(contract, "secType", "STK") != "STK":
            logger.warning(
                "ignoring a non-stock position in the account: %s %s",
                contract.secType, getattr(contract, "localSymbol", "?"),
            )
            continue
        # IB spells BRK.B as "BRK B"; the book uses the canonical ticker.
        symbol = (getattr(contract, "symbol", "") or "").replace(" ", ".").upper()
        if not symbol:
            continue
        out[symbol] = Holding(
            symbol=symbol,
            quantity=float(item.position),
            avg_cost=float(getattr(item, "avgCost", 0.0) or 0.0),
        )
    return out


async def account_values(ib: IB) -> dict[str, float | None]:
    """NetLiquidation, cash and buying power, as floats."""
    rows = await ib.accountSummaryAsync()
    wanted = set(ACCOUNT_TAGS.split(","))
    found: dict[str, float | None] = {}
    for row in rows:
        if row.tag in wanted and row.currency in ("USD", ""):
            found[row.tag] = _float(row.value)
    return found


def compare(
    book: PortfolioSnapshot,
    holdings: dict[str, Holding],
    *,
    broker_equity: float | None = None,
    broker_cash: float | None = None,
    broker_account: str = "",
) -> Reconciliation:
    """Build the comparison. Pure, so it is testable without a broker."""
    symbols = sorted(book.symbols | set(holdings))
    rows = tuple(
        Row(
            symbol=symbol,
            book_qty=(book.get(symbol).quantity if book.get(symbol) else 0.0),
            broker_qty=(holdings[symbol].quantity if symbol in holdings else 0.0),
        )
        for symbol in symbols
    )
    return Reconciliation(
        rows=rows,
        book_equity=book.equity,
        broker_equity=broker_equity,
        book_cash=book.cash,
        broker_cash=broker_cash,
        broker_account=broker_account,
    )


def render(result: Reconciliation) -> str:
    """The diff the human reads. Shown whether or not it matched.

    Printed on a match too, because "the book agrees with the account" is worth
    one line of reassurance before approving an order sized from it.
    """
    lines = [
        "BOOK vs BROKER"
        + (f"  (account {result.broker_account})" if result.broker_account else ""),
        f"  {'symbol':8} {'book':>10} {'broker':>10}  state",
    ]
    for row in result.rows:
        flag = "" if row.matches else "  <<"
        lines.append(
            f"  {row.symbol:8} {row.book_qty:10,.0f} {row.broker_qty:10,.0f}"
            f"  {row.state}{flag}"
        )

    lines.append("")
    lines.append(
        f"  equity   book {result.book_equity:>14,.2f}"
        + (f"   broker {result.broker_equity:>14,.2f}"
           if result.broker_equity is not None else "   broker n/a")
    )
    lines.append(
        f"  cash     book {result.book_cash:>14,.2f}"
        + (f"   broker {result.broker_cash:>14,.2f}"
           if result.broker_cash is not None else "   broker n/a")
    )

    drift = result.equity_drift_pct
    if drift is not None and abs(drift) > 0.5:
        # Not a gate -- equity moves with every mark -- but a large gap usually
        # means the book is from a different week, not a different cent.
        lines.append(
            f"  NOTE: broker equity is {drift:+.1f}% from the book's figure. "
            "Positions gate the order; this is context."
        )

    return "\n".join(lines)


def render_rejection(result: Reconciliation, book_path: object) -> str:
    """The REJECTED banner. No approval prompt follows this."""
    bad = result.mismatched
    lines = [
        "",
        "=" * 72,
        "REJECTED -- THE BOOK DISAGREES WITH THE BROKER. NO ORDER WAS SHOWN.",
        "=" * 72,
        f"  {len(bad)} position(s) do not match:",
    ]
    for row in bad:
        lines.append(
            f"    {row.symbol:8} book {row.book_qty:,.0f} vs broker "
            f"{row.broker_qty:,.0f}   ({row.state})"
        )
    lines += [
        "",
        "  The share count on this proposal was computed from the book, so a",
        "  book that is wrong makes the share count wrong. You are not being",
        "  asked to approve it: there is nothing safe to approve.",
        "",
        f"  Fix: rewrite {book_path} from the broker, then re-run `desk decide`",
        "  so the order is sized against what the account actually holds.",
        "=" * 72,
    ]
    return "\n".join(lines)


async def book_from_ib(
    ib: IB,
    *,
    as_of: date | None = None,
    stale_after_days: int | None = None,
) -> PortfolioSnapshot:
    """Build a ``PortfolioSnapshot`` from the account. ``source="ib"``.

    This is what ``data/portfolio.yaml``'s header has promised since Stage 6.

    **Marks come from the broker's average cost, not from a quote.** A position
    needs a ``last_price`` and fetching 15 live quotes to write a file would be
    slow, rate-limited and no more accurate than the next ``make
    portfolio-refresh`` against the bars cache. So the written book is correct
    about *quantities* -- which is what gates an order -- and its marks are
    refreshed by the existing command. ``source="ib"`` records which path wrote
    it, exactly as ``Bar.source`` does.
    """
    when = as_of or date.today()
    holdings = await holdings_from_ib(ib)
    values = await account_values(ib)

    positions = [
        Position(
            symbol=holding.symbol,
            quantity=holding.quantity,
            avg_cost=max(holding.avg_cost, 0.0),
            # avg_cost is a real number from the broker and a defensible mark
            # for a file whose marks are refreshed separately. Never zero: the
            # schema requires a positive price, and a zero would make the
            # position weightless rather than unpriced.
            last_price=holding.avg_cost if holding.avg_cost > 0 else 0.01,
            marked_on=when,
        )
        for holding in holdings.values()
        if abs(holding.quantity) > 1e-9
    ]

    cash = values.get("TotalCashValue")
    if cash is None:
        raise RuntimeError(
            "IB did not report TotalCashValue, so the book's cash balance "
            "cannot be written. Refusing to guess it: equity is derived from "
            "cash plus positions, and a wrong cash figure silently rescales "
            "every weight."
        )

    body: dict = {
        "as_of": when,
        "cash": cash,
        "positions": positions,
        "source": "ib",
        "note": (
            f"Written from the paper account on {when.isoformat()} by "
            "`execute`. Quantities are the broker's; marks are each position's "
            "average cost -- run `make portfolio-refresh` to mark to market."
        ),
    }
    if stale_after_days is not None:
        body["stale_after_days"] = stale_after_days

    return PortfolioSnapshot(**body)


__all__ = [
    "ACCOUNT_TAGS",
    "Holding",
    "Reconciliation",
    "Row",
    "account_values",
    "book_from_ib",
    "compare",
    "holdings_from_ib",
    "render",
    "render_rejection",
]
