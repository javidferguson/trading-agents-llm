"""Quote, preflight and place. **The only module that calls ``placeOrder``.**

Three rules carried from the options engine (§9), and each one is here because
skipping it cost something real:

* **``assert_paper_account()`` after connect AND again immediately before every
  ``placeOrder``.** The port is not a safety guarantee -- a socat mapping, an
  env var or a reconnect landing elsewhere can all point a nominally-paper
  setup at a live account. ``ib.managedAccounts()`` is the only thing that
  actually knows. The second call is the one that catches the reconnect.
* **``ib.whatIfOrder()`` before the human sees the prompt.** Free, places
  nothing, returns margin impact and commission. Those numbers go *into* the
  confirmation rather than arriving afterwards.
* **Marketable ``LimitOrder``, never ``MarketOrder``.** With delayed data you
  are looking at a 15-minute-old price, so a market order is an open-ended
  instruction priced off a number you cannot see.
  ``tests/execution/test_broker.py`` asserts ``MarketOrder`` is never imported.

**A BUY carries its own stop, and that is not a nicety.** Sizing's fourth cap is
``cap_risk = per_trade_risk_pct / stop_dist`` -- every position is sized on the
arithmetic that an 8% stop bounds the loss to 0.75% of equity. Without a stop
order that cap is a claim the system does not honour, which is the same class of
defect as a compliance veto that cannot fire. So the entry goes out as a parent
with ``transmit=False`` and the stop as a child with ``parentId`` set and
``transmit=True``: nothing reaches the market until the pair is complete.

*The honest limit of that:* the stop covers **only the shares just bought**. Add
to a symbol that already has one and the account ends up with two stops rather
than one for the whole position. Cancel-and-replace stop management is a real
piece of work and is recorded in FOLLOWUPS rather than half-built here.

A SELL gets no stop. It is already reducing exposure, and a stop on an exit is
a second exit.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from ib_async import IB, LimitOrder, Order, Stock, StopOrder, Trade

from ..models.modes import RunMode
from ..models.orders import OrderPlan
from ..models.state import FinalDecision
from .bars import EXCHANGE_TZ, ib_ticker
from .confirmation import Preflight
from .journal import Journal
from .safety import assert_can_trade, assert_paper_account

logger = logging.getLogger(__name__)

#: US equities trade in cents. Submitting a price with more precision earns a
#: rejection for violating the minimum tick, which arrives as a generic error.
TICK = 0.01

#: Delayed market data, no subscription required. Error 10091 is a substitution
#: notice rather than a failure, and ``logging_setup`` already demotes it to
#: DEBUG so it does not bury real problems once per contract.
DELAYED_MARKET_DATA = 3

#: How long to wait for a delayed quote before falling back. Delayed ticks
#: arrive in a few seconds; a longer wait just delays the fallback that is
#: going to happen anyway outside market hours.
QUOTE_TIMEOUT_S = 6.0

#: Statuses from which an order will not move on its own. IB's own vocabulary;
#: everything else (``PendingSubmit``, ``PreSubmitted``, ``Submitted``) means
#: the order is still working.
TERMINAL_STATUSES = frozenset({"Filled", "Cancelled", "ApiCancelled", "Inactive"})

#: How long a batch waits for one entry order to stop moving before it refuses
#: to size the next one.
#:
#: Only ``execute --all`` uses this. A single order does not need it: the human
#: is watching, and the book is re-read from the broker next run. A batch has to
#: know what filled before it can check the next order against the book, and
#: "probably filled" is not an input a compliance check can use.
#:
#: 30s is long for a marketable limit on a liquid name and short enough that a
#: stuck order does not strand the operator. Timing out is not an error -- see
#: ``wait_for_terminal``.
FILL_TIMEOUT_S = 30.0

#: Statuses that mean IB has accepted a cancel request.
#:
#: **``PendingCancel`` belongs here and NOT in ``TERMINAL_STATUSES``.**
#: ``ib_async.cancelOrder`` sets a working order to ``PendingCancel``, which
#: will move on its own to ``Cancelled`` -- so it is not terminal by that set's
#: definition, and waiting for a terminal status after a cancel would time out
#: on a cancel that worked. A held parent (``PendingSubmit`` with
#: ``transmit=False``) goes straight to ``Cancelled``.
CANCEL_ACKNOWLEDGED = frozenset(
    {"PendingCancel", "Cancelled", "ApiCancelled", "Inactive"}
)

#: How long to wait for IB to acknowledge a cancel. Short: the request either
#: lands or it does not, and there is no fill to wait on.
CANCEL_TIMEOUT_S = 10.0


class NoQuoteError(RuntimeError):
    """No usable price for this contract, from any field.

    Raised rather than defaulted. A limit order priced off a guessed number is
    worse than no order, and the one thing worse than either is a *silent*
    guess -- so the quote carries its own ``source`` and this raises when there
    is nothing to put in it.
    """


@dataclass(frozen=True)
class Quote:
    """A price, and where it came from.

    ``source`` exists because the fallback chain is not an implementation
    detail: a limit priced off the previous close outside market hours behaves
    very differently from one priced off a live spread, and the human approving
    it should be told which they are looking at.
    """

    symbol: str
    bid: float | None
    ask: float | None
    last: float | None
    close: float | None
    source: str

    @property
    def mid(self) -> float | None:
        if self.bid and self.ask and self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return None

    @property
    def reference(self) -> float:
        """The price a limit is computed from."""
        for value in (self.mid, self.last, self.close):
            if value and value > 0:
                return value
        raise NoQuoteError(f"{self.symbol}: no usable price in any field")


def contract_for(symbol: str, provider_symbols: dict[str, str] | None = None) -> Stock:
    """A SMART/USD stock contract, spelled the way IB spells it.

    Reuses ``bars.ib_ticker`` so there is one definition of "IB calls BRK.B
    'BRK B'". Two spellings of that mapping is how one of them goes stale.
    """
    return Stock(ib_ticker(symbol, provider_symbols), "SMART", "USD")


async def connect(
    *,
    host: str,
    port: int,
    client_id: int,
    mode: RunMode,
) -> tuple[IB, list[str]]:
    """Connect, then prove it is a paper account. **Safety gate, fire 1 of 2.**

    ``assert_can_trade`` runs too: both replay modes are driven by historical
    prices, so an order from one would be meaningless. It is enforced inside
    this path rather than only at the call site so a future caller cannot route
    around it by constructing a client directly.
    """
    assert_can_trade(mode)

    ib = IB()
    await ib.connectAsync(host, port, clientId=client_id)
    try:
        accounts = assert_paper_account(ib)
    except Exception:
        # Do not leave a socket open to an account we have just refused to
        # trade on.
        ib.disconnect()
        raise

    logger.info("connected to %s:%s as %s", host, port, ", ".join(accounts))
    return ib, accounts


async def quote(ib: IB, contract: Stock, *, timeout_s: float = QUOTE_TIMEOUT_S) -> Quote:
    """Fetch a delayed quote, falling back through last and previous close.

    Outside regular hours delayed data frequently has no bid or ask at all. The
    fallback is deliberate and *named* in the returned ``source`` rather than
    silently substituted, because it changes how the limit price should be
    read.
    """
    ib.reqMarketDataType(DELAYED_MARKET_DATA)
    ticker = ib.reqMktData(contract, "", False, False)

    try:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.25)
            if _has_spread(ticker) or _number(ticker, "last") or _number(ticker, "close"):
                break
    finally:
        ib.cancelMktData(contract)

    bid, ask = _number(ticker, "bid"), _number(ticker, "ask")
    last, close = _number(ticker, "last"), _number(ticker, "close")

    if bid and ask:
        source = "delayed bid/ask (15-minute lag)"
    elif last:
        source = "delayed last trade -- NO bid/ask available"
    elif close:
        source = "PREVIOUS CLOSE -- the market is not quoting this now"
    else:
        raise NoQuoteError(
            f"{contract.symbol}: IB returned no bid, ask, last or close. "
            "Nothing can be priced from this; try again when the market is "
            "quoting, and check the Gateway is logged in."
        )

    return Quote(symbol=contract.symbol, bid=bid, ask=ask, last=last,
                 close=close, source=source)


def _number(ticker: object, field: str) -> float | None:
    """A tick value, or None. IB uses NaN and -1 for "nothing here"."""
    import math

    value = getattr(ticker, field, None)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or number <= 0:
        return None
    return number


def _has_spread(ticker: object) -> bool:
    return bool(_number(ticker, "bid") and _number(ticker, "ask"))


def round_to_tick(price: float) -> float:
    return round(round(price / TICK) * TICK, 2)


def marketable_limit(quote: Quote, action: str, offset_bps: int) -> float:
    """A limit that crosses the spread, by ``offset_bps``.

    A BUY prices above the ask and a SELL below the bid, so the order is
    *marketable* -- it behaves like a market order for fill purposes while
    still capping the price at a number the human approved. That cap is the
    whole point: §9 forbids ``MarketOrder`` because with a 15-minute-delayed
    quote an uncapped instruction is priced off a number nobody can see.

    Computed from the far side of the spread when there is one and from the
    reference price otherwise, because crossing a spread you cannot see is not
    possible and pretending to would understate the real limit.
    """
    offset = offset_bps / 10_000.0
    side = action.upper()

    if side == "BUY":
        base = quote.ask or quote.reference
        return round_to_tick(base * (1 + offset))
    if side == "SELL":
        base = quote.bid or quote.reference
        return round_to_tick(base * (1 - offset))
    raise ValueError(f"cannot price a {action!r} order")


def stop_price_for(
    decision: FinalDecision, plan: OrderPlan, anchor_price: float
) -> float | None:
    """The protective stop for a BUY, or ``None``.

    **``anchor_price`` is the FRESH QUOTE's reference, not the limit and not
    the decide-time price.** There are three candidate prices here and only one
    is right; the other two were each wrong in a different way.

    *Not the decide-time ``plan.reference_price``.* That is what sizing used and
    it can be hours old -- on MRVL it was the prior close, 284.68, against a
    market near 269. Anchoring there puts the stop 5% away from where it was
    meant to be.

    *Not the limit either, which is what this used to do.* The limit is
    deliberately displaced from the market by ``limit_offset_bps``, so anchoring
    on it drags the stop along with the displacement. At 10 bps that was a 0.09%
    error and invisible. At the 100 bps this book runs, an 8% stop lands 7.1%
    below the fill -- and ``sizing``'s ``cap_risk`` sized the position on the
    arithmetic that 8% bounds the loss. The position does not become riskier
    (it loses slightly less when stopped); it gets shaken out on noise the
    thesis was sized to ride through. Widening the offset is exactly what made
    the old anchor wrong, which is why the two changes shipped together.

    So: the fresh quote's reference -- current, and undisplaced.
    """
    if plan.quantity <= 0 or not decision.stop_loss_pct:
        return None
    stop = anchor_price * (1 - decision.stop_loss_pct / 100.0)
    return round_to_tick(stop) if stop > 0 else None


def build_orders(
    plan: OrderPlan,
    limit_price: float,
    stop_price: float | None,
) -> tuple[Order, Order | None]:
    """The parent entry and its optional protective child.

    The parent is built with ``transmit=False`` whenever there is a child, so
    IB holds it until the child arrives. Transmitting the parent first would
    put an unprotected position on the book for however long the second call
    takes -- which is the window the whole bracket exists to close.
    """
    side = "BUY" if plan.quantity > 0 else "SELL"
    shares = abs(plan.quantity)

    parent = LimitOrder(side, shares, limit_price)
    parent.tif = "DAY"
    # Never outside regular hours: a marketable limit in a thin after-hours
    # book is how you cross a spread far wider than the one you approved.
    parent.outsideRth = False

    if stop_price is None:
        return parent, None

    parent.transmit = False
    child = StopOrder("SELL" if side == "BUY" else "BUY", shares, stop_price)
    child.tif = "GTC"
    child.outsideRth = False
    child.transmit = True
    return parent, child


async def preflight(ib: IB, contract: Stock, order: Order) -> Preflight:
    """``whatIfOrder``: margin impact and commission, placing nothing.

    §9 requires this *before* the human sees the prompt. A failure here is
    reported rather than raised -- the preflight is information for the human,
    and losing it is not a reason to block an order that compliance already
    cleared.

    ``whatIfOrder`` requires ``transmit=True`` to be evaluated, so a held
    parent is copied rather than mutated: flipping the real order's flag here
    and forgetting to flip it back would transmit an unprotected entry.
    """
    import copy

    probe = copy.copy(order)
    probe.transmit = True
    probe.whatIf = True

    try:
        state = await ib.whatIfOrderAsync(contract, probe)
    except Exception as exc:  # noqa: BLE001 -- information, not a gate
        logger.warning("whatIfOrder failed for %s: %s", contract.symbol, exc)
        return Preflight(warning=f"preflight unavailable: {type(exc).__name__}")

    return Preflight.from_order_state(state)


async def wait_for_terminal(
    trade: Trade, *, timeout_s: float = FILL_TIMEOUT_S, poll_s: float = 0.25,
) -> bool:
    """Wait until ``trade`` stops moving. True if it did, False on timeout.

    **Pass the PARENT, never the protective stop.** The child is a GTC
    ``StopOrder`` whose whole job is to sit there working until the thesis
    breaks; waiting for it to reach a terminal status would hang every batch
    for ``timeout_s`` and then report a false timeout.

    **A timeout is not an error, and this does not raise.** The order may well
    fill a second later. What the caller cannot do is *size the next order*
    against a book it cannot describe, so the batch stops and says why. The
    distinction matters: raising here would make a still-working order look
    like a failure, and the operator would go looking for a problem that is
    not there.

    ``ib_async`` updates ``trade.orderStatus`` in place from the event loop, so
    polling it is reading live state rather than a snapshot.
    """
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        status = trade.orderStatus.status
        if status in TERMINAL_STATUSES:
            return True
        if asyncio.get_running_loop().time() >= deadline:
            logger.warning(
                "order %s is still %s after %.0fs (filled %s of %s)",
                trade.order.orderId, status, timeout_s,
                trade.orderStatus.filled, trade.order.totalQuantity,
            )
            return False
        await asyncio.sleep(poll_s)


def filled_quantity(trade: Trade) -> tuple[int, float]:
    """Signed shares actually filled, and the average price paid.

    **Signed from the order's own side, not from the plan**, so a caller cannot
    advance a book in the wrong direction by passing the wrong sign. And it is
    the *filled* quantity: a partial fill moves the book by what filled, and
    treating it as the full order would make every later check in a batch wrong
    in the one direction that matters.
    """
    status = trade.orderStatus
    shares = int(status.filled or 0)
    if shares == 0:
        return 0, 0.0
    side = -1 if trade.order.action.upper() == "SELL" else 1
    price = float(status.avgFillPrice or 0.0)
    return side * shares, price


@dataclass(frozen=True)
class Session:
    """Whether the exchange is in its regular session, and what it says.

    Built from IB's own ``ContractDetails.liquidHours`` rather than a hardcoded
    09:30-16:00, because that string already accounts for holidays and
    half-days and this repo has no trading calendar. Checked: there is no
    market-hours knowledge anywhere in the codebase -- ``EXCHANGE_TZ`` is a bare
    timezone and ``useRTH=True`` on the bars request only asks IB to filter what
    it returns.

    ``is_open=None`` means IB did not say. Unknown is not closed, and this is
    only ever a warning, so an absent answer stays silent rather than crying
    wolf.
    """

    is_open: bool | None
    now: datetime | None = None
    opens_at: datetime | None = None
    closes_at: datetime | None = None
    note: str = ""

    @property
    def warning(self) -> str | None:
        """One line for the confirmation screen, or ``None`` when all is well."""
        if self.is_open is not False:
            return None
        when = f" (now {self.now:%H:%M %Z})" if self.now else ""
        nxt = (
            f" Next session opens {self.opens_at:%a %d %b %H:%M %Z}."
            if self.opens_at else ""
        )
        return (
            f"THE REGULAR SESSION IS CLOSED{when}. This order is "
            f"outsideRth=False, so it will not fill until the market reopens."
            + nxt
        )


async def market_session(ib: IB, contract: Stock) -> Session:
    """Is the exchange in its regular session right now?

    One IB call, and the caller is expected to reuse the answer: every US
    equity shares one session, so asking per order in a batch is the same
    question fifteen times.

    **Never raises.** A session check that can break order placement would be a
    worse defect than the one it reports, so anything unexpected -- no contract
    details, an empty hours string, a timezone IB spells in a way
    ``zoneinfo`` does not know -- comes back as ``is_open=None`` and says
    nothing at the gate.
    """
    try:
        details = await ib.reqContractDetailsAsync(contract)
    except Exception as exc:  # noqa: BLE001 -- a warning must not break an order
        logger.debug("could not read contract details for the session: %s", exc)
        return Session(is_open=None, note=f"contract details unavailable: {exc}")

    if not details:
        return Session(is_open=None, note="IB returned no contract details")

    detail = details[0]
    try:
        sessions = detail.liquidSessions()
    except Exception as exc:  # noqa: BLE001 -- ib_async raises on odd tz strings
        logger.debug("could not parse liquidHours: %s", exc)
        return Session(is_open=None, note=f"unparseable liquidHours: {exc}")

    if not sessions:
        return Session(is_open=None, note="IB reported no liquid sessions")

    now = datetime.now(EXCHANGE_TZ)
    for window in sessions:
        if window.start <= now <= window.end:
            return Session(is_open=True, now=now,
                           opens_at=window.start, closes_at=window.end)

    upcoming = [w.start for w in sessions if w.start > now]
    return Session(
        is_open=False, now=now,
        opens_at=min(upcoming) if upcoming else None,
    )


async def wait_for_cancel(
    trade: Trade, *, timeout_s: float = CANCEL_TIMEOUT_S, poll_s: float = 0.25,
) -> bool:
    """Wait for IB to acknowledge a cancel. True if it did.

    **A separate function from ``wait_for_terminal``, not a parameter on it**,
    for two independent reasons:

    * ``PendingCancel`` is an acknowledged cancel but is deliberately not a
      terminal status -- see ``CANCEL_ACKNOWLEDGED``. Reusing the fill wait
      would report a successful cancel as a timeout.
    * ``tests/execution/test_batch.py`` asserts ``_execute_one`` contains
      exactly one ``wait_for_terminal`` call, on ``trades[0]``. That assertion
      is right -- the GTC stop must never be waited on -- so the cancel path
      needs its own name rather than a second call it would have to excuse.
    """
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        if trade.orderStatus.status in CANCEL_ACKNOWLEDGED:
            return True
        if asyncio.get_running_loop().time() >= deadline:
            logger.warning(
                "order %s is still %s %.0fs after a cancel request",
                trade.order.orderId, trade.orderStatus.status, timeout_s,
            )
            return False
        await asyncio.sleep(poll_s)


async def cancel(
    ib: IB,
    trades: list[Trade],
    *,
    journal: Journal,
    run_id: str,
    symbol: str,
) -> bool:
    """Cancel a bracket that never filled. **Child first, then the parent.**

    The order matters. Cancelling the parent first leaves the protective stop
    alone in the account for however long the second call takes -- a resting
    SELL STOP against a position that was never opened. Child first means the
    account passes through "no orders" rather than "a naked stop".

    **Only for an unfilled bracket.** The caller checks that; this does not
    second-guess it, but the reason is worth stating where the cancel lives: the
    stop is sized for the full quantity, so cancelling the parent remainder of a
    PARTIAL fill leaves a stop that would oversell. Fixing that is
    cancel-and-replace, which ``FOLLOWUPS.md`` records as real work rather than
    something to half-build inside a cancel helper.

    Journals ``order_cancelled`` -- a new event name on purpose. The
    ``order_submitted`` events carry exactly two roles, ``entry`` and
    ``protective_stop``, and a test asserts that set; a cancel is a different
    thing that happened, not a third submission.
    """
    acknowledged = True
    for trade in reversed(trades):  # child first
        try:
            ib.cancelOrder(trade.order)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not cancel order %s: %s",
                           trade.order.orderId, exc)
            acknowledged = False
            continue
        settled = await wait_for_cancel(trade)
        acknowledged = acknowledged and settled
        journal.write(
            "order_cancelled", run_id=run_id, symbol=symbol,
            order_id=trade.order.orderId, order_type=trade.order.orderType,
            status=trade.orderStatus.status, acknowledged=settled,
        )
        logger.info(
            "cancelled %s order %s: %s",
            symbol, trade.order.orderId, trade.orderStatus.status,
        )
    return acknowledged


async def place(
    ib: IB,
    contract: Stock,
    parent: Order,
    child: Order | None,
    *,
    journal: Journal,
    run_id: str,
    settle_s: float = 2.0,
) -> list[Trade]:
    """Place the order. **Safety gate, fire 2 of 2 -- immediately before this.**

    The second ``assert_paper_account`` is the entire reason this function does
    not simply take a ``Trade``: the check has to happen here, microseconds
    before the call, rather than anywhere a caller might forget it.
    """
    accounts = assert_paper_account(ib)
    logger.info(
        "paper account re-verified immediately before placeOrder: %s",
        ", ".join(accounts),
    )

    trades: list[Trade] = []

    parent_trade = ib.placeOrder(contract, parent)
    trades.append(parent_trade)
    journal.write(
        "order_submitted", run_id=run_id, symbol=contract.symbol,
        role="entry", action=parent.action, quantity=parent.totalQuantity,
        order_type=parent.orderType, limit_price=parent.lmtPrice,
        order_id=parent_trade.order.orderId, accounts=accounts,
    )

    if child is not None:
        # parentId is only known after the parent is placed.
        child.parentId = parent_trade.order.orderId
        child_trade = ib.placeOrder(contract, child)
        trades.append(child_trade)
        journal.write(
            "order_submitted", run_id=run_id, symbol=contract.symbol,
            role="protective_stop", action=child.action,
            quantity=child.totalQuantity, order_type=child.orderType,
            stop_price=child.auxPrice, order_id=child_trade.order.orderId,
            parent_id=child.parentId,
        )

    # Let the first status updates arrive so the journal records what IB said
    # rather than only what we sent.
    await asyncio.sleep(settle_s)
    for trade in trades:
        journal.write(
            "order_status", run_id=run_id, symbol=contract.symbol,
            order_id=trade.order.orderId, status=trade.orderStatus.status,
            filled=trade.orderStatus.filled,
            remaining=trade.orderStatus.remaining,
            avg_fill_price=trade.orderStatus.avgFillPrice,
        )
        logger.info(
            "%s order %s: %s (filled %s of %s)",
            contract.symbol, trade.order.orderId, trade.orderStatus.status,
            trade.orderStatus.filled, trade.order.totalQuantity,
        )

    return trades


__all__ = [
    "CANCEL_ACKNOWLEDGED",
    "CANCEL_TIMEOUT_S",
    "DELAYED_MARKET_DATA",
    "FILL_TIMEOUT_S",
    "TERMINAL_STATUSES",
    "NoQuoteError",
    "Quote",
    "Session",
    "TICK",
    "build_orders",
    "cancel",
    "connect",
    "contract_for",
    "filled_quantity",
    "market_session",
    "marketable_limit",
    "place",
    "preflight",
    "quote",
    "round_to_tick",
    "stop_price_for",
    "wait_for_cancel",
    "wait_for_terminal",
]
