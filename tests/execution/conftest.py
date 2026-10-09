"""A fake IB, so every execution test runs offline.

The alternative -- mocking ``ib_async`` call by call in each test -- makes the
tests pass while telling you nothing about ORDER of operations, and the order is
the whole safety property here: ``assert_paper_account`` has to fire once after
connect and again immediately before ``placeOrder``.

So ``FakeIB`` records a **call log** rather than just return values, and the
tests assert against the sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeTicker:
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    close: float | None = None


@dataclass
class FakeContract:
    symbol: str = "TSM"
    secType: str = "STK"
    localSymbol: str = ""


@dataclass
class FakeOrderStatus:
    status: str = "PreSubmitted"
    filled: float = 0.0
    remaining: float = 0.0
    avgFillPrice: float = 0.0


@dataclass
class FakeTrade:
    contract: Any
    order: Any
    orderStatus: FakeOrderStatus = field(default_factory=FakeOrderStatus)


@dataclass
class FakeSession:
    """One window from ``ContractDetails.liquidSessions()``."""

    start: Any
    end: Any


@dataclass
class FakeContractDetails:
    """What ``reqContractDetailsAsync`` returns, for the session check.

    ``liquidSessions()`` is a METHOD on the real ``ContractDetails`` (ib_async
    parses ``liquidHours`` + ``timeZoneId`` into tz-aware windows), so the fake
    implements the method rather than the raw string. Tests that want the
    parsing itself exercised should go at ib_async, not at this.
    """

    sessions: list[FakeSession] = field(default_factory=list)
    timeZoneId: str = "US/Eastern"
    liquidHours: str = ""
    raises: bool = False

    def liquidSessions(self) -> list[FakeSession]:
        if self.raises:
            # The real one raises when timeZoneId is something zoneinfo does
            # not know, which is a shape `market_session` must survive.
            raise ValueError("unknown timezone")
        return list(self.sessions)


@dataclass
class FakePosition:
    contract: FakeContract
    position: float
    avgCost: float


@dataclass
class FakeAccountValue:
    tag: str
    value: str
    currency: str = "USD"


@dataclass
class FakeOrderState:
    """Mirrors ib_async's OrderState field names, including the unset sentinel."""

    status: str = "PreSubmitted"
    initMarginChange: str = "2832.90"
    maintMarginChange: str = "1416.45"
    equityWithLoanChange: str = "-1.00"
    commission: float = 1.0
    commissionCurrency: str = "USD"
    warningText: str = ""


class FakeIB:
    """Records every call, in order, so tests can assert the sequence."""

    def __init__(
        self,
        *,
        accounts: list[str] | None = None,
        ticker: FakeTicker | None = None,
        positions: list[FakePosition] | None = None,
        account_values: list[FakeAccountValue] | None = None,
        order_state: FakeOrderState | None = None,
        next_order_id: int = 101,
        fills: bool = False,
        fill_price: float | None = None,
        contract_details: list | None = None,
    ) -> None:
        self._accounts = accounts if accounts is not None else ["DU1234567"]
        self._ticker = ticker or FakeTicker(bid=472.0, ask=472.3, last=472.15,
                                            close=471.0)
        self._positions = positions or []
        self._account_values = account_values or [
            FakeAccountValue("NetLiquidation", "250000.00"),
            FakeAccountValue("TotalCashValue", "90279.72"),
            FakeAccountValue("BuyingPower", "180000.00"),
        ]
        self._order_state = order_state or FakeOrderState()
        self._next_order_id = next_order_id
        #: Fill entry orders immediately on placeOrder, leaving STOP orders
        #: working. Off by default so every pre-batch test sees the original
        #: behaviour (an order that is placed and stays PreSubmitted).
        #:
        #: STOP orders are deliberately NOT filled: a protective stop is
        #: supposed to sit there, and `wait_for_terminal` waiting on one would
        #: hang a batch for the full timeout. Tests need that asymmetry to be
        #: real rather than assumed.
        self._fills = fills
        self._fill_price = fill_price
        #: None means "the method is unavailable", which is how `market_session`
        #: behaves against a Gateway that will not answer -- distinct from an
        #: empty list, which means "answered, and knows of no sessions".
        self._contract_details = contract_details

        #: Every call, in order. The safety tests read this.
        self.calls: list[str] = []
        self.placed: list[tuple[Any, Any]] = []
        #: orderId -> FakeTrade, so a cancel can look up what it is cancelling.
        self.trades: dict[int, FakeTrade] = {}
        self.cancelled: list[int] = []
        self.disconnected = False
        self.market_data_type: int | None = None

    # --- the surface broker.py and reconcile.py actually use ----------------

    def managedAccounts(self) -> list[str]:
        self.calls.append("managedAccounts")
        return list(self._accounts)

    async def connectAsync(self, host, port, clientId):  # noqa: ANN001, N803
        self.calls.append(f"connect:{host}:{port}:{clientId}")

    def reqMarketDataType(self, kind: int) -> None:
        self.calls.append(f"reqMarketDataType:{kind}")
        self.market_data_type = kind

    def reqMktData(self, contract, *args, **kwargs):  # noqa: ANN001
        self.calls.append("reqMktData")
        return self._ticker

    def cancelMktData(self, contract) -> None:  # noqa: ANN001
        self.calls.append("cancelMktData")

    async def qualifyContractsAsync(self, *contracts):  # noqa: ANN002
        self.calls.append("qualifyContracts")
        return list(contracts)

    async def whatIfOrderAsync(self, contract, order):  # noqa: ANN001
        self.calls.append("whatIfOrder")
        return self._order_state

    def placeOrder(self, contract, order):  # noqa: ANN001
        self.calls.append("placeOrder")
        if not order.orderId:
            order.orderId = self._next_order_id
            self._next_order_id += 1
        trade = FakeTrade(contract=contract, order=order)
        # Kept, keyed by orderId, so cancelOrder can find the trade it has to
        # mutate. Before the cancel work this object was built and dropped.
        self.trades[order.orderId] = trade
        if self._fills and getattr(order, "orderType", "") != "STP":
            quantity = float(order.totalQuantity)
            trade.orderStatus = FakeOrderStatus(
                status="Filled",
                filled=quantity,
                remaining=0.0,
                avgFillPrice=(
                    self._fill_price
                    if self._fill_price is not None
                    else float(getattr(order, "lmtPrice", 0.0) or 0.0)
                ),
            )
        self.placed.append((contract, order))
        return trade

    def cancelOrder(self, order, manualCancelOrderTime: str = ""):  # noqa: ANN001, N803
        """Mirror ib_async's STATUS semantics, which are not what you'd guess.

        A working order goes to **PendingCancel**, not ``Cancelled`` -- it is a
        request, and IB confirms it separately. A parent still being HELD
        (``PendingSubmit`` with ``transmit=False``, which is how a bracket's
        entry sits until its child arrives) short-circuits straight to
        ``Cancelled`` because it was never live.

        Getting this wrong in the fake would hide the real bug it exists to
        catch: ``wait_for_terminal`` does not accept ``PendingCancel``, so code
        that reused it after a cancel would report a successful cancel as a
        timeout.
        """
        self.calls.append("cancelOrder")
        self.cancelled.append(order.orderId)
        trade = self.trades.get(order.orderId)
        if trade is None:
            return None
        held = (
            trade.orderStatus.status == "PendingSubmit"
            and not getattr(order, "transmit", True)
        )
        trade.orderStatus.status = "Cancelled" if held else "PendingCancel"
        return trade

    async def reqContractDetailsAsync(self, contract):  # noqa: ANN001
        self.calls.append("reqContractDetails")
        if self._contract_details is None:
            raise RuntimeError("contract details unavailable")
        return list(self._contract_details)

    def positions(self) -> list[FakePosition]:
        self.calls.append("positions")
        return list(self._positions)

    async def accountSummaryAsync(self) -> list[FakeAccountValue]:
        self.calls.append("accountSummary")
        return list(self._account_values)

    def disconnect(self) -> None:
        self.calls.append("disconnect")
        self.disconnected = True

    # --- helpers for assertions --------------------------------------------

    def count(self, name: str) -> int:
        return sum(1 for c in self.calls if c == name or c.startswith(name + ":"))

    def index_of(self, name: str, occurrence: int = 1) -> int:
        seen = 0
        for i, call in enumerate(self.calls):
            if call == name or call.startswith(name + ":"):
                seen += 1
                if seen == occurrence:
                    return i
        raise AssertionError(f"{name!r} occurrence {occurrence} not in {self.calls}")
