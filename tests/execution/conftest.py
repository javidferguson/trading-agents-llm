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

        #: Every call, in order. The safety tests read this.
        self.calls: list[str] = []
        self.placed: list[tuple[Any, Any]] = []
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
        self.placed.append((contract, order))
        return trade

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
