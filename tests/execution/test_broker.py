"""``broker.py`` -- the order path, and §9's three rules.

**The first test in this file is the Stage 7 exit gate.** The migration plan
words it as *"one paper trade, human-confirmed, with `assert_paper_account`
having fired twice -- once after connect, once immediately before
`placeOrder`"*, and the second firing is the one that matters: it catches a
reconnect that landed on a different account between the check and the order.

Asserted on the *sequence* of calls rather than a count, because two calls in
the wrong order is the same as one.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from research_desk.execution import broker
from research_desk.execution.journal import Journal
from research_desk.execution.safety import LiveAccountError, ReplayTradeError
from research_desk.models.modes import RunMode
from research_desk.models.orders import OrderPlan
from research_desk.models.state import FinalDecision

from .conftest import FakeContract, FakeIB, FakeTicker

AS_OF = date(2026, 10, 7)


def decision(**overrides) -> FinalDecision:
    body = {
        "action": "BUY", "symbol": "TSM", "conviction": 0.65,
        "target_weight_pct": 4.5, "horizon_days": 45, "stop_loss_pct": 8.0,
        "rationale": "relative strength is improving against the sector",
        "dissent": "a liquidity shock would hit this name harder than the index",
        "invalidation": "a close below the 200-day moving average at 389",
        "expires_at": datetime.now() + timedelta(hours=12),
    }
    body.update(overrides)
    return FinalDecision(**body)


def plan(quantity: int = 12, action: str = "BUY") -> OrderPlan:
    return OrderPlan(
        symbol="TSM", as_of=AS_OF, action=action, quantity=quantity,
        reference_price=472.15, estimated_notional=abs(quantity) * 472.15,
        target_weight_pct=4.34, current_weight_pct=2.08, binding_cap="requested",
    )


# --------------------------------------------------------------------------- #
# THE EXIT GATE: assert_paper_account fires twice, in the right places
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_ib(monkeypatch):
    """Make ``broker.connect`` build our fake instead of a real ``IB``."""
    ib = FakeIB()
    monkeypatch.setattr(broker, "IB", lambda: ib)
    return ib


async def test_the_paper_account_gate_fires_twice_in_the_right_order(
    fake_ib, tmp_path
) -> None:
    """**The Stage 7 exit gate**, asserted across the whole flow on one account.

    Once after connect, once immediately before ``placeOrder``. The second is
    not redundant: between the first check and the order, a reconnect can land
    on a different account, and the port number is not a safety guarantee -- a
    socat mapping or an env-var mistake can point a nominally-paper setup at a
    live one. ``managedAccounts()`` is the only thing that actually knows.

    Asserted on the SEQUENCE, because two calls in the wrong order is the same
    as one.
    """
    ib, _ = await broker.connect(host="h", port=4004, client_id=11,
                                 mode=RunMode.LIVE)
    assert ib is fake_ib
    assert fake_ib.count("managedAccounts") == 1, "fire 1 of 2, after connect"

    parent, child = broker.build_orders(plan(), 472.62, 434.81)
    await broker.place(
        fake_ib, FakeContract(), parent, child,
        journal=Journal(tmp_path), run_id="t", settle_s=0,
    )

    assert fake_ib.count("managedAccounts") == 2, (
        f"expected exactly two account checks, got {fake_ib.calls}"
    )

    # Fire 1 after connect; fire 2 before the first placeOrder.
    first = fake_ib.index_of("managedAccounts", 1)
    second = fake_ib.index_of("managedAccounts", 2)
    connected = fake_ib.index_of("connect")
    placed = fake_ib.index_of("placeOrder", 1)

    assert connected < first < second < placed, (
        "the order must be connect -> check -> check -> placeOrder, got "
        f"{fake_ib.calls}"
    )
    # Nothing between the second check and the order.
    assert second == placed - 1, (
        "something ran between the final account check and placeOrder: "
        f"{fake_ib.calls[second + 1:placed]}"
    )


async def test_connect_verifies_the_account_before_returning(monkeypatch) -> None:
    """Fire 1 of 2. A connection to a non-paper account must not be handed back."""
    ib = FakeIB(accounts=["U1234567"])  # a live-looking account
    monkeypatch.setattr(broker, "IB", lambda: ib)

    with pytest.raises(LiveAccountError, match="Non-paper account"):
        await broker.connect(host="h", port=4004, client_id=11, mode=RunMode.LIVE)

    # And it must not leave a socket open to an account it just refused.
    assert ib.disconnected


async def test_connect_refuses_in_replay_mode() -> None:
    """Both replay modes are driven by historical prices, so an order from one
    would be meaningless. Checked before a socket is even opened."""
    for mode in (RunMode.REPLAY, RunMode.REPLAY_LLM):
        with pytest.raises(ReplayTradeError):
            await broker.connect(host="h", port=4004, client_id=11, mode=mode)


async def test_place_refuses_when_the_account_turns_live_mid_session(tmp_path) -> None:
    """The exact scenario the second gate exists for.

    The account looked fine at connect and does not at order time. Nothing is
    placed.
    """
    ib = FakeIB(accounts=["U9999999"])
    parent, _ = broker.build_orders(plan(), 472.62, None)
    with pytest.raises(LiveAccountError):
        await broker.place(ib, FakeContract(), parent, None,
                           journal=Journal(tmp_path), run_id="t", settle_s=0)
    assert ib.count("placeOrder") == 0


# --------------------------------------------------------------------------- #
# Marketable limit, never market
# --------------------------------------------------------------------------- #


def test_a_buy_crosses_above_the_ask() -> None:
    q = broker.Quote("TSM", bid=472.0, ask=472.3, last=472.15, close=471.0, source="s")
    assert broker.marketable_limit(q, "BUY", 10) == pytest.approx(472.77, abs=0.005)


def test_a_sell_crosses_below_the_bid() -> None:
    q = broker.Quote("TSM", bid=472.0, ask=472.3, last=472.15, close=471.0, source="s")
    assert broker.marketable_limit(q, "SELL", 10) == pytest.approx(471.53, abs=0.005)


def test_the_limit_is_rounded_to_a_cent() -> None:
    """US equities trade in cents; more precision earns a rejection for
    violating the minimum tick, which arrives as a generic error."""
    q = broker.Quote("X", bid=None, ask=333.333, last=None, close=None, source="s")
    price = broker.marketable_limit(q, "BUY", 7)
    assert price == round(price, 2)


def test_pricing_falls_back_when_there_is_no_spread() -> None:
    """Outside regular hours delayed data frequently has no bid or ask."""
    q = broker.Quote("X", bid=None, ask=None, last=None, close=100.0,
                     source="previous close")
    assert broker.marketable_limit(q, "BUY", 10) == pytest.approx(100.10, abs=0.005)


def test_a_quote_with_nothing_usable_raises_rather_than_guessing() -> None:
    """A limit priced off a guessed number is worse than no order."""
    q = broker.Quote("X", bid=None, ask=None, last=None, close=None, source="none")
    with pytest.raises(broker.NoQuoteError):
        _ = q.reference


def test_no_market_order_is_even_imported() -> None:
    """§9: *"Marketable LimitOrder, never MarketOrder."*

    With delayed data you are looking at a 15-minute-old price, so an uncapped
    instruction is priced off a number nobody can see. An AST check, in the
    shape ``tests/test_layering.py`` uses -- a comment mentioning MarketOrder
    must not fail the build, and ``from ib_async import MarketOrder as M``
    must not sneak past.
    """
    import ast
    from pathlib import Path

    src = Path(broker.__file__)
    tree = ast.parse(src.read_text(), filename=str(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = {a.name for a in node.names}
            assert "MarketOrder" not in names, (
                "broker.py imports MarketOrder. §9 forbids it: with a "
                "15-minute-delayed quote a market order is an open-ended "
                "instruction priced off a number you cannot see."
            )


def test_the_order_is_always_a_limit() -> None:
    parent, _ = broker.build_orders(plan(), 472.62, None)
    assert parent.orderType == "LMT"
    assert parent.lmtPrice == pytest.approx(472.62)


def test_orders_never_go_outside_regular_hours() -> None:
    """A marketable limit in a thin after-hours book crosses a spread far
    wider than the one that was approved."""
    parent, child = broker.build_orders(plan(), 472.62, 434.81)
    assert parent.outsideRth is False
    assert child.outsideRth is False


# --------------------------------------------------------------------------- #
# The protective stop
# --------------------------------------------------------------------------- #


def test_a_buy_gets_a_stop_at_the_decisions_distance() -> None:
    """Sizing's ``cap_risk`` assumed this stop existed. Now it does."""
    stop = broker.stop_price_for(decision(), plan(), 472.62)
    assert stop == pytest.approx(472.62 * 0.92, abs=0.01)


def test_the_stop_is_measured_from_the_limit_not_the_stale_reference() -> None:
    """So the distance the human sees is the distance they get."""
    near = broker.stop_price_for(decision(), plan(), 500.0)
    far = broker.stop_price_for(decision(), plan(), 400.0)
    assert near > far


def test_the_parent_is_held_until_the_child_exists() -> None:
    """**Transmitting the parent first would put an unprotected position on the
    book** for however long the second call takes -- which is the window the
    bracket exists to close."""
    parent, child = broker.build_orders(plan(), 472.62, 434.81)
    assert parent.transmit is False
    assert child.transmit is True
    assert child.orderType == "STP"
    assert child.action == "SELL"
    assert child.totalQuantity == parent.totalQuantity


async def test_the_child_gets_its_parent_id_only_after_the_parent_is_placed(
    tmp_path,
) -> None:
    """``parentId`` is not knowable before IB assigns the parent an order id."""
    ib = FakeIB()
    parent, child = broker.build_orders(plan(), 472.62, 434.81)
    assert child.parentId == 0

    trades = await broker.place(
        ib, FakeContract(), parent, child,
        journal=Journal(tmp_path), run_id="t", settle_s=0,
    )
    assert len(trades) == 2
    assert child.parentId == trades[0].order.orderId
    assert child.parentId != 0


def test_a_sell_gets_no_stop() -> None:
    """It is already reducing exposure, and a stop on an exit is a second exit."""
    assert broker.stop_price_for(decision(action="SELL"), plan(-25, "SELL"), 529.0) is None
    parent, child = broker.build_orders(plan(-25, "SELL"), 529.0, None)
    assert child is None
    assert parent.transmit is True, "a lone order must transmit"
    assert parent.action == "SELL"


def test_no_stop_when_the_decision_carries_no_stop_pct() -> None:
    assert broker.stop_price_for(decision(stop_loss_pct=None), plan(), 472.62) is None


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


async def test_preflight_does_not_mutate_the_real_order() -> None:
    """``whatIfOrder`` needs ``transmit=True`` to be evaluated.

    Flipping the real parent's flag and forgetting to flip it back would
    transmit an unprotected entry, so the probe is a copy.
    """
    ib = FakeIB()
    parent, _ = broker.build_orders(plan(), 472.62, 434.81)
    assert parent.transmit is False

    report = await broker.preflight(ib, FakeContract(), parent)

    assert parent.transmit is False, "preflight mutated the order it was given"
    assert parent.whatIf is False
    assert report.init_margin_change == "2832.90"


async def test_a_failed_preflight_is_reported_not_raised() -> None:
    """The preflight is information for the human. Losing it is not a reason to
    block an order compliance already cleared."""
    class Broken(FakeIB):
        async def whatIfOrderAsync(self, contract, order):
            raise RuntimeError("IB said no")

    report = await broker.preflight(Broken(), FakeContract(), broker.build_orders(plan(), 1.0, None)[0])
    assert report.warning and "preflight unavailable" in report.warning


async def test_the_unset_commission_sentinel_is_shown_as_none() -> None:
    """IB returns 1.7976931348623157e+308 for "unset", and printing that in a
    confirmation prompt would be worse than printing nothing."""
    from .conftest import FakeOrderState

    ib = FakeIB(order_state=FakeOrderState(commission=1.7976931348623157e308))
    report = await broker.preflight(ib, FakeContract(), broker.build_orders(plan(), 1.0, None)[0])
    assert report.commission is None


# --------------------------------------------------------------------------- #
# Quotes
# --------------------------------------------------------------------------- #


async def test_the_quote_names_its_own_source() -> None:
    """The fallback chain is not an implementation detail: a limit priced off
    the previous close behaves very differently from one off a live spread."""
    spread = await broker.quote(FakeIB(ticker=FakeTicker(bid=1.0, ask=1.1)), FakeContract())
    assert "bid/ask" in spread.source

    last_only = await broker.quote(
        FakeIB(ticker=FakeTicker(last=5.0)), FakeContract()
    )
    assert "last" in last_only.source and "NO bid/ask" in last_only.source

    close_only = await broker.quote(
        FakeIB(ticker=FakeTicker(close=7.0)), FakeContract()
    )
    assert "CLOSE" in close_only.source.upper()


async def test_the_quote_requests_delayed_data() -> None:
    """Delayed data needs no subscription; 10091 is a substitution notice that
    ``logging_setup`` already demotes."""
    ib = FakeIB()
    await broker.quote(ib, FakeContract())
    assert ib.market_data_type == broker.DELAYED_MARKET_DATA


async def test_the_quote_subscription_is_always_cancelled() -> None:
    ib = FakeIB()
    await broker.quote(ib, FakeContract())
    assert ib.count("cancelMktData") == 1


async def test_no_usable_tick_raises() -> None:
    ib = FakeIB(ticker=FakeTicker())
    with pytest.raises(broker.NoQuoteError, match="no bid, ask, last or close"):
        await broker.quote(ib, FakeContract(), timeout_s=0.3)


def test_nan_and_negative_ticks_are_treated_as_absent() -> None:
    """IB uses NaN and -1 for "nothing here", and -1 as a price would invert
    every comparison downstream."""
    assert broker._number(FakeTicker(bid=float("nan")), "bid") is None
    assert broker._number(FakeTicker(bid=-1.0), "bid") is None
    assert broker._number(FakeTicker(bid=0.0), "bid") is None
    assert broker._number(FakeTicker(bid=1.5), "bid") == 1.5


# --------------------------------------------------------------------------- #
# Contracts and journaling
# --------------------------------------------------------------------------- #


def test_the_contract_uses_ibs_spelling() -> None:
    """One definition of "IB calls BRK.B 'BRK B'", shared with bars.py. Two
    copies of that mapping is how one of them goes stale."""
    assert broker.contract_for("BRK.B", {"ib": "BRK B"}).symbol == "BRK B"
    plain = broker.contract_for("TSM")
    assert plain.symbol == "TSM"
    assert plain.exchange == "SMART"
    assert plain.currency == "USD"


async def test_both_legs_are_journaled_with_their_roles(tmp_path) -> None:
    import json

    ib = FakeIB()
    parent, child = broker.build_orders(plan(), 472.62, 434.81)
    journal = Journal(tmp_path)
    await broker.place(ib, FakeContract(), parent, child,
                       journal=journal, run_id="run-1", settle_s=0)

    events = [json.loads(line) for line in journal.path.read_text().splitlines()]
    submitted = [e for e in events if e["event"] == "order_submitted"]
    assert {e["role"] for e in submitted} == {"entry", "protective_stop"}
    assert all(e["run_id"] == "run-1" for e in submitted)
    # And the status that came back, not only what we sent.
    assert any(e["event"] == "order_status" for e in events)
