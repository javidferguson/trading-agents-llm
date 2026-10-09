"""``execute --all``. **A batch is not the single-order path in a shell loop.**

Three properties separate them, and each one is a defect in the loop version:

* **The book advances.** Five of ``rules.check``'s rules read
  ``post_trade(portfolio, ...)``, so handing every order the same pre-batch
  snapshot walks straight past ``max_positions``, ``min_cash_pct``,
  ``max_gross_exposure_pct``, ``max_sector_pct`` and ``max_position_pct``.
* **The aggregate is shown before anything is approved.** Every order was sized
  and vetoed in isolation; a set of individually-compliant orders can still
  move the book somewhere nobody chose.
* **An ambiguous fill stops the batch** rather than being carried forward as a
  guess.

And one property it must NOT have: a weaker gate. Every order still goes
through ``CLIConfirmationGate`` with its ticker typed.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from research_desk.execution import cli
from research_desk.execution.broker import filled_quantity, wait_for_terminal
from research_desk.execution.confirmation import BatchProjection, BatchRow
from research_desk.models.orders import OrderPlan

from .conftest import FakeOrderStatus, FakeTrade

AS_OF = date(2026, 10, 8)


# --------------------------------------------------------------------------- #
# Writing proposals on disk, the way `persist` does
# --------------------------------------------------------------------------- #


def write_proposal(
    directory: Path,
    symbol: str,
    *,
    run_id: str | None = None,
    action: str = "BUY",
    quantity: int = 10,
    price: float = 100.0,
    hours_to_expiry: float = 12.0,
    blocked: bool = False,
    absent: list[str] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    run = run_id or f"2026100 8-{symbol}".replace(" ", "")
    decision = {
        "schema_version": 1,
        "action": action,
        "symbol": symbol,
        "conviction": 0.6,
        "target_weight_pct": 0.0 if action == "HOLD" else 3.0,
        "order_type": "LMT",
        "limit_offset_bps": 10,
        "stop_loss_pct": 8.0,
        "horizon_days": 45,
        "rationale": "relative strength is improving against the sector",
        "dissent": "a liquidity shock would hit this name harder than the index",
        "invalidation": "a close below the 200-day moving average",
        "expires_at": (
            datetime.now() + timedelta(hours=hours_to_expiry)
        ).isoformat(),
        "degraded": False,
        "absent_analysts": absent or [],
    }
    plan = OrderPlan(
        symbol=symbol, as_of=AS_OF,
        action="HOLD" if blocked or quantity == 0 else action,
        quantity=0 if blocked else quantity,
        reference_price=price,
        estimated_notional=0.0 if blocked else abs(quantity) * price,
        violations=(
            [{"rule": "max_positions", "severity": "block",
              "message": "the book is full", "limit": 15.0, "actual": 16.0}]
            if blocked else []
        ),
        **({"skip_reason": "blocked by max_positions"} if blocked else {}),
    )
    path = directory / f"{symbol}_{run}.json"
    path.write_text(json.dumps({
        "run_id": run, "as_of": AS_OF.isoformat(), "mode": "live",
        "config_hash": "h", "prompt_pack_version": "1",
        "decision": decision,
        "order_plan": json.loads(plan.model_dump_json()),
    }, indent=2))
    return path


def touch_receipt(proposal: Path) -> None:
    cli.receipt_path(proposal).write_text(
        json.dumps({"at": datetime.now().isoformat(), "outcome": "placed"})
    )


# --------------------------------------------------------------------------- #
# Selection: what a batch may act on
# --------------------------------------------------------------------------- #


def test_actionable_proposals_are_selected(tmp_path) -> None:
    write_proposal(tmp_path, "GOOGL", run_id="r1", quantity=100)
    write_proposal(tmp_path, "JPM", run_id="r2", quantity=22)
    selected = cli._select_all(tmp_path)
    assert {d.symbol for _, d, _ in selected} == {"GOOGL", "JPM"}


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("hold", {"action": "HOLD", "quantity": 0}),
        ("blocked", {"blocked": True}),
        ("zero quantity", {"quantity": 0}),
        ("expired", {"hours_to_expiry": -1.0}),
    ],
)
def test_what_a_batch_never_offers(tmp_path, label: str, kwargs: dict) -> None:
    """Each of these is something ``assert_actionable`` would RAISE for on a
    single proposal. In a batch they are simply not candidates -- raising on
    the first one would let a single stale file hide fourteen good orders."""
    write_proposal(tmp_path, "GOOGL", run_id="r1", quantity=100)
    write_proposal(tmp_path, "BAD", run_id="r2", **kwargs)
    selected = cli._select_all(tmp_path)
    assert [d.symbol for _, d, _ in selected] == ["GOOGL"], label


def test_a_receipted_proposal_is_not_offered_again(tmp_path) -> None:
    """The double-submission guard, at batch scope."""
    done = write_proposal(tmp_path, "GOOGL", run_id="r1")
    touch_receipt(done)
    write_proposal(tmp_path, "JPM", run_id="r2")
    assert [d.symbol for _, d, _ in cli._select_all(tmp_path)] == ["JPM"]


def test_two_proposals_for_one_symbol_collapse_to_the_newest(tmp_path) -> None:
    """**The safety property, not a tidiness one.**

    Two proposals for one symbol are two sizings of the same intent, not two
    orders -- placing both doubles the position. Live example: AMD had pending
    proposals for +46 and +77 shares from two runs on the same day.

    It also keeps a batch clear of the stop-management hole in ``broker.py``'s
    header: the protective stop covers only the shares just bought, so two
    orders in one symbol leave two stops on one position.
    """
    older = write_proposal(tmp_path, "AMD", run_id="r1", quantity=77)
    newer = write_proposal(tmp_path, "AMD", run_id="r2", quantity=46)
    # `proposals()` sorts by mtime, so make the intent explicit.
    import os
    import time
    now = time.time()
    os.utime(older, (now - 600, now - 600))
    os.utime(newer, (now, now))

    selected = cli._select_all(tmp_path)
    assert len(selected) == 1
    assert selected[0][2].quantity == 46


def test_a_symbol_with_an_executed_proposal_is_out_of_the_batch(tmp_path) -> None:
    """**Found by the first real rehearsal, and it is the double-position bug.**

    AMD's newest proposal (+46) had been executed. Dropping receipted files
    before deduplicating left the OLDER sibling (+77) as the newest survivor,
    so the batch offered to buy 77 more shares of a position it had just
    opened -- sized against a book from before the fill. Filtering first
    reintroduced exactly what the dedupe exists to prevent.
    """
    import os
    import time

    older = write_proposal(tmp_path, "AMD", run_id="r1", quantity=77)
    newer = write_proposal(tmp_path, "AMD", run_id="r2", quantity=46)
    now = time.time()
    os.utime(older, (now - 600, now - 600))
    os.utime(newer, (now, now))
    touch_receipt(newer)

    write_proposal(tmp_path, "JPM", run_id="r3", quantity=5)

    selected = cli._select_all(tmp_path)
    assert [d.symbol for _, d, _ in selected] == ["JPM"], (
        "an older proposal for an already-executed symbol must not be offered"
    )


def test_an_unreadable_proposal_does_not_take_the_batch_down(tmp_path) -> None:
    (tmp_path / "JUNK_r0.json").write_text("{not json")
    write_proposal(tmp_path, "GOOGL", run_id="r1")
    assert [d.symbol for _, d, _ in cli._select_all(tmp_path)] == ["GOOGL"]


def test_an_empty_directory_selects_nothing(tmp_path) -> None:
    assert cli._select_all(tmp_path) == []


# --------------------------------------------------------------------------- #
# The running book -- the regression this whole design exists to prevent
# --------------------------------------------------------------------------- #


def test_a_second_order_is_checked_against_the_book_after_the_first_fill(
    seeded_book,
) -> None:
    """The defect in a shell loop, stated as arithmetic.

    Two orders, each individually under ``max_positions``. The second is only a
    breach once the first has filled -- which is invisible to any check handed
    the pre-batch snapshot.
    """
    from research_desk.intent import compliance as rules
    from research_desk.intent.engine import load_intent

    intent = load_intent()
    # A book one slot below the cap, so one new symbol fits and two do not.
    room_for_one = seeded_book.model_copy(update={
        "positions": seeded_book.positions[: intent.risk.max_positions - 1],
    })

    # Symbols DERIVED from the universe and the book rather than invented:
    # `universe_membership` is itself a blocking rule, so a made-up ticker
    # would be refused for the wrong reason and the test would pass while
    # proving nothing. (Third time this repo has been bitten by a test
    # hardcoding something that is really state.)
    free = [s for s in sorted(intent.universe.include)
            if s not in room_for_one.symbols]
    assert len(free) >= 2, "need two universe symbols the book does not hold"
    first_symbol, second_symbol = free[0], free[1]

    def check(book, symbol: str, shares: int):
        plan = OrderPlan(
            symbol=symbol, as_of=book.as_of, action="BUY", quantity=shares,
            reference_price=100.0, estimated_notional=shares * 100.0,
        )
        decision = _decision(symbol)
        return rules.check(plan, decision, intent, book, as_of=book.as_of)

    first = check(room_for_one, first_symbol, 5)
    assert not first.blocked, "the last free slot should take one new symbol"

    # The stale-book reading: still looks fine, because nothing was applied.
    stale = check(room_for_one, second_symbol, 5)
    assert not stale.blocked, (
        "pre-condition for this test: a stale book does NOT catch this"
    )

    # The running-book reading, which is what `_run_all` does.
    advanced = rules.post_trade(room_for_one, first_symbol, 5, 100.0)
    running = check(advanced, second_symbol, 5)
    assert running.blocked
    assert "max_positions" in [v.rule for v in running.blocking]


def _decision(symbol: str):
    from research_desk.models.state import FinalDecision

    return FinalDecision(
        action="BUY", symbol=symbol, conviction=0.6, target_weight_pct=3.0,
        horizon_days=45,
        rationale="r", dissent="d", invalidation="i",
        expires_at=datetime.now() + timedelta(hours=12),
    )


def test_the_batch_advances_its_book_with_post_trade() -> None:
    """Structural: the projection and the per-order veto must use ONE function.

    Two implementations of "what would the book look like" is two answers, and
    the human would be shown whichever one was wrong.
    """
    import ast

    src = Path(cli.__file__)
    tree = ast.parse(src.read_text(), filename=str(src))
    for name in ("_run_all", "_project_batch"):
        node = next(
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == name
        )
        calls = {
            inner.func.id for inner in ast.walk(node)
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
        }
        assert "post_trade" in calls, f"{name} must advance the book via post_trade"


# --------------------------------------------------------------------------- #
# Waiting for a fill
# --------------------------------------------------------------------------- #


def trade(status: str, *, filled: float = 0.0, qty: float = 10.0,
          action: str = "BUY", price: float = 100.0, order_type: str = "LMT"):
    class Order:
        orderId = 101
        totalQuantity = qty
        orderType = order_type

    Order.action = action
    return FakeTrade(
        contract=None, order=Order(),
        orderStatus=FakeOrderStatus(status=status, filled=filled,
                                    remaining=qty - filled, avgFillPrice=price),
    )


@pytest.mark.parametrize("status",
                         ["Filled", "Cancelled", "ApiCancelled", "Inactive"])
async def test_a_terminal_status_returns_immediately(status: str) -> None:
    assert await wait_for_terminal(trade(status), timeout_s=0.2) is True


async def test_a_working_order_times_out_without_raising() -> None:
    """**A timeout is not an error.** The order may fill a second later. What
    the caller cannot do is size the NEXT order against a book it cannot
    describe, so the batch stops and says why -- raising here would send the
    operator looking for a problem that is not there."""
    assert await wait_for_terminal(
        trade("Submitted"), timeout_s=0.2, poll_s=0.05
    ) is False


async def test_a_gtc_stop_is_never_waited_on() -> None:
    """A protective stop sits there working by design. Waiting for it to reach
    a terminal status would hang every batch for the full timeout and then
    report a false stall, so `_execute_one` passes the PARENT only."""
    import ast

    src = Path(cli.__file__)
    tree = ast.parse(src.read_text(), filename=str(src))
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "_execute_one"
    )
    waits = [
        inner for inner in ast.walk(node)
        if isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id == "wait_for_terminal"
    ]
    assert len(waits) == 1
    arg = waits[0].args[0]
    assert isinstance(arg, ast.Subscript), "wait on trades[0], the parent"
    assert arg.slice.value == 0


def test_a_partial_fill_moves_the_book_by_what_filled() -> None:
    """Not by what was ordered. Treating a partial as complete would make every
    later check in the batch wrong in the one direction that matters."""
    shares, price = filled_quantity(trade("Filled", filled=4.0, qty=10.0))
    assert shares == 4
    assert price == pytest.approx(100.0)


def test_a_sell_fill_is_signed_from_the_order_not_the_plan() -> None:
    shares, _ = filled_quantity(
        trade("Filled", filled=10.0, qty=10.0, action="SELL")
    )
    assert shares == -10


def test_nothing_filled_moves_the_book_not_at_all() -> None:
    assert filled_quantity(trade("Cancelled", filled=0.0)) == (0, 0.0)


# --------------------------------------------------------------------------- #
# The review screen
# --------------------------------------------------------------------------- #


def projection(**overrides) -> BatchProjection:
    body = dict(
        cash_before=100_000.0, cash_after=80_000.0, cash_pct_after=40.0,
        cash_pct_floor=5.0,
        positions_before=1, positions_after=3, positions_cap=15,
        gross_before_pct=10.0, gross_after_pct=30.0, gross_cap_pct=95.0,
        sectors=(),
    )
    body.update(overrides)
    return BatchProjection(**body)


def test_the_review_shows_every_order_and_the_aggregate() -> None:
    from research_desk.execution.confirmation import render_batch

    text = render_batch(
        [BatchRow("GOOGL", "BUY", 100, 24_800.0, ("news",)),
         BatchRow("JPM", "BUY", 22, 6_400.0)],
        projection(),
    )
    assert "NOTHING HAS BEEN SENT" in text
    assert "GOOGL" in text and "JPM" in text
    # The aggregate is the only thing a sequence of single prompts cannot show.
    assert "AGGREGATE EFFECT" in text
    assert "31,200" in text, "total notional"
    assert "max 15" in text
    # And absence stays visible here too.
    assert "partial:news" in text


@pytest.mark.parametrize(
    ("overrides", "rule"),
    [
        ({"cash_pct_after": 1.0}, "min_cash_pct"),
        ({"positions_after": 20}, "max_positions"),
        ({"gross_after_pct": 99.0}, "max_gross_exposure_pct"),
        ({"sectors": (("AI semis", 10.0, 55.0, 40.0),)}, "max_sector_pct"),
    ],
)
def test_a_projected_breach_is_named_before_anything_is_typed(
    overrides: dict, rule: str
) -> None:
    """Reported, never enforced here. The per-order ``rules.check`` against the
    running book stays the only thing that can veto an order -- two places that
    can refuse are two places that can disagree about why."""
    from research_desk.execution.confirmation import render_batch

    proj = projection(**overrides)
    assert any(rule in b for b in proj.breaches)
    text = render_batch([BatchRow("X", "BUY", 1, 100.0)], proj)
    assert "WOULD CROSS" in text
    assert rule in text


def test_a_clean_projection_says_nothing_alarming() -> None:
    from research_desk.execution.confirmation import render_batch

    assert projection().breaches == ()
    assert "WOULD CROSS" not in render_batch(
        [BatchRow("X", "BUY", 1, 100.0)], projection()
    )


def test_the_review_prompt_is_not_an_order_approval(monkeypatch) -> None:
    """It moves to the per-order gates and says so. One token standing for
    fifteen orders would make the ticker mean a set rather than a trade."""
    from research_desk.execution import confirmation as conf

    assert conf.REVIEW_WORD != "y"
    monkeypatch.setattr("builtins.input", lambda prompt="": "REVIEW")
    assert conf.confirm_batch_review([BatchRow("X", "BUY", 1, 1.0)], projection())

    monkeypatch.setattr("builtins.input", lambda prompt="": "TSM")
    assert not conf.confirm_batch_review(
        [BatchRow("TSM", "BUY", 1, 1.0)], projection()
    ), "a ticker must not satisfy the review prompt"


@pytest.mark.parametrize("boom", [EOFError, KeyboardInterrupt])
def test_a_closed_stdin_stops_the_batch(monkeypatch, boom) -> None:
    """§15.8: `execute` is never scheduled, and `--all` must be no more
    schedulable than a single order."""
    from research_desk.execution import confirmation as conf

    def raise_it(prompt=""):
        raise boom()

    monkeypatch.setattr("builtins.input", raise_it)
    assert not conf.confirm_batch_review(
        [BatchRow("X", "BUY", 1, 1.0)], projection()
    )


# --------------------------------------------------------------------------- #
# Flags that must not quietly do something else
# --------------------------------------------------------------------------- #


def test_yes_is_refused_with_all(capsys) -> None:
    """`--yes` skips the book-rewrite confirmation and nothing else. A flag
    that is silently IGNORED is worse than one refused: the operator believes
    something about what just ran."""
    assert cli.main(["--yes", "--all"]) == 2
    assert "no off switch" in capsys.readouterr().err


@pytest.mark.parametrize("flag", [["--symbol", "TSM"], ["--run-id", "abc"]])
def test_all_is_refused_with_a_single_proposal_selector(capsys, flag) -> None:
    assert cli.main(["--all", *flag]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_the_batch_gate_has_no_parameter_that_disables_it() -> None:
    """The fourth lock, matching the one on ``ConfirmationGate.confirm``."""
    import inspect

    from research_desk.execution import confirmation as conf

    for fn in (conf.confirm_batch_review, conf.render_batch):
        names = set(inspect.signature(fn).parameters)
        assert not names & {
            "yes", "force", "auto", "skip", "assume_yes", "no_confirm",
            "require_confirmation",
        }, f"{fn.__name__} grew a bypass"


# --------------------------------------------------------------------------- #
# End to end, through a fake broker. The wiring, not just the pieces.
# --------------------------------------------------------------------------- #


@pytest.fixture
def batch_env(tmp_path, monkeypatch, seeded_book):
    """Drive ``_run_all`` offline: fake broker, fixture book, tmp proposals.

    The book is a FIXTURE and the portfolio path is redirected into ``tmp_path``
    -- ``_run_all`` writes the book at the end, and a test that overwrote the
    real ``data/portfolio.yaml`` would be a test that cost you an afternoon.
    """
    from research_desk.config import Settings
    from research_desk.execution import broker, cli as cli_mod
    from research_desk.intent import engine

    from .conftest import FakeIB, FakePosition, FakeContract, FakeAccountValue

    settings = Settings(data_dir=tmp_path)
    monkeypatch.setattr(cli_mod, "load_settings", lambda: settings)
    monkeypatch.setattr(cli_mod, "resolve_ib_endpoint", lambda s: ("fake", 1))

    # One slot free, so a second new symbol trips max_positions once the first
    # order fills -- the whole point of the running book.
    intent = engine.load_intent()
    book = seeded_book.model_copy(update={
        "positions": seeded_book.positions[: intent.risk.max_positions - 1],
    })
    monkeypatch.setattr(engine, "load_portfolio", lambda **kw: book)
    monkeypatch.setattr(engine, "portfolio_path",
                        lambda **kw: tmp_path / "portfolio.yaml")

    ib = FakeIB(
        fills=True,
        positions=[
            FakePosition(FakeContract(symbol=p.symbol), p.quantity, p.avg_cost)
            for p in book.positions
        ],
        account_values=[
            FakeAccountValue("NetLiquidation", f"{book.equity:.2f}"),
            FakeAccountValue("TotalCashValue", f"{book.cash:.2f}"),
        ],
    )

    async def fake_connect(**kwargs):
        ib.calls.append("connect")
        ib.managedAccounts()
        return ib, ["DU1234567"]

    monkeypatch.setattr(broker, "connect", fake_connect)
    # Nothing should sit waiting in a unit test.
    monkeypatch.setattr(broker, "FILL_TIMEOUT_S", 0.2)

    free = [s for s in sorted(intent.universe.include) if s not in book.symbols]
    return {
        "settings": settings, "ib": ib, "book": book, "intent": intent,
        "proposals": settings.proposals_dir, "free": free, "tmp": tmp_path,
    }


def answer_each_gate(monkeypatch, *, review: str = "REVIEW",
                     decline: set[str] | None = None) -> list[str]:
    """Type the right ticker at whatever order comes up, in whatever order.

    The batch offers proposals newest-first, so a test that hardcoded the
    sequence would be asserting `proposals()`'s sort rather than the behaviour
    it means to check -- and would break the next time a fixture's mtimes
    landed differently. This reads the expected ticker out of the prompt, which
    is exactly what a human does.
    """
    skip = {s.upper() for s in (decline or set())}
    seen: list[str] = []

    def typed(prompt: str = "") -> str:
        if "REVIEW" in prompt:
            return review
        # "Type TSM to place this order, or anything else to skip: "
        expected = prompt.split("Type ", 1)[1].split(" to place", 1)[0]
        seen.append(expected)
        return "no" if expected.upper() in skip else expected

    monkeypatch.setattr("builtins.input", typed)
    return seen


def args(**overrides):
    import argparse

    body = dict(all=True, dry_run=False, force=False, symbol=None,
                run_id=None, yes=False, list=False, sync_book=False,
                check_session=False, command=None)
    body.update(overrides)
    return argparse.Namespace(**body)


async def test_a_dry_run_batch_writes_no_receipts(batch_env, monkeypatch) -> None:
    """A rehearsal that consumed the proposals would be worse than no
    rehearsal. Same property the single-order dry run has, at batch scope."""
    first, second = batch_env["free"][0], batch_env["free"][1]
    write_proposal(batch_env["proposals"], first, run_id="r1", quantity=3)
    write_proposal(batch_env["proposals"], second, run_id="r2", quantity=3)
    monkeypatch.setattr("builtins.input", lambda prompt="": "REVIEW")

    assert await cli._run_all(args(dry_run=True)) == 0

    assert batch_env["ib"].count("placeOrder") == 0
    assert not list(batch_env["proposals"].glob("*.result.json"))
    # Still selectable, which is the thing that actually matters.
    assert len(cli._select_all(batch_env["proposals"])) == 2


async def test_the_second_order_is_refused_once_the_first_fills(
    batch_env, monkeypatch, capsys
) -> None:
    """**The regression this whole feature exists to prevent, end to end.**

    Two orders for two new symbols against a book with one free slot. Order 1
    fills and takes the slot; order 2 must then be refused by
    ``max_positions``. A loop re-reading the file would place both.
    """
    first, second = batch_env["free"][0], batch_env["free"][1]
    write_proposal(batch_env["proposals"], first, run_id="r1", quantity=3)
    write_proposal(batch_env["proposals"], second, run_id="r2", quantity=3)

    offered = answer_each_gate(monkeypatch)
    await cli._run_all(args())
    out = capsys.readouterr().out

    assert batch_env["ib"].count("placeOrder") >= 1
    placed = {
        json.loads(p.read_text())["outcome"]: p.name
        for p in batch_env["proposals"].glob("*.result.json")
    }
    assert "placed" in placed
    assert "rejected_stale_plan" in placed, (
        "the second order was NOT re-checked against the advanced book"
    )
    assert "max_positions" in out
    # Whichever was offered first took the slot; the second was refused.
    assert offered[0] in placed["placed"]
    assert {first, second} >= set(offered)


async def test_the_paper_gate_fires_before_every_order(
    batch_env, monkeypatch
) -> None:
    """``assert_paper_account`` once at connect and once per ``placeOrder``,
    with nothing in between. The batch must not amortise the second check
    across orders -- it is the one that catches a reconnect."""
    free = batch_env["free"]
    write_proposal(batch_env["proposals"], free[0], run_id="r1", quantity=2)
    write_proposal(batch_env["proposals"], free[1], run_id="r2", quantity=2)

    # Plenty of room, so both orders go through.
    big = batch_env["book"].model_copy(update={
        "positions": batch_env["book"].positions[:2]
    })
    from research_desk.intent import engine
    monkeypatch.setattr(engine, "load_portfolio", lambda **kw: big)
    batch_env["ib"]._positions = [
        p for p in batch_env["ib"]._positions
        if p.contract.symbol in big.symbols
    ]

    answer_each_gate(monkeypatch)
    await cli._run_all(args())

    ib = batch_env["ib"]
    placements = [i for i, c in enumerate(ib.calls) if c == "placeOrder"]
    entries = [
        i for i in placements
        if ib.placed[placements.index(i)][1].orderType != "STP"
    ]
    assert len(entries) == 2, ib.calls
    for index in entries:
        assert ib.calls[index - 1] == "managedAccounts", (
            f"something ran between the account check and placeOrder: "
            f"{ib.calls[max(0, index - 3):index + 1]}"
        )


async def test_declining_one_order_does_not_stop_the_batch(
    batch_env, monkeypatch
) -> None:
    free = batch_env["free"]
    write_proposal(batch_env["proposals"], free[0], run_id="r1", quantity=2)
    write_proposal(batch_env["proposals"], free[1], run_id="r2", quantity=2)

    big = batch_env["book"].model_copy(update={
        "positions": batch_env["book"].positions[:2]
    })
    from research_desk.intent import engine
    monkeypatch.setattr(engine, "load_portfolio", lambda **kw: big)
    batch_env["ib"]._positions = [
        p for p in batch_env["ib"]._positions
        if p.contract.symbol in big.symbols
    ]

    # Decline one by name, approve the other, whatever order they come in.
    answer_each_gate(monkeypatch, decline={free[0]})
    await cli._run_all(args())

    outcomes = {
        p.name.split("_")[0]: json.loads(p.read_text())["outcome"]
        for p in batch_env["proposals"].glob("*.result.json")
    }
    assert outcomes[free[0]] == "declined"
    assert outcomes[free[1]] == "placed"


async def test_stopping_at_the_review_places_nothing(
    batch_env, monkeypatch
) -> None:
    write_proposal(batch_env["proposals"], batch_env["free"][0], run_id="r1")
    monkeypatch.setattr("builtins.input", lambda prompt="": "no")

    assert await cli._run_all(args()) == 0
    assert batch_env["ib"].count("placeOrder") == 0
    assert not list(batch_env["proposals"].glob("*.result.json"))


async def test_an_empty_batch_says_so_rather_than_connecting(
    batch_env, monkeypatch, capsys
) -> None:
    batch_env["proposals"].mkdir(parents=True, exist_ok=True)
    assert await cli._run_all(args()) == 1
    assert "No actionable pending proposals" in capsys.readouterr().out
    assert batch_env["ib"].count("connect") == 0


async def test_an_unsettled_order_stops_the_batch(
    batch_env, monkeypatch, capsys
) -> None:
    """**Never size order N+1 against an order N that is still working.**

    The batch cannot describe the book, so it stops rather than guessing. The
    orders it never reached keep no receipt and stay pending, which is what
    makes re-running after the fill the obvious move.
    """
    free = batch_env["free"]
    big = batch_env["book"].model_copy(update={
        "positions": batch_env["book"].positions[:2]
    })
    from research_desk.intent import engine
    monkeypatch.setattr(engine, "load_portfolio", lambda **kw: big)
    batch_env["ib"]._positions = [
        p for p in batch_env["ib"]._positions
        if p.contract.symbol in big.symbols
    ]
    # Orders are placed but never fill.
    batch_env["ib"]._fills = False

    for i, symbol in enumerate(free[:3]):
        write_proposal(batch_env["proposals"], symbol, run_id=f"r{i}", quantity=2)

    answer_each_gate(monkeypatch)
    assert await cli._run_all(args()) == 0
    out = capsys.readouterr().out

    assert "STOPPED" in out
    assert "still pending" in out
    # Exactly one entry order went out, and the other two are untouched.
    receipts = list(batch_env["proposals"].glob("*.result.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["outcome"] == "placed"
    assert len(cli._select_all(batch_env["proposals"])) == 2


async def test_the_book_is_written_when_the_account_agrees(
    batch_env, monkeypatch
) -> None:
    """A batch that leaves the book N orders stale is a trap for the next
    `decide`. Unlike `--sync-book` this writes only after confirming the delta
    is the one the batch intended."""
    free = batch_env["free"]
    big = batch_env["book"].model_copy(update={
        "positions": batch_env["book"].positions[:2]
    })
    from research_desk.intent import engine
    monkeypatch.setattr(engine, "load_portfolio", lambda **kw: big)

    ib = batch_env["ib"]
    ib._positions = [p for p in ib._positions if p.contract.symbol in big.symbols]
    write_proposal(batch_env["proposals"], free[0], run_id="r1", quantity=2)

    # The account reports the new position after the fill, as it would.
    from .conftest import FakeContract, FakePosition
    original_place = ib.placeOrder

    def place_and_settle(contract, order):
        trade = original_place(contract, order)
        if getattr(order, "orderType", "") != "STP":
            ib._positions.append(
                FakePosition(FakeContract(symbol=contract.symbol),
                             float(order.totalQuantity),
                             float(order.lmtPrice))
            )
        return trade

    monkeypatch.setattr(ib, "placeOrder", place_and_settle)

    answer_each_gate(monkeypatch)
    await cli._run_all(args())

    written = batch_env["tmp"] / "portfolio.yaml"
    assert written.exists(), "the batch did not write the book"
    assert "source: ib" in written.read_text()


async def test_the_book_is_not_written_when_the_account_disagrees(
    batch_env, monkeypatch, capsys
) -> None:
    """Something happened this process did not do -- a different fill size, a
    manual trade, another session. Writing then would launder an unexplained
    change into the file that sizes the next run."""
    free = batch_env["free"]
    big = batch_env["book"].model_copy(update={
        "positions": batch_env["book"].positions[:2]
    })
    from research_desk.intent import engine
    monkeypatch.setattr(engine, "load_portfolio", lambda **kw: big)

    ib = batch_env["ib"]
    ib._positions = [p for p in ib._positions if p.contract.symbol in big.symbols]
    write_proposal(batch_env["proposals"], free[0], run_id="r1", quantity=2)

    # The order fills, but the account never reflects it -- so the running
    # book and the broker disagree at the end.
    answer_each_gate(monkeypatch)
    await cli._run_all(args())

    out = capsys.readouterr().out
    assert "does NOT match" in out
    assert "make sync-book" in out
    assert not (batch_env["tmp"] / "portfolio.yaml").exists()


# --------------------------------------------------------------------------- #
# The offset comes from CONFIG, not from the proposal
# --------------------------------------------------------------------------- #


def test_the_offset_is_read_from_config_not_the_proposal() -> None:
    """Structural, because the alternative is a live broker.

    ``limit_offset_bps`` moved into ``portfolio-intent.yaml`` and is read at
    execute time, so a proposal written yesterday says 10 while the order goes
    out at whatever config says now. Reading ``decision.limit_offset_bps`` for
    PRICING would silently revert the change.
    """
    import ast

    src = Path(cli.__file__)
    tree = ast.parse(src.read_text(), filename=str(src))
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "_execute_one"
    )

    # Over the AST, not the text: the comment above the change explains why the
    # proposal's value is NOT read, and a substring check would forbid saying so.
    def reads(dotted: str) -> bool:
        attr, obj = dotted.rsplit(".", 1)[::-1]
        return any(
            isinstance(n, ast.Attribute)
            and n.attr == attr
            and ast.unparse(n.value) == obj
            for n in ast.walk(node)
        )

    assert reads("intent.execution.limit_offset_bps")
    assert not reads("decision.limit_offset_bps"), (
        "pricing must not read the proposal's decide-time offset"
    )


def test_the_gate_is_shown_the_offset_actually_used() -> None:
    """**The one way this change could lie to the operator.**

    The approval screen's whole job is to show what is being sent. Rendering
    ``decision.limit_offset_bps`` while pricing from config would print 10 on
    an order going out at 100.
    """
    from research_desk.execution import confirmation as conf

    decision = _decision("TSM").model_copy(update={"limit_offset_bps": 10})
    plan = OrderPlan(
        symbol="TSM", as_of=AS_OF, action="BUY", quantity=12,
        reference_price=472.15, estimated_notional=5665.8,
    )
    text = conf.render_decision(
        decision, plan, limit_price=476.87, offset_bps=100,
    )
    assert "100 bps" in text
    assert "10 bps" not in text


def test_a_sell_offset_reads_as_reaching_down() -> None:
    """``marketable_limit`` SUBTRACTS the offset on a SELL. The screen said
    ``+10 bps`` for both sides, which described the wrong direction."""
    from research_desk.execution import confirmation as conf

    decision = _decision("TSM").model_copy(update={"action": "SELL"})
    plan = OrderPlan(
        symbol="TSM", as_of=AS_OF, action="SELL", quantity=-12,
        reference_price=472.15, estimated_notional=5665.8,
    )
    text = conf.render_decision(
        decision, plan, limit_price=467.43, offset_bps=100,
    )
    assert "-100 bps" in text


def test_the_configured_offset_is_bounded() -> None:
    """The one genuinely dangerous number in portfolio-intent.yaml: it decides
    how far an order will chase a price. Everything else there caps exposure."""
    import pydantic

    from research_desk.models.intent import Execution

    assert Execution().limit_offset_bps == 10, "the default stays conservative"
    for bad in (-1, 501):
        with pytest.raises(pydantic.ValidationError):
            Execution(limit_offset_bps=bad)


def test_the_shipped_config_actually_carries_the_field() -> None:
    """Every model in intent.py uses Pydantic's default ``extra='ignore'``, so
    a YAML key with no matching field is silently dropped. That makes "the
    config says 100" and "the code reads 100" two different claims."""
    from research_desk.config import load_yaml
    from research_desk.intent.engine import load_intent

    raw = (load_yaml("portfolio-intent.yaml").get("execution") or {})
    assert "limit_offset_bps" in raw, "not set in the shipped config"
    assert load_intent().execution.limit_offset_bps == raw["limit_offset_bps"]


# --------------------------------------------------------------------------- #
# An order that did not settle
# --------------------------------------------------------------------------- #


def test_the_cancel_prompt_needs_its_own_word(monkeypatch) -> None:
    """Not the ticker. The ticker places an order; this retracts one, and no
    single muscle-memory answer should do both."""
    from research_desk.execution import confirmation as conf

    assert conf.CANCEL_WORD not in {"y", "TSM"}
    monkeypatch.setattr("builtins.input", lambda prompt="": "CANCEL")
    assert conf.confirm_cancel("TSM", 53) is True

    monkeypatch.setattr("builtins.input", lambda prompt="": "TSM")
    assert conf.confirm_cancel("TSM", 53) is False


@pytest.mark.parametrize("boom", [EOFError, KeyboardInterrupt])
def test_a_closed_stdin_leaves_the_order_working(monkeypatch, boom) -> None:
    """**The opposite default from the order gate, deliberately.**

    The order gate declines on a closed stdin because doing nothing is the safe
    outcome when nobody is watching. Here the order already exists and a human
    already approved it; cancelling unattended would be the system reversing a
    decision on its own.
    """
    from research_desk.execution import confirmation as conf

    def raise_it(prompt=""):
        raise boom()

    monkeypatch.setattr("builtins.input", raise_it)
    assert conf.confirm_cancel("TSM", 53) is False


def test_the_unsettled_screen_names_the_attached_stop() -> None:
    """The reason this screen exists. The batch used to print "STOPPED" and
    exit, reading as though nothing were outstanding -- while MRVL order 53 sat
    working with a stop that would arm against a position the book had never
    heard of."""
    from research_desk.execution.confirmation import render_unsettled

    text = render_unsettled(
        "MRVL", order_id=53, action="BUY", ordered=105, filled=0,
        limit_price=269.47, status="Submitted", stop_price=247.91,
        timeout_s=30.0,
    )
    assert "ORDER 53 IS STILL WORKING" in text
    assert "247.91" in text
    assert "position the book does not know about" in text
    # And it must not imply the order is doomed -- it may be about to fill.
    assert "Leaving it is a legitimate choice" in text


async def test_a_stalled_order_offers_the_cancel_and_takes_it(
    batch_env, monkeypatch, capsys
) -> None:
    """End to end: the stall, the prompt, both legs cancelled."""
    free = batch_env["free"]
    big = batch_env["book"].model_copy(update={
        "positions": batch_env["book"].positions[:2]
    })
    from research_desk.intent import engine
    monkeypatch.setattr(engine, "load_portfolio", lambda **kw: big)
    ib = batch_env["ib"]
    ib._positions = [p for p in ib._positions if p.contract.symbol in big.symbols]
    ib._fills = False          # placed, never fills

    write_proposal(batch_env["proposals"], free[0], run_id="r1", quantity=2)
    write_proposal(batch_env["proposals"], free[1], run_id="r2", quantity=2)

    def typed(prompt: str = "") -> str:
        if "REVIEW" in prompt:
            return "REVIEW"
        if "CANCEL" in prompt:
            return "CANCEL"
        return prompt.split("Type ", 1)[1].split(" to place", 1)[0]

    monkeypatch.setattr("builtins.input", typed)
    assert await cli._run_all(args()) == 0
    out = capsys.readouterr().out

    assert "IS STILL WORKING" in out
    assert "cancelled" in out
    # Both legs: entry and its stop.
    assert len(ib.cancelled) == 2
    # The batch still stopped, and the untouched proposal is still pending.
    assert "STOPPED" in out
    assert len(cli._select_all(batch_env["proposals"])) == 1


async def test_declining_the_cancel_leaves_the_order_alone(
    batch_env, monkeypatch, capsys
) -> None:
    free = batch_env["free"]
    big = batch_env["book"].model_copy(update={
        "positions": batch_env["book"].positions[:2]
    })
    from research_desk.intent import engine
    monkeypatch.setattr(engine, "load_portfolio", lambda **kw: big)
    ib = batch_env["ib"]
    ib._positions = [p for p in ib._positions if p.contract.symbol in big.symbols]
    ib._fills = False

    write_proposal(batch_env["proposals"], free[0], run_id="r1", quantity=2)

    def typed(prompt: str = "") -> str:
        if "REVIEW" in prompt:
            return "REVIEW"
        if "CANCEL" in prompt:
            return "no"
        return prompt.split("Type ", 1)[1].split(" to place", 1)[0]

    monkeypatch.setattr("builtins.input", typed)
    await cli._run_all(args())

    assert ib.cancelled == []
    receipt = next(batch_env["proposals"].glob("*.result.json"))
    body = json.loads(receipt.read_text())
    assert body["outcome"] == "placed"
    assert body["cancelled"] is False


async def test_a_partial_fill_is_not_offered_a_cancel(
    batch_env, monkeypatch, capsys
) -> None:
    """**Not caution -- correctness.**

    The stop child is sized for the FULL quantity, so pulling the parent
    remainder leaves a stop that would oversell what was actually bought.
    Fixing that is cancel-and-replace, which FOLLOWUPS records as real work.
    """
    free = batch_env["free"]
    big = batch_env["book"].model_copy(update={
        "positions": batch_env["book"].positions[:2]
    })
    from research_desk.intent import engine
    monkeypatch.setattr(engine, "load_portfolio", lambda **kw: big)

    ib = batch_env["ib"]
    ib._positions = [p for p in ib._positions if p.contract.symbol in big.symbols]
    ib._fills = False
    write_proposal(batch_env["proposals"], free[0], run_id="r1", quantity=10)

    original = ib.placeOrder

    def partially_fill(contract, order):
        trade = original(contract, order)
        if getattr(order, "orderType", "") != "STP":
            trade.orderStatus.status = "Submitted"
            trade.orderStatus.filled = 4.0
            trade.orderStatus.remaining = 6.0
            trade.orderStatus.avgFillPrice = 100.0
        return trade

    monkeypatch.setattr(ib, "placeOrder", partially_fill)
    answer_each_gate(monkeypatch)
    await cli._run_all(args())
    out = capsys.readouterr().out

    assert "PARTIALLY FILLED" in out
    assert "no cancel is offered" in out
    assert ib.cancelled == [], "a partial fill must not be cancelled"
    # And the running book moved by what filled, not by what was ordered.
    receipt = json.loads(
        next(batch_env["proposals"].glob("*.result.json")).read_text()
    )
    assert receipt["filled"] == 4


async def test_the_receipt_records_what_settled(batch_env, monkeypatch) -> None:
    """MRVL's receipt said ``status: Submitted, filled: 0.0`` -- accurate at
    placement, and reading like a final state. ``outcome`` stays "placed"
    because that is what ``execute --list`` keys on."""
    free = batch_env["free"]
    big = batch_env["book"].model_copy(update={
        "positions": batch_env["book"].positions[:2]
    })
    from research_desk.intent import engine
    monkeypatch.setattr(engine, "load_portfolio", lambda **kw: big)
    ib = batch_env["ib"]
    ib._positions = [p for p in ib._positions if p.contract.symbol in big.symbols]

    write_proposal(batch_env["proposals"], free[0], run_id="r1", quantity=2,
                   price=100.0)
    answer_each_gate(monkeypatch)
    await cli._run_all(args())

    body = json.loads(
        next(batch_env["proposals"].glob("*.result.json")).read_text()
    )
    assert body["outcome"] == "placed"
    assert body["settled"] is True
    assert body["status"] == "Filled"
    assert body["filled"] == 2
    assert body["offset_bps_used"] == 100
