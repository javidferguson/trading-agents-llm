"""The ``execute`` CLI: which refusals fire, and in what order.

**The ordering is the design, not a convenience.** Cheap refusals come first,
and nothing connects to a broker until the proposal is known to be worth acting
on. So these tests mostly assert that a given proposal is refused *without the
network being touched* -- which is checked by not providing a broker at all: if
the code tried to connect, these would fail.

The sharpest one is the receipt guard. Running ``execute`` twice on one
proposal would place a second order and double the position, and nothing else in
the design prevents it.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from research_desk.execution import cli
from research_desk.models.orders import OrderPlan, Violation
from research_desk.models.state import FinalDecision

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


def plan(**overrides) -> OrderPlan:
    body = {
        "symbol": "TSM", "as_of": AS_OF, "action": "BUY", "quantity": 12,
        "reference_price": 472.15, "estimated_notional": 5665.80,
        "target_weight_pct": 4.34, "current_weight_pct": 2.08,
        "binding_cap": "requested",
    }
    body.update(overrides)
    return OrderPlan(**body)


def write_proposal(
    directory: Path,
    *,
    name: str = "TSM_20261007-120000-aaaaaa.json",
    decision_obj: FinalDecision | None = None,
    plan_obj: OrderPlan | None = None,
    include_plan: bool = True,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    body: dict = {
        "run_id": name.split("_", 1)[1].removesuffix(".json"),
        "as_of": AS_OF.isoformat(),
        "mode": "live",
        "decision": (decision_obj or decision()).model_dump(mode="json"),
    }
    if include_plan:
        body["order_plan"] = (plan_obj or plan()).model_dump(mode="json")
    path = directory / name
    path.write_text(json.dumps(body, indent=2, default=str))
    return path


def refuse(path: Path, *, force: bool = False) -> str:
    """Run the pre-network checks and return the refusal message."""
    d, p, _ = cli.load_proposal(path)
    with pytest.raises(cli.ProposalError) as exc:
        cli.assert_actionable(path, d, p, force=force)
    return str(exc.value)


# --------------------------------------------------------------------------- #
# The double-submission guard
# --------------------------------------------------------------------------- #


def test_a_proposal_with_a_receipt_is_refused(tmp_path) -> None:
    """**Running execute twice would double the position.**

    Nothing else in the design prevents it: the proposal file is unchanged by a
    successful placement, so a second run would read the same instruction and
    send it again.
    """
    path = write_proposal(tmp_path)
    cli.write_receipt(path, outcome="placed", orders=[{"order_id": 101}])

    message = refuse(path)
    assert "already has a receipt" in message
    assert "double the position" in message
    assert "--force" in message


def test_force_overrides_the_receipt(tmp_path) -> None:
    """Deliberately available, deliberately not the default -- the first
    attempt may genuinely have placed nothing."""
    path = write_proposal(tmp_path)
    cli.write_receipt(path, outcome="declined")

    d, p, _ = cli.load_proposal(path)
    assert cli.assert_actionable(path, d, p, force=True).quantity == 12


def test_the_receipt_message_names_the_previous_outcome(tmp_path) -> None:
    path = write_proposal(tmp_path)
    cli.write_receipt(path, outcome="rejected_stale_book")
    assert "rejected_stale_book" in refuse(path)


def test_a_receipt_is_not_itself_a_proposal(tmp_path) -> None:
    """``*.result.json`` matches ``*.json``, so the listing has to exclude it
    or every placed order becomes a new pending proposal."""
    path = write_proposal(tmp_path)
    cli.write_receipt(path, outcome="placed")
    assert cli.proposals(tmp_path) == [path]


def test_an_unreadable_receipt_still_blocks(tmp_path) -> None:
    """A corrupt receipt means *something* happened. Failing open there would
    be the one outcome worse than a confusing message."""
    path = write_proposal(tmp_path)
    cli.receipt_path(path).write_text("{ truncated")
    assert "already has a receipt" in refuse(path)


# --------------------------------------------------------------------------- #
# The other refusals, all before the network
# --------------------------------------------------------------------------- #


def test_an_expired_proposal_is_refused(tmp_path) -> None:
    from research_desk.execution.confirmation import ExpiredProposalError, assert_not_expired

    path = write_proposal(
        tmp_path, decision_obj=decision(expires_at=datetime.now() - timedelta(hours=2))
    )
    d, _, _ = cli.load_proposal(path)
    with pytest.raises(ExpiredProposalError):
        assert_not_expired(d)


def test_a_vetoed_proposal_reports_the_RULE_not_just_hold(tmp_path) -> None:
    """**Check order matters here.**

    Compliance rewrites a vetoed decision's action to HOLD, so testing HOLD
    first makes the blocked branch unreachable and reports "is a HOLD" when the
    useful fact is *which rule* refused it.
    """
    blocked = plan(
        quantity=0, action="HOLD",
        violations=[Violation(rule="max_positions", message="the book is full")],
        skip_reason="blocked by max_positions",
    )
    path = write_proposal(
        tmp_path, decision_obj=decision(action="HOLD", target_weight_pct=0.0),
        plan_obj=blocked,
    )
    message = refuse(path)
    assert "VETOED by compliance (max_positions)" in message
    assert "§9" in message


def test_a_plain_hold_is_refused_as_a_hold(tmp_path) -> None:
    path = write_proposal(
        tmp_path,
        decision_obj=decision(action="HOLD", target_weight_pct=0.0),
        plan_obj=plan(quantity=0, action="HOLD", skip_reason="the decision is HOLD"),
    )
    assert "is a HOLD" in refuse(path)


def test_a_degraded_hold_says_it_was_a_failure(tmp_path) -> None:
    """§4 puts ``degraded`` on the decision so `execute` can tell a failed HOLD
    from a decided one."""
    path = write_proposal(
        tmp_path,
        decision_obj=decision(action="HOLD", target_weight_pct=0.0, degraded=True),
        plan_obj=plan(quantity=0, action="HOLD"),
    )
    assert "a failure, not a judgement" in refuse(path)


def test_a_zero_quantity_plan_is_refused(tmp_path) -> None:
    path = write_proposal(
        tmp_path, plan_obj=plan(quantity=0, skip_reason="below min_trade_usd")
    )
    message = refuse(path)
    assert "sized to zero shares" in message
    assert "min_trade_usd" in message


def test_a_proposal_with_no_order_plan_is_refused(tmp_path) -> None:
    """`execute` does not size. A second sizing implementation is how the two
    drift, and a proposal written before the compliance node existed was never
    checked by it."""
    path = write_proposal(tmp_path, include_plan=False)
    message = refuse(path)
    assert "no order_plan" in message
    assert "compliance node" in message


def test_a_proposal_with_no_decision_block_is_refused(tmp_path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "X_1.json"
    path.write_text(json.dumps({"run_id": "1"}))
    with pytest.raises(cli.ProposalError, match="no `decision` block"):
        cli.load_proposal(path)


def test_unparseable_json_is_refused_with_the_filename(tmp_path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "X_2.json"
    path.write_text("{ not json")
    with pytest.raises(cli.ProposalError, match="X_2.json could not be read"):
        cli.load_proposal(path)


def test_an_actionable_proposal_passes_every_check(tmp_path) -> None:
    """Guard against the suite passing because everything is refused."""
    path = write_proposal(tmp_path)
    d, p, _ = cli.load_proposal(path)
    assert cli.assert_actionable(path, d, p, force=False) is p
    assert p.actionable


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


class Args:
    def __init__(self, **kw):
        self.symbol = kw.get("symbol")
        self.run_id = kw.get("run_id")
        self.dry_run = kw.get("dry_run", False)
        self.force = kw.get("force", False)
        self.list = kw.get("list", False)
        self.command = kw.get("command")


def test_selection_defaults_to_the_newest_proposal_with_no_receipt(tmp_path) -> None:
    import os
    import time

    old = write_proposal(tmp_path, name="TSM_20261007-100000-aaa.json")
    time.sleep(0.01)
    new = write_proposal(tmp_path, name="NVDA_20261007-110000-bbb.json")
    # Make the ordering unambiguous regardless of filesystem timestamp
    # granularity.
    os.utime(old, (1, 1))

    assert cli._select(Args(), tmp_path) == new


def test_selection_skips_proposals_that_already_have_receipts(tmp_path) -> None:
    import os

    done = write_proposal(tmp_path, name="NVDA_20261007-110000-bbb.json")
    cli.write_receipt(done, outcome="placed")
    pending = write_proposal(tmp_path, name="TSM_20261007-100000-aaa.json")
    os.utime(done, (9_999_999_999, 9_999_999_999))  # newest by mtime

    assert cli._select(Args(), tmp_path) == pending


def test_run_id_matches_by_substring(tmp_path) -> None:
    write_proposal(tmp_path, name="TSM_20261007-153546-a4ac1a.json")
    write_proposal(tmp_path, name="TSM_20261007-200907-18d340.json")
    chosen = cli._select(Args(run_id="153546"), tmp_path)
    assert "153546" in chosen.name


def test_symbol_selection_does_not_match_a_prefix_of_another_symbol(tmp_path) -> None:
    """``MSFT`` must not be found by asking for ``MS``, and ``TSM`` must not
    match ``TSMC``. The underscore in the filename is the boundary."""
    write_proposal(tmp_path, name="MSFT_20261007-120000-aaa.json")
    with pytest.raises(cli.ProposalError, match="No proposal for MS"):
        cli._select(Args(symbol="MS"), tmp_path)
    assert cli._select(Args(symbol="msft"), tmp_path).name.startswith("MSFT_")


def test_an_empty_directory_says_to_run_decide(tmp_path) -> None:
    with pytest.raises(cli.ProposalError, match="desk decide"):
        cli._select(Args(), tmp_path)


def test_all_proposals_executed_says_so_rather_than_none_found(tmp_path) -> None:
    path = write_proposal(tmp_path)
    cli.write_receipt(path, outcome="placed")
    with pytest.raises(cli.ProposalError, match="already has a receipt"):
        cli._select(Args(), tmp_path)


def test_selecting_by_symbol_falls_back_to_a_receipted_one(tmp_path) -> None:
    """So the receipt refusal explains itself rather than saying "no proposal
    for TSM", which would read as a different problem entirely."""
    path = write_proposal(tmp_path)
    cli.write_receipt(path, outcome="placed")
    assert cli._select(Args(symbol="TSM"), tmp_path) == path


# --------------------------------------------------------------------------- #
# The book rewrite keeps the file's documentation
# --------------------------------------------------------------------------- #


def test_rewriting_the_book_preserves_its_header(tmp_path) -> None:
    """The header says why the file exists, that it must not be hand-edited,
    and which commands maintain it. Writing bare YAML over it would delete the
    only documentation at the point somebody is most likely to look."""
    from research_desk.models.portfolio import PortfolioSnapshot, Position

    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "# The book. State, not configuration.\n"
        "# DO NOT HAND-EDIT.\n"
        "\n"
        "as_of: '2026-01-01'\ncash: 1.0\npositions: []\n"
    )

    fresh = PortfolioSnapshot(
        as_of=AS_OF, cash=90_000.0, source="ib",
        positions=[Position(symbol="TSM", quantity=11, avg_cost=370.5,
                            last_price=370.5, marked_on=AS_OF)],
    )
    cli._write_book(fresh, path)

    written = path.read_text()
    assert "DO NOT HAND-EDIT" in written
    assert "source: ib" in written
    assert PortfolioSnapshot.load(path).get("TSM").quantity == 11


# --------------------------------------------------------------------------- #
# Receipts record every outcome, not only placements
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "outcome", ["placed", "declined", "dry_run", "expired", "rejected_stale_book"]
)
def test_a_receipt_is_written_for_every_outcome(tmp_path, outcome: str) -> None:
    """`--list` reads receipts to separate pending from done, so an outcome
    with no receipt is a proposal that looks pending forever."""
    path = write_proposal(tmp_path)
    receipt = cli.write_receipt(path, outcome=outcome)
    body = json.loads(receipt.read_text())
    assert body["outcome"] == outcome
    assert body["proposal"] == path.name
    assert body["at"]


# --------------------------------------------------------------------------- #
# A dry run is a rehearsal and must change nothing
# --------------------------------------------------------------------------- #


def test_a_dry_run_writes_no_receipt(tmp_path) -> None:
    """**Found by running one against the live Gateway.**

    The reconciliation branch wrote its receipt before consulting the flag, so
    `execute --dry-run` marked the proposal consumed and the next real
    `execute` refused it as "already has a receipt". A rehearsal that changes
    what the next command does is not a rehearsal.
    """
    path = write_proposal(tmp_path)
    assert cli._receipt(path, dry_run=True, outcome="rejected_stale_book") is None
    assert not cli.receipt_path(path).exists()

    # And the proposal is still selectable afterwards.
    assert cli._select(Args(), tmp_path) == path


def test_a_real_run_does_write_a_receipt(tmp_path) -> None:
    """Guard against the fix above silencing receipts entirely."""
    path = write_proposal(tmp_path)
    receipt = cli._receipt(path, dry_run=False, outcome="declined")
    assert receipt is not None and receipt.exists()
    assert json.loads(receipt.read_text())["outcome"] == "declined"


def test_every_receipt_write_in_the_run_path_goes_through_the_guard() -> None:
    """A new outcome added later must not bypass the dry-run check.

    Asserted on the source because the alternative is noticing it the next time
    a dry run eats a proposal.
    """
    import ast
    from pathlib import Path

    src = Path(cli.__file__)
    tree = ast.parse(src.read_text(), filename=str(src))
    run = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run"
    )
    direct = [
        node for node in ast.walk(run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_receipt"
    ]
    assert not direct, (
        "_run calls write_receipt directly; use _receipt(dry_run=...) so a "
        "rehearsal leaves the proposal pending"
    )


# --------------------------------------------------------------------------- #
# The execute-time compliance re-check
# --------------------------------------------------------------------------- #
#
# `rules.check` used to be called in exactly ONE place -- the decide graph. So
# a proposal's share count was sized against the book at decide time and
# `execute` never asked whether it still passed the vetoes.
#
# Reconciliation looks like it covers this and does not: it asks "does the book
# match the broker?", never "does this order still pass against that book?".
# The two have the same answer only when nothing changed in between.
#
# The concrete failure: execute A, it fills, book != broker so execute B is
# refused, `make sync-book`, now book == broker so execute B PASSES
# reconciliation -- and places a quantity computed when you held nothing, with
# no limit re-evaluated. Repeat and you walk past max_positions one proposal at
# a time, each sync making the check pass without the check happening.


def _intent(**risk_overrides):
    from research_desk.models.intent import PortfolioIntent

    risk = {
        "per_trade_risk_pct": 0.75, "default_stop_pct": 8.0,
        "max_position_pct": 10.0, "max_sector_pct": 65.0,
        "min_cash_pct": 10.0, "max_gross_exposure_pct": 95.0,
        "max_positions": 2, "min_trade_usd": 500, "max_order_shares": 2000,
        "max_adv_participation_pct": 2.0,
        "earnings_blackout_days_before": 2, "earnings_blackout_days_after": 1,
    }
    risk.update(risk_overrides)
    return PortfolioIntent.from_dict({
        "objective": {"horizon_days": 60},
        "risk": risk,
        "universe": {"include": ["AAA", "BBB", "TSM"], "exclude": []},
        "themes": [{"name": "T", "target_weight_pct": 15.0, "conviction": 0.8,
                    "exemplars": ["AAA", "BBB", "TSM"]}],
    })


def _book(cash, **holdings):
    from research_desk.models.portfolio import PortfolioSnapshot, Position

    return PortfolioSnapshot(
        as_of=date.today(), cash=cash,
        positions=[
            Position(symbol=s, quantity=q, avg_cost=p, last_price=p,
                     marked_on=date.today())
            for s, (q, p) in holdings.items()
        ],
    )


def test_a_plan_that_breaches_max_positions_now_is_refused() -> None:
    """Sized when there was room; executed when there is not.

    The plan itself is untouched and still looks fine -- it is the BOOK that
    moved. Only a re-check against the current book can see that.
    """
    from research_desk.intent import compliance as rules

    intent = _intent(max_positions=2)
    # Two positions already held, so opening a third breaches the limit.
    book = _book(500_000.0, AAA=(100, 100.0), BBB=(100, 100.0))

    sized_when_there_was_room = plan(symbol="TSM", quantity=10,
                                     reference_price=100.0)
    result = rules.check(
        sized_when_there_was_room, decision(), intent, book,
        as_of=date.today(), sectors={}, dollar_adv=None, earnings=None,
        prior_decisions_today=0,
    )

    assert result.blocked
    assert "max_positions" in [v.rule for v in result.blocking]
    assert result.quantity == 0, "a blocked plan must carry no shares"


def test_the_same_plan_passes_when_the_book_still_has_room() -> None:
    """Guard against the re-check refusing everything."""
    from research_desk.intent import compliance as rules

    result = rules.check(
        plan(symbol="TSM", quantity=10, reference_price=100.0),
        decision(), _intent(max_positions=15), _book(500_000.0),
        as_of=date.today(), sectors={}, dollar_adv=None, earnings=None,
        prior_decisions_today=0,
    )
    assert not result.blocked
    assert result.quantity == 10


def test_decide_time_only_inputs_warn_rather_than_block() -> None:
    """`execute` has no MarketSnapshot and no provider registry, so the ADV and
    earnings checks cannot be evaluated there.

    They must degrade to warnings. Blocking on data that is simply absent at
    this stage would refuse every order, and the proposal already carries the
    verdict from when the data existed.
    """
    from research_desk.intent import compliance as rules

    result = rules.check(
        plan(symbol="TSM", quantity=10, reference_price=100.0),
        decision(), _intent(max_positions=15), _book(500_000.0),
        as_of=date.today(), sectors={}, dollar_adv=None, earnings=None,
        prior_decisions_today=0,
    )
    warned = [v.rule for v in result.warnings]
    assert "adv_participation" in warned
    assert not result.blocked, "an unevaluable check must not block"


def test_the_recheck_runs_before_the_gate_is_drawn() -> None:
    """**Placement in the flow, not merely presence.**

    Same reasoning as reconciliation: do not render an approval screen for an
    order that cannot be placed. Asserted on the source order rather than
    behaviour, because the alternative is a live broker and a human.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2]
           / "src" / "research_desk" / "execution" / "cli.py").read_text()
    body = src.split("async def _run(")[1]

    recheck = body.index("rechecked = rules.check(")
    reconcile = body.index("result = compare(")
    gate = body.index("approved = gate.confirm(")

    assert reconcile < recheck < gate, (
        "the compliance re-check must sit AFTER reconciliation (it needs a book "
        "that matches the broker) and BEFORE the gate (never prompt for an "
        "order that cannot be placed)"
    )


def test_a_stale_plan_refusal_is_its_own_receipt_outcome() -> None:
    """So `execute --list` can tell it apart from a decline or a veto."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2]
           / "src" / "research_desk" / "execution" / "cli.py").read_text()
    assert 'outcome="rejected_stale_plan"' in src


def test_execute_actually_calls_the_checker() -> None:
    """The regression guard. If this import disappears, the hole is back."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2]
           / "src" / "research_desk" / "execution" / "cli.py").read_text()
    assert "from ..intent import compliance as rules" in src
    assert "rules.check(" in src
