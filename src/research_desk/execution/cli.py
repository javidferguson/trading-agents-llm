"""``execute`` -- the second process. Reads a proposal, asks a human, places it.

The other half of the §0 split:

    [ decide ]  agents + LLM + free data  ->  proposal.json  ->  [ execute ]
     no IB connection, no ib_async import                  human gate + ib_async

They share nothing but the filesystem. `decide` has never held an IB connection
and `execute` has never called a model, which is what makes the dangerous half
of this system small enough to read in one sitting.

**The order of operations is the design.** Cheap refusals come first, and
nothing connects to a broker until the proposal is known to be worth acting on:

1. a receipt already exists  -> refuse (the double-submission guard)
2. the proposal has expired  -> refuse
3. the plan is blocked, zero, or a HOLD -> refuse
4. connect, **assert_paper_account (1 of 2)**, assert_can_trade
5. reconcile against the account -> on mismatch, refuse and offer a rewrite
6. quote, marketable limit, whatIf preflight
7. the human types the ticker
8. **assert_paper_account (2 of 2)**, place, journal
9. write the receipt

**`execute` is never scheduled.** §15.8: *"Do not schedule `execute`."* A cron
for `decide` is fine and intended; this one requires a person, which is the
entire point of the confirmation gate.

``--all`` does not change any of that. It reviews every actionable pending
proposal in one screen, then walks them through step 6 onward **individually**,
each with its own live price and its own typed ticker. What it adds over
running the single-order path N times is a *running book*: each order is
re-checked against the account as it is after the previous fill, because five
of ``rules.check``'s rules read the post-trade state and handing each order the
same pre-batch snapshot walks straight past all five. See ``_run_all``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..config import ConfigError, load_settings, resolve_ib_endpoint
from ..logging_setup import setup_logging
from ..models.orders import OrderPlan
from ..models.state import FinalDecision

logger = logging.getLogger(__name__)

#: Appended to a proposal's filename to record what happened to it. Its
#: presence is what makes a second `execute` on the same proposal refuse.
RECEIPT_SUFFIX = ".result.json"


class ProposalError(RuntimeError):
    """The proposal cannot be acted on, and the message says why."""


# --------------------------------------------------------------------------- #
# Proposals on disk
# --------------------------------------------------------------------------- #


def receipt_path(proposal: Path) -> Path:
    return proposal.with_name(proposal.name.removesuffix(".json") + RECEIPT_SUFFIX)


def proposals(directory: Path) -> list[Path]:
    """Proposal files, newest first. Receipts are not proposals."""
    if not directory.exists():
        return []
    return sorted(
        (p for p in directory.glob("*.json") if not p.name.endswith(RECEIPT_SUFFIX)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def load_proposal(path: Path) -> tuple[FinalDecision, OrderPlan | None, dict[str, Any]]:
    """Parse one ``proposal.json``.

    A proposal with no ``order_plan`` was written by a graph without the
    compliance node, which means nothing sized or checked it. That is refused
    rather than sized here: `execute` does not do arithmetic, and a second
    sizing implementation is how the two drift.
    """
    try:
        body = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProposalError(f"{path.name} could not be read: {exc}") from exc

    if "decision" not in body:
        raise ProposalError(f"{path.name} has no `decision` block.")

    decision = FinalDecision.model_validate(body["decision"])
    plan = (
        OrderPlan.model_validate(body["order_plan"])
        if body.get("order_plan") is not None else None
    )
    return decision, plan, body


def assert_actionable(
    path: Path, decision: FinalDecision, plan: OrderPlan | None, *, force: bool
) -> OrderPlan:
    """Every reason not to proceed, checked before touching the network."""
    existing = receipt_path(path)
    if existing.exists() and not force:
        try:
            previous = json.loads(existing.read_text())
            when = previous.get("at", "?")
            outcome = previous.get("outcome", "?")
        except (OSError, json.JSONDecodeError):
            when, outcome = "?", "?"
        raise ProposalError(
            f"{path.name} already has a receipt ({outcome} at {when}).\n"
            "Refusing to act on it twice -- running this again would place a "
            "second order and double the position.\n"
            f"  Receipt: {existing.name}\n"
            "  Pass --force only if you are certain the first attempt placed "
            "nothing."
        )

    if plan is None:
        raise ProposalError(
            f"{path.name} carries no order_plan, so nothing sized or "
            "compliance-checked it. It was written by a graph without the "
            "compliance node. Re-run `desk decide`."
        )

    # The veto is checked BEFORE the HOLD, and the order matters. A vetoed plan
    # always carries a HOLD decision -- compliance rewrites the action -- so
    # checking HOLD first makes the blocked branch unreachable and reports
    # "is a HOLD" when the useful fact is WHICH RULE refused it.
    if plan.blocked:
        rules = ", ".join(v.rule for v in plan.blocking)
        raise ProposalError(
            f"{path.name} was VETOED by compliance ({rules}).\n"
            "  " + (plan.skip_reason or "")
            + "\nPython blocked this regardless of what the fund manager "
            "decided, which is the design (§9). It is not executable."
        )

    if decision.action == "HOLD":
        raise ProposalError(
            f"{path.name} is a HOLD"
            + (" (degraded -- a failure, not a judgement)" if decision.degraded else "")
            + ". There is nothing to place."
        )

    if plan.quantity == 0:
        raise ProposalError(
            f"{path.name} sized to zero shares"
            + (f": {plan.skip_reason}" if plan.skip_reason else "")
            + "."
        )

    return plan


def write_receipt(path: Path, **fields: Any) -> Path:
    """Record the outcome next to the proposal. Written for every real outcome."""
    target = receipt_path(path)
    body = {"at": datetime.now().isoformat(), "proposal": path.name, **fields}
    target.write_text(json.dumps(body, indent=2, default=str))
    logger.info("wrote %s", target)
    return target


def _receipt(path: Path, *, dry_run: bool, **fields: Any) -> Path | None:
    """Write a receipt unless this was a rehearsal.

    **A dry run must leave the proposal pending.** Found by running one: the
    reconciliation branch wrote its receipt before consulting the flag, so
    `execute --dry-run` marked a proposal as consumed and the next real
    `execute` refused it as "already has a receipt". A rehearsal that changes
    what the next command does is not a rehearsal.
    """
    if dry_run:
        logger.info("dry run: not writing a receipt for %s", path.name)
        return None
    return write_receipt(path, **fields)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_list(args: argparse.Namespace) -> int:
    """Pending and completed proposals. No broker, no model."""
    settings = load_settings()
    found = proposals(settings.proposals_dir)
    if not found:
        print(f"No proposals in {settings.proposals_dir}.")
        print("Run `desk decide` first.")
        return 1

    pending, done = [], []
    for path in found:
        try:
            decision, plan, _ = load_proposal(path)
        except ProposalError as exc:
            done.append((path, "UNREADABLE", str(exc).splitlines()[0]))
            continue

        shares = f"{plan.quantity:+d}" if plan and plan.quantity else "-"
        note = ""
        if plan is not None and plan.blocked:
            note = "vetoed: " + ",".join(v.rule for v in plan.blocking)
        elif decision.action == "HOLD":
            note = "hold"
        elif decision.expires_at <= datetime.now():
            note = f"EXPIRED {decision.expires_at:%m-%d %H:%M}"

        row = (path, f"{decision.action} {shares}", note)
        (done if receipt_path(path).exists() else pending).append(row)

    print(f"PENDING ({len(pending)})")
    for path, action, note in pending:
        print(f"  {path.name:<46} {action:<10} {note}")
    if not pending:
        print("  (none)")

    if done:
        print(f"\nALREADY EXECUTED OR UNREADABLE ({len(done)})")
        for path, action, note in done:
            print(f"  {path.name:<46} {action:<10} {note}")

    print("\n`execute --symbol SYM`, `--run-id PREFIX`, or no argument for the "
          "newest pending one.")
    return 0


def _select(args: argparse.Namespace, directory: Path) -> Path:
    found = proposals(directory)
    if not found:
        raise ProposalError(f"No proposals in {directory}. Run `desk decide` first.")

    if args.run_id:
        matches = [p for p in found if args.run_id in p.name]
        if not matches:
            raise ProposalError(f"No proposal matching run-id {args.run_id!r}.")
        return matches[0]

    if args.symbol:
        wanted = args.symbol.strip().upper()
        matches = [p for p in found if p.name.upper().startswith(wanted + "_")]
        if not matches:
            raise ProposalError(f"No proposal for {wanted}.")
        # Newest for that symbol that has no receipt, else the newest overall
        # so the receipt refusal explains itself rather than saying "no
        # proposal".
        fresh = [p for p in matches if not receipt_path(p).exists()]
        return (fresh or matches)[0]

    fresh = [p for p in found if not receipt_path(p).exists()]
    if not fresh:
        raise ProposalError(
            "Every proposal already has a receipt. `execute --list` shows them; "
            "run `desk decide` for a new one."
        )
    return fresh[0]


def _select_all(
    directory: Path,
) -> list[tuple[Path, FinalDecision, OrderPlan]]:
    """Every proposal a batch may act on, newest first, **one per symbol.**

    Not ``_select`` in a loop: that returns ``[0]`` and deliberately keeps
    receipted files so its refusal can explain itself. A batch needs the
    opposite -- the set of things that *can* be placed -- so the filtering is
    inverted here rather than bolted onto the single-order selector.

    Dropped: receipted (already acted on), unreadable, expired, HOLD, blocked,
    and zero-quantity. Each of those is a refusal ``assert_actionable`` would
    raise for one proposal; in a batch they are simply not candidates, and
    raising on the first one would make a single stale file hide fourteen good
    orders.

    **Deduplicated by symbol, newest wins, and that is a safety property.**
    Two proposals for one symbol are two sizings of the same intent, not two
    orders -- placing both doubles the position. It is not hypothetical: AMD
    had pending proposals for ``+46`` and ``+77`` shares from two runs on the
    same day. It also keeps a batch out of the known stop-management hole at
    ``broker.py``'s header: the protective stop covers only the shares just
    bought, so two orders in one symbol would leave two stops on one position.

    **A symbol with ANY receipt is out of the batch entirely**, not merely that
    one file. Found by running the first real rehearsal: AMD's newest proposal
    (``+46``) had been executed, so dropping receipted files *first* left its
    older sibling (``+77``) as the newest survivor -- and the batch offered to
    buy 77 more shares of a position it had just opened, sized against a book
    from before the fill. Dedupe-after-filter reintroduced exactly the
    double-position the dedupe exists to prevent.

    The rule reads the same way the receipt does: a receipt means *this symbol
    has been acted on*, and every older proposal for it was sized against an
    even older book. Re-run ``desk decide`` for a current one.
    """
    now = datetime.now()
    out: list[tuple[Path, FinalDecision, OrderPlan]] = []
    seen: set[str] = set()

    acted_on = {
        _symbol_of(path) for path in proposals(directory)
        if receipt_path(path).exists()
    }

    for path in proposals(directory):
        if receipt_path(path).exists():
            continue
        try:
            decision, plan, _ = load_proposal(path)
        except ProposalError:
            continue
        if plan is None or plan.blocked or plan.quantity == 0:
            continue
        if decision.action == "HOLD":
            continue
        if decision.expires_at <= now:
            continue
        symbol = decision.symbol.upper()
        if symbol in acted_on:
            logger.info(
                "%s: %s already has an executed proposal today -- out of the "
                "batch. Re-run `desk decide` for a current sizing.",
                path.name, symbol,
            )
            continue
        if symbol in seen:
            logger.info(
                "%s: superseded by a newer proposal for %s", path.name, symbol
            )
            continue
        seen.add(symbol)
        out.append((path, decision, plan))

    return out


def _symbol_of(path: Path) -> str:
    """The symbol from a proposal filename, ``SYMBOL_runid.json``.

    From the name rather than the contents because this runs over receipted
    files too, and re-reading every one of them to learn something the
    filename already carries is work for nothing. ``persist`` writes the name;
    ``BRK.B_20261008-...json`` splits correctly because the run id has no dot.
    """
    return path.name.split("_", 1)[0].upper()


def _project_batch(
    selected: list[tuple[Path, FinalDecision, OrderPlan]],
    book: Any,
    intent: Any,
) -> tuple[list[Any], Any]:
    """Fold every order onto the book and read the caps off the result.

    Uses ``compliance.post_trade`` -- the same function the per-order veto uses
    internally -- so the projection on the review screen cannot disagree with
    the checks that follow it. A second implementation of "what would the book
    look like" is a second answer, and the human would be shown whichever one
    was wrong.
    """
    from .confirmation import BatchProjection, BatchRow
    from ..intent.compliance import post_trade, sector_exposure_pct
    from ..intent.engine import sector_map

    sectors = sector_map()
    rows = [
        BatchRow(
            symbol=decision.symbol,
            action=plan.action,
            quantity=plan.quantity,
            notional=abs(plan.quantity) * (plan.reference_price or 0.0),
            absent_analysts=tuple(decision.absent_analysts),
        )
        for _, decision, plan in selected
    ]

    projected = book
    for _, decision, plan in selected:
        price = plan.reference_price or 0.0
        if price > 0:
            projected = post_trade(projected, decision.symbol, plan.quantity, price)

    # Only the sectors this batch actually moves. Listing all seven themes
    # would bury the two that changed.
    touched = {
        sectors[d.symbol] for _, d, _ in selected
        if sectors.get(d.symbol)
    }
    allow_shorts = intent.universe.allow_shorts
    return rows, BatchProjection(
        cash_before=book.cash,
        cash_after=projected.cash,
        cash_pct_after=projected.cash_pct,
        cash_pct_floor=intent.risk.min_cash_pct,
        positions_before=book.position_count,
        positions_after=projected.position_count,
        positions_cap=intent.risk.max_positions,
        gross_before_pct=book.gross_exposure_pct,
        gross_after_pct=projected.gross_exposure_pct,
        gross_cap_pct=intent.risk.effective_gross_cap_pct(
            allow_shorts=allow_shorts
        ),
        sectors=tuple(
            (
                name,
                sector_exposure_pct(book, sectors, name),
                sector_exposure_pct(projected, sectors, name),
                intent.risk.max_sector_pct,
            )
            for name in sorted(touched)
        ),
    )


async def _check_session(args: argparse.Namespace) -> int:
    """Prove the Gateway has a LIVE API session, not merely an open port.

    **The port is not evidence and §14 says why.** The image runs socat relaying
    4004 -> 4002, and socat listens from container start regardless of whether
    the Gateway behind it ever logged in -- so a TCP probe reports healthy while
    the Gateway sits on the login screen. `make check-gateway` and `desk doctor`
    both do a bare connect, deliberately, so that `decide` keeps no ib_async in
    its dependency tree. This is the honest check, and it lives here because
    this is the process allowed to import ib_async.

    It matters in a specific, non-obvious way: logging into IB's web portal with
    the same credentials can evict the Gateway's session -- one username, one
    session -- and nothing about the port will change when it does.
    """
    from .broker import connect
    from .reconcile import account_values, holdings_from_ib

    settings = load_settings()

    try:
        host, port = resolve_ib_endpoint(settings)
    except ConfigError as exc:
        print(f"  no Gateway endpoint answered TCP: {exc}", file=sys.stderr)
        return 1

    try:
        ib, accounts = await connect(
            host=host, port=port, client_id=settings.ib_client_id,
            mode=settings.mode,
        )
    except Exception as exc:  # noqa: BLE001 -- a CLI verdict, not a traceback
        print(f"  PORT OPEN AT {host}:{port} BUT NO API SESSION", file=sys.stderr)
        print(f"    {type(exc).__name__}: {exc}", file=sys.stderr)
        print("    The Gateway is not logged in, or its session was evicted --"
              " logging", file=sys.stderr)
        print("    into IB's web portal with the same credentials does exactly"
              " that.", file=sys.stderr)
        print("    Fix: make gateway-stop && make gateway-start, then watch"
              " `make gateway-logs`.", file=sys.stderr)
        return 1

    try:
        holdings = await holdings_from_ib(ib)
        values = await account_values(ib)
        print(f"  ok   live API session on {host}:{port}")
        print(f"  ok   account(s): {', '.join(accounts)}")
        print(f"       {len(holdings)} position(s), "
              f"NetLiquidation {values.get('NetLiquidation') or 0:,.2f}, "
              f"cash {values.get('TotalCashValue') or 0:,.2f}")
        for symbol, holding in sorted(holdings.items()):
            print(f"         {symbol:8} {holding.quantity:>10,.0f} "
                  f"@ avg {holding.avg_cost:,.2f}")
        return 0
    finally:
        ib.disconnect()


async def _sync_book(args: argparse.Namespace) -> int:
    """Overwrite ``data/portfolio.yaml`` from the broker. The missing command.

    `decide` cannot do this -- it has no IB connection and must not grow one
    (§0) -- and ``make portfolio-refresh`` cannot either: it re-prices positions
    the book ALREADY LISTS, from the bars cache, and never talks to IB. So after
    a fill there was no command that could tell the book a new position exists.
    Reconciliation would eventually catch it on the next `execute` and offer the
    rewrite, but that is a circuitous route to something you want directly.

    Shows the diff first and confirms, because it overwrites a file. Quantities
    and cash come from the broker and are authoritative; marks are each
    position's average cost, so follow it with ``make portfolio-refresh`` to
    mark to market from the bars cache.
    """
    from .broker import connect
    from .confirmation import confirm_rewrite
    from .reconcile import (
        account_values,
        book_from_ib,
        compare,
        holdings_from_ib,
        render,
    )
    from ..intent.engine import load_portfolio, portfolio_path

    settings = load_settings()

    try:
        host, port = resolve_ib_endpoint(settings)
    except ConfigError as exc:
        raise ProposalError(str(exc)) from exc

    ib, accounts = await connect(
        host=host, port=port, client_id=settings.ib_client_id, mode=settings.mode,
    )
    try:
        holdings = await holdings_from_ib(ib)
        values = await account_values(ib)

        try:
            current = load_portfolio()
            stale_after = current.stale_after_days
            print(render(compare(
                current, holdings,
                broker_equity=values.get("NetLiquidation"),
                broker_cash=values.get("TotalCashValue"),
                broker_account=", ".join(accounts),
            )))
        except FileNotFoundError:
            stale_after = None
            print(f"No book yet. The broker reports {len(holdings)} position(s).")

        if args.yes or confirm_rewrite(portfolio_path()):
            fresh = await book_from_ib(ib, stale_after_days=stale_after)
            _write_book(fresh, portfolio_path())
            print(f"\nWrote {portfolio_path()}")
            print(f"  {fresh.position_count} position(s), cash {fresh.cash:,.2f}, "
                  f"equity {fresh.equity:,.2f}, source=ib")
            print()
            print("Marks are each position's average cost. Mark them to market:")
            print("    make portfolio-refresh")
            return 0

        print("\nLeft the book alone.")
        return 1
    finally:
        ib.disconnect()


@dataclass(frozen=True)
class OrderOutcome:
    """What happened to one proposal. The unit a batch reports and loops on.

    ``stop_batch`` is the only field that is not a record: it is set when the
    batch cannot honestly continue -- an order that is still working, so the
    book's next state is unknown. Everything else (a refusal, a decline, a
    missing quote) is per-order and the batch moves on.
    """

    path: Path
    symbol: str
    outcome: str
    reason: str = ""
    filled: int = 0
    avg_price: float = 0.0
    stop_batch: bool = False
    #: Record-only, for the closing summary. The Trade objects deliberately do
    #: NOT travel out of `_execute_one`: the cancel decision is made where the
    #: stall is detected, so nothing downstream needs a handle on a live order.
    order_id: int | None = None
    cancelled: bool = False


async def _execute_one(
    *,
    ib: Any,
    path: Path,
    decision: FinalDecision,
    plan: OrderPlan,
    body: dict,
    book: Any,
    intent: Any,
    journal: Any,
    gate: Any,
    args: argparse.Namespace,
    universe: dict,
    accounts: list[str],
    session: Any = None,
    prior_in_batch: int = 0,
) -> OrderOutcome:
    """Price, re-check, gate and place ONE order against ``book``.

    Extracted from ``_run`` so that ``_run_all`` can call it per order with a
    *running* book rather than the file loaded once at the start. That
    distinction is the whole reason a batch is not a shell loop: five of
    ``rules.check``'s rules read ``post_trade(portfolio, ...)``, so handing
    each order the same pre-batch snapshot silently under-counts
    ``max_positions``, ``min_cash_pct``, ``max_gross_exposure_pct``,
    ``max_sector_pct`` and ``max_position_pct``.

    The sequence below is load-bearing and ``tests/execution/test_execute_cli``
    asserts its source order: reconciliation (the caller's job, once per
    session), then the compliance re-check, and only then the gate. Never draw
    an approval screen for an order that cannot be placed.
    """
    from .broker import (
        FILL_TIMEOUT_S,
        NoQuoteError,
        build_orders,
        cancel,
        contract_for,
        filled_quantity,
        marketable_limit,
        place,
        preflight,
        quote,
        stop_price_for,
        wait_for_terminal,
    )
    from .confirmation import confirm_cancel, render_unsettled
    from ..intent import compliance as rules
    from ..intent.engine import sector_map

    run_id = body.get("run_id", "?")

    # --- price it -----------------------------------------------------------
    contract = contract_for(
        decision.symbol, _provider_symbols(universe, decision.symbol)
    )
    await ib.qualifyContractsAsync(contract)

    try:
        market = await quote(ib, contract)
    except NoQuoteError as exc:
        _receipt(path, dry_run=args.dry_run, outcome="no_quote", reason=str(exc))
        return OrderOutcome(path, decision.symbol, "no_quote", reason=str(exc))

    # The offset comes from CONFIG, not from the proposal.
    #
    # `decision.limit_offset_bps` is the decide-time record and stays in the
    # journal; pricing reads `intent.execution.limit_offset_bps`, so a change
    # to portfolio-intent.yaml applies to proposals already written rather than
    # needing a re-decide. The gate is handed the value actually used, because
    # the two can differ and the approval screen must not print the other one.
    offset_bps = intent.execution.limit_offset_bps
    limit_price = marketable_limit(market, plan.action, offset_bps)
    # Anchored on the FRESH QUOTE, not the limit. A wide offset displaces the
    # limit from the market on purpose, and anchoring the stop there drags it
    # along -- turning the 8% that `cap_risk` sized on into ~7.1% at 100 bps.
    stop = stop_price_for(decision, plan, market.reference)
    parent, child = build_orders(plan, limit_price, stop)

    report = await preflight(ib, contract, parent)

    # --- re-check compliance against the book AS IT IS NOW ------------------
    #
    # The share count in this proposal was sized against the book at DECIDE
    # time. Reconciliation proved the book matches the broker; it did not ask
    # whether the order still passes the vetoes. Those are different
    # questions, and for a single symbol executed immediately they happen to
    # have the same answer -- which is why this was missing.
    #
    # They diverge the moment anything changes in between: a fill from another
    # proposal, a sweep that sized fifteen symbols against the same empty
    # book, or simply deciding before lunch and executing after. The concrete
    # failure was walking past max_positions one proposal at a time, each sync
    # making reconciliation pass without the limit ever being re-evaluated.
    rechecked = rules.check(
        plan, decision, intent, book,
        as_of=date.today(),
        sectors=sector_map(),
        # Both of these only exist at decide time. `check` degrades each to a
        # WARN rather than a block when it cannot evaluate them, which is the
        # honest behaviour: the proposal already carries the verdict from when
        # the data was available.
        #   dollar_adv -- came from the run's MarketSnapshot
        #   earnings   -- needs the provider registry
        dollar_adv=None,
        earnings=None,
        # Zero for a single order: already enforced at decide time against the
        # same journal, and re-counting it here would veto the first execution
        # of a legitimately decided proposal.
        #
        # Inside a batch it is the number of DISTINCT SYMBOLS THIS PROCESS HAS
        # ALREADY PLACED, which is a different number and the honest one.
        # `decisions_today` counts symbols the desk *decided* on; this counts
        # the ones it actually touched, and "how many positions the desk
        # touches" is what the limit says it protects.
        prior_decisions_today=prior_in_batch,
    )

    if rechecked.blocked:
        print()
        print("=" * 72)
        print(f"REJECTED -- {decision.symbol} NO LONGER PASSES COMPLIANCE.")
        print("=" * 72)
        for violation in rechecked.blocking:
            print(f"  {violation.render()}")
        print()
        print("  It was compliant when `desk decide` wrote it. The book has")
        print("  changed since -- a fill, a sync, or simply time -- and the")
        print("  share count was computed against the older one.")
        print()
        print("  You are not being asked to approve it: the number on screen")
        print("  is arithmetic over a book that no longer exists.")
        print()
        print("  Fix: re-run `desk decide` so the order is sized against what")
        print("  the account holds now.")
        print("=" * 72)

        journal.write(
            "compliance_recheck_failed", run_id=run_id,
            symbol=decision.symbol, quantity=plan.quantity,
            rules=[v.rule for v in rechecked.blocking],
        )
        _receipt(
            path, dry_run=args.dry_run, outcome="rejected_stale_plan",
            rules=[v.rule for v in rechecked.blocking],
            limit_price=limit_price,
        )
        return OrderOutcome(
            path, decision.symbol, "rejected_stale_plan",
            reason=", ".join(v.rule for v in rechecked.blocking),
        )

    # Warnings the re-check surfaced that the original plan did not carry (an
    # ADV check that can no longer be evaluated, say) are merged in so the
    # human sees them on the gate rather than nowhere.
    if rechecked.warnings:
        plan = plan.model_copy(update={"violations": rechecked.violations})

    # --- the gate -----------------------------------------------------------
    approved = gate.confirm(
        decision, plan,
        limit_price=limit_price, stop_price=stop,
        preflight=report, quote_source=market.source,
        offset_bps=offset_bps,
        session_warning=session.warning if session is not None else None,
    )

    if not approved:
        journal.write(
            "order_declined", run_id=run_id,
            symbol=decision.symbol, quantity=plan.quantity,
            limit_price=limit_price, dry_run=args.dry_run,
        )
        _receipt(path, dry_run=args.dry_run, outcome="declined",
                 limit_price=limit_price, stop_price=stop)
        print(f"\n{decision.symbol}: nothing was placed.")
        return OrderOutcome(path, decision.symbol, "declined")

    # --- place --------------------------------------------------------------
    trades = await place(
        ib, contract, parent, child, journal=journal, run_id=run_id,
    )
    # Through the guard like every other outcome. A dry run cannot reach here
    # -- the gate declined -- but keeping the invariant total means a future
    # outcome cannot bypass it by being added in the wrong place.
    _receipt(
        path, dry_run=args.dry_run, outcome="placed",
        limit_price=limit_price, stop_price=stop,
        accounts=accounts,
        # The offset ACTUALLY used, which is config's and not the proposal's.
        # Recorded here as well as in the second write because a single-order
        # run never reaches the second one, and a receipt that does not say
        # which offset priced the limit cannot be audited against the config
        # that has since changed.
        offset_bps_used=offset_bps,
        orders=[
            {
                "order_id": t.order.orderId,
                "action": t.order.action,
                "quantity": t.order.totalQuantity,
                "type": t.order.orderType,
                "status": t.orderStatus.status,
                "filled": t.orderStatus.filled,
                "avg_fill_price": t.orderStatus.avgFillPrice,
            }
            for t in trades
        ],
    )
    print(f"\nPlaced {len(trades)} order(s) for {decision.symbol}.")

    # --- what filled, so the caller can advance the book --------------------
    #
    # Only a batch needs this. A single order is followed by a human reading
    # the fill and running `make sync-book`; a batch has to check the NEXT
    # order against the book, and "probably filled" is not an input a
    # compliance check can use.
    if not args.all:
        return OrderOutcome(path, decision.symbol, "placed")

    # The PARENT only. The child is a GTC stop whose job is to stay working.
    #
    # Passed explicitly rather than relying on the default: a default argument
    # is bound at import, so the value in the message below could drift from
    # the value actually waited for, and the operator would be told the wrong
    # number about the one thing that just stopped their batch.
    settled = await wait_for_terminal(trades[0], timeout_s=FILL_TIMEOUT_S)
    shares, price = filled_quantity(trades[0])
    order_id = trades[0].order.orderId

    def _settled(**extra: Any) -> OrderOutcome:
        """The receipt, enriched, and the outcome. Written a SECOND time.

        The first write happened immediately after `place`, and that one is
        what makes the double-submission guard exist if this process dies
        during the wait -- which is exactly the failure that guard is for. So
        this does not move it; it adds what only became known afterwards.
        MRVL's receipt said `status: Submitted, filled: 0.0`, accurate at
        placement and reading like a final state.

        `outcome` stays "placed" on purpose. It is what `execute --list` keys
        on and what the batch tests assert; the settlement detail belongs in
        fields beside it, not in a different verdict.
        """
        _receipt(
            path, dry_run=args.dry_run, outcome="placed",
            limit_price=limit_price, stop_price=stop, accounts=accounts,
            offset_bps_used=offset_bps,
            settled=settled,
            status=trades[0].orderStatus.status,
            filled=shares,
            avg_fill_price=price,
            orders=[
                {
                    "order_id": t.order.orderId,
                    "action": t.order.action,
                    "quantity": t.order.totalQuantity,
                    "type": t.order.orderType,
                    "status": t.orderStatus.status,
                    "filled": t.orderStatus.filled,
                    "avg_fill_price": t.orderStatus.avgFillPrice,
                }
                for t in trades
            ],
            **extra,
        )
        return OrderOutcome(
            path, decision.symbol, "placed", filled=shares, avg_price=price,
            order_id=order_id, **{
                k: v for k, v in extra.items()
                if k in {"cancelled", "reason", "stop_batch"}
            },
        )

    if settled:
        return _settled()

    # --- it did not settle --------------------------------------------------
    print(render_unsettled(
        decision.symbol, order_id=order_id, action=plan.action,
        ordered=plan.quantity, filled=abs(shares), limit_price=limit_price,
        status=trades[0].orderStatus.status, stop_price=stop,
        timeout_s=FILL_TIMEOUT_S,
    ))
    reason = f"still {trades[0].orderStatus.status} after {int(FILL_TIMEOUT_S)}s"

    # A PARTIAL fill is not offered a cancel, and the reason is not caution.
    # The stop child is sized for the FULL quantity, so pulling the parent
    # remainder leaves a stop that would oversell what was actually bought.
    # Fixing that is cancel-and-replace, which FOLLOWUPS records as real work;
    # half-building it here would be worse than saying so.
    if shares:
        print(f"  PARTIALLY FILLED ({abs(shares)} of {abs(plan.quantity)}), so "
              "no cancel is offered:")
        print("  the attached stop covers the full order and pulling the rest")
        print("  would leave it oversized. Handle this one in the IB UI.")
        return _settled(reason=reason, stop_batch=True)

    cancelled = False
    if not args.dry_run and confirm_cancel(decision.symbol, order_id):
        cancelled = await cancel(
            ib, trades, journal=journal, run_id=run_id, symbol=decision.symbol,
        )
        print(f"  cancelled {len(trades)} order(s) for {decision.symbol}."
              if cancelled else
              "  the cancel was not acknowledged -- check the IB UI.")

    return _settled(reason=reason, stop_batch=True, cancelled=cancelled)


async def _run(args: argparse.Namespace) -> int:
    from .broker import connect
    from .confirmation import (
        CLIConfirmationGate,
        ExpiredProposalError,
        RejectAllGate,
        assert_not_expired,
    )
    from .journal import Journal
    from ..config import load_yaml
    from ..intent.engine import load_intent, load_portfolio

    settings = load_settings()

    # --- the refusals that need nothing but the file ------------------------
    path = _select(args, settings.proposals_dir)
    decision, plan, body = load_proposal(path)
    print(f"proposal: {path.name}")

    plan = assert_actionable(path, decision, plan, force=args.force)

    try:
        assert_not_expired(decision)
    except ExpiredProposalError as exc:
        _receipt(path, dry_run=args.dry_run, outcome="expired", reason=str(exc))
        raise ProposalError(str(exc)) from exc

    book = load_portfolio()
    intent = load_intent()

    # --- the broker ---------------------------------------------------------
    try:
        host, port = resolve_ib_endpoint(settings)
    except ConfigError as exc:
        raise ProposalError(str(exc)) from exc

    ib, accounts = await connect(
        host=host, port=port, client_id=settings.ib_client_id, mode=settings.mode,
    )

    journal = Journal(settings.journal_dir)

    try:
        matched = await _reconcile(
            ib, book, accounts, journal, args,
            run_id=body.get("run_id", "?"), symbol=decision.symbol,
            receipt_for=path,
        )
        if not matched:
            return 2

        universe = load_yaml("universe.yaml")
        outcome = await _execute_one(
            ib=ib, path=path, decision=decision, plan=plan, body=body,
            book=book, intent=intent, journal=journal,
            gate=RejectAllGate() if args.dry_run else CLIConfirmationGate(),
            args=args, universe=universe, accounts=accounts,
            session=await _session_once(ib, universe, decision.symbol),
        )

        if outcome.outcome == "no_quote":
            raise ProposalError(outcome.reason)
        if outcome.outcome == "rejected_stale_plan":
            return 2
        if outcome.outcome == "placed":
            _print_sync_reminder()
        return 0

    finally:
        ib.disconnect()


async def _run_all(args: argparse.Namespace) -> int:
    """``execute --all``: a reviewed batch, one running book, one gate per order.

    Three things make this more than ``_run`` in a shell loop, and each one is
    a defect in the loop version:

    **The book advances.** ``_execute_one`` re-checks every order against the
    book as it is after the previous fill, using ``compliance.post_trade`` --
    the same function the veto itself uses. A loop that re-read
    ``data/portfolio.yaml`` would see the pre-batch state and walk straight
    past every portfolio-level limit.

    **The aggregate is shown first.** Every order was sized and vetoed in
    isolation, so a set of individually-compliant orders can still move the
    book somewhere nobody chose. The review screen is the only place that is
    visible; no sequence of per-order prompts can show it.

    **An ambiguous fill stops the batch.** Never size order N+1 against an
    order N that is still working. Same rule as reconciliation's: a thing we
    cannot describe is a refusal, not a warning.

    What it deliberately does NOT do is weaken the gate. Each order goes
    through the same ``CLIConfirmationGate`` a single run uses, and approval is
    still the ticker.
    """
    from .broker import connect
    from .confirmation import (
        CLIConfirmationGate,
        RejectAllGate,
        confirm_batch_review,
    )
    from .journal import Journal
    from .reconcile import account_values, book_from_ib, holdings_from_ib
    from ..config import load_yaml
    from ..intent.compliance import post_trade
    from ..intent.engine import load_intent, load_portfolio, portfolio_path

    settings = load_settings()

    selected = _select_all(settings.proposals_dir)
    if not selected:
        print(f"No actionable pending proposals in {settings.proposals_dir}.")
        print("`execute --list` shows what is there and why each one is out.")
        return 1

    book = load_portfolio()
    intent = load_intent()

    print(f"{len(selected)} actionable proposal(s), newest first:")
    for path, decision, plan in selected:
        print(f"  {path.name:<46} {decision.action} {plan.quantity:+d}")

    try:
        host, port = resolve_ib_endpoint(settings)
    except ConfigError as exc:
        raise ProposalError(str(exc)) from exc

    ib, accounts = await connect(
        host=host, port=port, client_id=settings.ib_client_id, mode=settings.mode,
    )
    journal = Journal(settings.journal_dir)

    try:
        # One reconciliation for the whole batch. A stale book refuses the
        # BATCH, and deliberately writes no per-proposal receipts: every one of
        # them is still valid once the book is synced, and consuming fifteen
        # proposals over one stale file would be the opposite of helpful.
        matched = await _reconcile(
            ib, book, accounts, journal, args,
            run_id="batch", symbol=f"{len(selected)} symbols", receipt_for=None,
        )
        if not matched:
            return 2

        universe = load_yaml("universe.yaml")
        # Asked once for the whole batch, BEFORE the review screen. Every US
        # equity shares one session, so this is one call rather than N -- and
        # putting it ahead of the review means a closed market is visible
        # before you decide to walk through fifteen orders, rather than on the
        # first gate after you already have.
        session = await _session_once(ib, universe, selected[0][1].symbol)

        rows, projection = _project_batch(selected, book, intent)
        if session is not None and session.warning:
            print()
            print(f"  >> {session.warning}")
            print("     Orders can still be placed -- they will rest until the")
            print("     market reopens -- but nothing will fill today.")
        if not confirm_batch_review(rows, projection):
            print("\nNothing was placed.")
            return 0

        gate = RejectAllGate() if args.dry_run else CLIConfirmationGate()
        outcomes: list[OrderOutcome] = []
        placed_symbols: set[str] = set()
        stopped = ""

        for index, (path, decision, plan) in enumerate(selected, start=1):
            print()
            print("-" * 72)
            print(f"ORDER {index} OF {len(selected)}   {path.name}")
            print("-" * 72)

            outcome = await _execute_one(
                ib=ib, path=path, decision=decision, plan=plan,
                body=load_proposal(path)[2],
                book=book, intent=intent, journal=journal, gate=gate,
                args=args, universe=universe, accounts=accounts,
                session=session,
                prior_in_batch=len(placed_symbols),
            )
            outcomes.append(outcome)

            if outcome.outcome == "placed":
                placed_symbols.add(outcome.symbol)
                # Advance the running book by what ACTUALLY filled. In a dry
                # run nothing filled, so project the proposed order instead --
                # otherwise a rehearsal would check every order against an
                # unchanged book and prove nothing about the sequence.
                shares = plan.quantity if args.dry_run else outcome.filled
                price = (
                    plan.reference_price or 0.0
                ) if args.dry_run else outcome.avg_price
                if shares and price > 0:
                    book = post_trade(book, outcome.symbol, shares, price)

            if outcome.stop_batch:
                stopped = outcome.reason
                break

        print()
        print("=" * 72)
        print(f"BATCH COMPLETE -- {len(outcomes)} of {len(selected)} order(s) reached")
        print("=" * 72)
        for outcome in outcomes:
            detail = f"  {outcome.reason}" if outcome.reason else ""
            fill = f" filled {outcome.filled:+d}" if outcome.filled else ""
            print(f"  {outcome.symbol:<6} {outcome.outcome:<22}{fill}{detail}")
        skipped = len(selected) - len(outcomes)
        if skipped:
            print(f"  ({skipped} not reached -- no receipt, still pending)")

        if stopped:
            print()
            print("  STOPPED: an order did not settle "
                  f"({stopped}).")
            print("  The remaining proposals were not offered, carry no receipt,")
            print("  and are still pending. Check the order in the IB UI, run")
            print("  `make sync-book`, then re-run.")

        if args.dry_run:
            print("\nDRY RUN -- nothing was sent and no receipts were written.")
            return 0

        if not placed_symbols:
            print("\nNothing was placed, so the book is unchanged.")
            return 0

        await _write_book_if_it_reconciles(
            ib, book, journal,
            holdings_from_ib=holdings_from_ib,
            account_values=account_values,
            book_from_ib=book_from_ib,
            portfolio_path=portfolio_path,
        )
        return 0

    finally:
        ib.disconnect()


async def _session_once(ib: Any, universe: dict, symbol: str) -> Any:
    """The exchange session, asked once and reused.

    Every US equity shares one regular session, so asking per order in a batch
    is the same question fifteen times over a network call. Named
    ``_session_once`` rather than anything containing "check" on purpose: the
    source-order test in ``tests/execution/test_execute_cli.py`` keys calls by
    attribute name and already owns ``"check"`` for ``rules.check``.

    Returns ``None`` on any trouble. This only ever produces a warning line, so
    an absent answer must stay silent rather than invent one.
    """
    from .broker import contract_for, market_session

    try:
        contract = contract_for(symbol, _provider_symbols(universe, symbol))
        await ib.qualifyContractsAsync(contract)
        return await market_session(ib, contract)
    except Exception as exc:  # noqa: BLE001 -- a warning must never block an order
        logger.debug("could not determine the market session: %s", exc)
        return None


async def _reconcile(
    ib: Any,
    book: Any,
    accounts: list[str],
    journal: Any,
    args: argparse.Namespace,
    *,
    run_id: str,
    symbol: str,
    receipt_for: Path | None,
) -> bool:
    """Book versus account. True to continue, False to refuse.

    Shared by ``_run`` and ``_run_all`` so there is one definition of "the
    share count was computed from this file, so the file has to be right".

    ``receipt_for`` is the proposal to mark as refused, or ``None`` for a batch
    -- see the call site in ``_run_all``.
    """
    from .confirmation import confirm_rewrite
    from .reconcile import (
        account_values,
        book_from_ib,
        compare,
        holdings_from_ib,
        render,
        render_rejection,
    )
    from ..intent.engine import portfolio_path

    holdings = await holdings_from_ib(ib)
    values = await account_values(ib)
    result = compare(
        book, holdings,
        broker_equity=values.get("NetLiquidation"),
        broker_cash=values.get("TotalCashValue"),
        broker_account=", ".join(accounts),
    )
    print()
    print(render(result))

    if result.matches:
        return True

    print(render_rejection(result, portfolio_path()))
    journal.write(
        "reconciliation_failed", run_id=run_id, symbol=symbol,
        mismatched=[r.symbol for r in result.mismatched],
    )
    if receipt_for is not None:
        _receipt(
            receipt_for, dry_run=args.dry_run, outcome="rejected_stale_book",
            mismatched=[
                {"symbol": r.symbol, "book": r.book_qty, "broker": r.broker_qty}
                for r in result.mismatched
            ],
        )

    if args.dry_run:
        print("\nDRY RUN -- not offering to rewrite the book.")
        return False

    if confirm_rewrite(portfolio_path()):
        fresh = await book_from_ib(ib, stale_after_days=book.stale_after_days)
        _write_book(fresh, portfolio_path())
        print(f"\nWrote {portfolio_path()} from the broker "
              f"({fresh.position_count} position(s), source=ib).")
        print("Marks are average cost -- run `make portfolio-refresh` "
              "to mark to market.")
        print("\nNow re-run `desk decide` so the order is sized against "
              "what the account actually holds.")
    else:
        print("\nLeft the book alone. Nothing was placed.")
    return False


async def _write_book_if_it_reconciles(
    ib: Any,
    running: Any,
    journal: Any,
    *,
    holdings_from_ib: Any,
    account_values: Any,
    book_from_ib: Any,
    portfolio_path: Any,
) -> bool:
    """Write the book from the broker, but only if the broker agrees with us.

    A batch that places five orders and leaves ``data/portfolio.yaml`` five
    orders stale is a trap for the next ``decide`` run, so unlike single-order
    `execute` this does write the book. What makes that safe rather than
    convenient is the check: ``--sync-book`` overwrites unconditionally, while
    this one first confirms that **the delta is the one the batch intended.**

    A disagreement here means something happened that this process did not do
    -- a fill at a different size, a manual trade, another session. Writing
    then would launder an unexplained change into the file that sizes the next
    run, so it refuses and names the command that overwrites on purpose.
    """
    from .reconcile import compare, render

    print()
    print("Confirming the account matches what this batch intended...")
    holdings = await holdings_from_ib(ib)
    values = await account_values(ib)
    result = compare(
        running, holdings,
        broker_equity=values.get("NetLiquidation"),
        broker_cash=values.get("TotalCashValue"),
    )

    if not result.matches:
        print(render(result))
        print()
        print("  The account does NOT match the batch's own running book, so")
        print("  the file was left alone. Something changed that this process")
        print("  did not do -- a different fill size, a manual trade, or")
        print("  another session.")
        print()
        print("  Run `make sync-book` to overwrite it from the broker")
        print("  deliberately, then `make portfolio-refresh` to mark to market.")
        journal.write(
            "batch_book_write_refused", run_id="batch", symbol="-",
            mismatched=[r.symbol for r in result.mismatched],
        )
        return False

    fresh = await book_from_ib(ib, stale_after_days=running.stale_after_days)
    _write_book(fresh, portfolio_path())
    print(f"  ok -- wrote {portfolio_path()} from the broker "
          f"({fresh.position_count} position(s), source=ib).")
    print()
    print("Marks are average cost. Run `make portfolio-refresh` to mark them")
    print("to market before the next `desk decide`.")
    return True


def _print_sync_reminder() -> None:
    print()
    print("The book no longer matches the account. Once the fill settles:")
    print("    make sync-book          # quantities and cash, FROM THE BROKER")
    print("    make portfolio-refresh  # then mark those positions to market")
    print()
    print("`portfolio-refresh` alone cannot do this -- it re-prices")
    print("positions the book already lists, from the bars cache, and never")
    print("talks to IB. It has no way to discover a position you just opened.")


def _provider_symbols(universe: dict, symbol: str) -> dict[str, str] | None:
    for entry in universe.get("symbols") or []:
        if str(entry.get("symbol", "")).upper() == symbol.upper():
            return entry.get("provider_symbols")
    return None


def _write_book(snapshot: Any, path: Path) -> None:
    """Overwrite data/portfolio.yaml, keeping its explanatory header.

    The header says why the file exists, that it must not be hand-edited, and
    which commands maintain it. Writing bare YAML over it would delete the only
    documentation at the point somebody is most likely to go looking.
    """
    import yaml

    header = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                header.append(line)
            else:
                break

    body = yaml.safe_dump(
        snapshot.to_yaml_dict(), sort_keys=False, default_flow_style=False
    )
    path.write_text("\n".join(header).rstrip() + "\n\n" + body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="execute",
        description="Place an approved proposal. Requires a human. Never scheduled.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="pending and completed proposals").set_defaults(
        func=cmd_list
    )

    parser.add_argument("--list", action="store_true",
                        help="same as the `list` subcommand")
    parser.add_argument("--sync-book", action="store_true",
                        help="overwrite data/portfolio.yaml from the broker. "
                             "Run this after a fill -- portfolio-refresh cannot, "
                             "it never talks to IB.")
    parser.add_argument("--check-session", action="store_true",
                        help="prove the Gateway has a LIVE API session. The open "
                             "port does not: socat answers it regardless.")
    parser.add_argument("--yes", action="store_true",
                        help="skip the confirmation on --sync-book only. Never "
                             "affects the order gate.")
    parser.add_argument("--all", action="store_true",
                        help="review every actionable pending proposal, then go "
                             "through them one at a time. Each order still "
                             "needs its ticker typed.")
    parser.add_argument("--symbol", default=None, help="newest proposal for this symbol")
    parser.add_argument("--run-id", default=None, help="run id, or any part of one")
    parser.add_argument("--dry-run", action="store_true",
                        help="everything up to the gate, then decline. Places nothing.")
    parser.add_argument("--force", action="store_true",
                        help="act on a proposal that already has a receipt. "
                             "Only if you are certain the first attempt placed nothing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)

    if args.list or args.command == "list":
        return cmd_list(args)

    # --- flag combinations that must not silently do something else ---------
    #
    # `--yes` skips one confirmation: the book rewrite in `--sync-book`. Its
    # help text has always promised it "never affects the order gate", and it
    # does not. But `--yes --all` is a request nobody should be able to make
    # by accident, and a flag that is quietly IGNORED is worse than one that
    # is refused -- the operator believes something about what just ran.
    if args.yes and args.all:
        print(
            "REFUSED: --yes does not apply to --all.\n\n"
            "  --yes skips the book-rewrite confirmation and nothing else.\n"
            "  Every order in a batch is approved by typing its ticker, and\n"
            "  there is no flag that changes that (architecture §9: the\n"
            "  confirmation gate has no off switch).",
            file=sys.stderr,
        )
        return 2

    if args.all and (args.symbol or args.run_id):
        print(
            "REFUSED: --all selects every actionable pending proposal, so\n"
            "  --symbol / --run-id would contradict it. Drop --all to act on\n"
            "  one proposal.",
            file=sys.stderr,
        )
        return 2

    try:
        if args.check_session:
            return asyncio.run(_check_session(args))
        if args.all:
            return asyncio.run(_run_all(args))
        if args.sync_book:
            return asyncio.run(_sync_book(args))
        return asyncio.run(_run(args))
    except ProposalError as exc:
        print(f"\nREFUSED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 -- a CLI message beats a traceback
        logger.exception("execute failed")
        print(f"\nFAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
