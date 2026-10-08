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
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
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


async def _run(args: argparse.Namespace) -> int:
    from .broker import (
        NoQuoteError,
        build_orders,
        connect,
        contract_for,
        marketable_limit,
        place,
        preflight,
        quote,
        stop_price_for,
    )
    from .confirmation import (
        CLIConfirmationGate,
        ExpiredProposalError,
        RejectAllGate,
        assert_not_expired,
        confirm_rewrite,
    )
    from .journal import Journal
    from .reconcile import (
        account_values,
        book_from_ib,
        compare,
        holdings_from_ib,
        render,
        render_rejection,
    )
    from ..config import load_yaml
    from ..intent.engine import load_portfolio, portfolio_path

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
        # --- reconcile ------------------------------------------------------
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

        if not result.matches:
            print(render_rejection(result, portfolio_path()))
            journal.write(
                "reconciliation_failed", run_id=body.get("run_id", "?"),
                symbol=decision.symbol,
                mismatched=[r.symbol for r in result.mismatched],
            )
            _receipt(
                path, dry_run=args.dry_run, outcome="rejected_stale_book",
                mismatched=[
                    {"symbol": r.symbol, "book": r.book_qty, "broker": r.broker_qty}
                    for r in result.mismatched
                ],
            )

            if args.dry_run:
                print("\nDRY RUN -- not offering to rewrite the book.")
                return 2

            if confirm_rewrite(portfolio_path()):
                fresh = await book_from_ib(
                    ib, stale_after_days=book.stale_after_days
                )
                _write_book(fresh, portfolio_path())
                print(f"\nWrote {portfolio_path()} from the broker "
                      f"({fresh.position_count} position(s), source=ib).")
                print("Marks are average cost -- run `make portfolio-refresh` "
                      "to mark to market.")
                print("\nNow re-run `desk decide` so the order is sized against "
                      "what the account actually holds.")
            else:
                print("\nLeft the book alone. Nothing was placed.")
            return 2

        # --- price it -------------------------------------------------------
        contract = contract_for(
            decision.symbol, _provider_symbols(load_yaml("universe.yaml"), decision.symbol)
        )
        await ib.qualifyContractsAsync(contract)

        try:
            market = await quote(ib, contract)
        except NoQuoteError as exc:
            _receipt(path, dry_run=args.dry_run, outcome="no_quote",
                     reason=str(exc))
            raise ProposalError(str(exc)) from exc

        limit_price = marketable_limit(market, plan.action, decision.limit_offset_bps)
        stop = stop_price_for(decision, plan, limit_price)
        parent, child = build_orders(plan, limit_price, stop)

        report = await preflight(ib, contract, parent)

        # --- the gate -------------------------------------------------------
        gate = RejectAllGate() if args.dry_run else CLIConfirmationGate()
        approved = gate.confirm(
            decision, plan,
            limit_price=limit_price, stop_price=stop,
            preflight=report, quote_source=market.source,
        )

        if not approved:
            journal.write(
                "order_declined", run_id=body.get("run_id", "?"),
                symbol=decision.symbol, quantity=plan.quantity,
                limit_price=limit_price, dry_run=args.dry_run,
            )
            _receipt(path, dry_run=args.dry_run, outcome="declined",
                     limit_price=limit_price, stop_price=stop)
            print("\nNothing was placed.")
            return 0

        # --- place ----------------------------------------------------------
        trades = await place(
            ib, contract, parent, child,
            journal=journal, run_id=body.get("run_id", "?"),
        )
        # Through the guard like every other outcome. A dry run cannot reach
        # here -- the gate declined -- but keeping the invariant total means a
        # future outcome cannot bypass it by being added in the wrong place.
        _receipt(
            path, dry_run=args.dry_run, outcome="placed",
            limit_price=limit_price, stop_price=stop,
            accounts=accounts,
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
        print(f"\nPlaced {len(trades)} order(s). "
              f"`make gateway-logs` and the trade journal have the detail.")
        print()
        print("The book no longer matches the account. Once the fill settles:")
        print("    make sync-book          # quantities and cash, FROM THE BROKER")
        print("    make portfolio-refresh  # then mark those positions to market")
        print()
        print("`portfolio-refresh` alone cannot do this -- it re-prices")
        print("positions the book already lists, from the bars cache, and never")
        print("talks to IB. It has no way to discover a position you just opened.")
        return 0

    finally:
        ib.disconnect()


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

    try:
        if args.check_session:
            return asyncio.run(_check_session(args))
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
