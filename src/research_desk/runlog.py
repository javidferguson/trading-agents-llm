"""Read and render a recorded run. The reader half of the §1 JSONL contract.

> *"This JSONL is the source of truth for replay and audit. The LangGraph
> checkpointer exists only for crash-resume within a single live run. Two
> mechanisms, two jobs."* (architecture §1)

Something had to be able to read it. Nine `desk` subcommands shipped before
this one and none of them could: `data/proposals/*.json` holds the decision and
the order plan but none of the reasoning, Langfuse shows one run beautifully but
cannot answer "every SELL this week", and the journal itself is ~50KB per run.

**One renderer, two call sites.** `desk decide` and `desk review` both print
through `render_run`, so a live run and a replayed one cannot drift. That was
the actual reason for a separate module: the printing used to live inside
`cli._run_decide`, where `review` could only have copied it.

**Why this is not in ``eval/``.** `eval/` is Stage 8's package and will acquire
plotting dependencies for the calibration plot. `decide` prints through this
module, so putting it there would pull matplotlib into the decide dependency
tree at import time -- exactly the creep `tests/test_layering.py` exists to
prevent. The dependency runs the other way: this stays stdlib + pydantic, and
Stage 8 imports it.

**Deliberately tolerant of a damaged tail.** A run killed mid-write leaves a
truncated final line. That is normal, and it must not make the whole day
unreadable -- the same reasoning (and the same shape) as
`intent.compliance.decisions_today`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from .models.state import DecisionState

logger = logging.getLogger(__name__)

#: How much of a debate claim a summary line shows. Long enough to tell two
#: claims apart, short enough that a 20-run listing still fits a terminal.
CLAIM_WIDTH = 110


def journal_files(journal_dir: Path, *, on: date | None = None) -> list[Path]:
    """Journal files, newest day first. One file per decision date."""
    if not journal_dir.exists():
        return []
    if on is not None:
        path = journal_dir / f"decisions_{on:%Y%m%d}.jsonl"
        return [path] if path.exists() else []
    return sorted(journal_dir.glob("decisions_*.jsonl"), reverse=True)


def _rows(path: Path) -> Iterator[dict]:
    """Parse one journal file, skipping what cannot be parsed.

    Streams rather than reading the file into a list of objects: a day of runs
    is ~50KB each and there is no reason to hold them all to answer
    ``--last 1``.
    """
    try:
        with path.open() as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A truncated last line is what a killed run leaves behind.
                    # Log at debug: it is expected, not a problem.
                    logger.debug("%s:%d is not valid JSON; skipped", path, number)
    except OSError as exc:
        logger.warning("could not read %s: %s", path, exc)


def read_runs(
    journal_dir: Path,
    *,
    symbol: str | None = None,
    on: date | None = None,
    run_id: str | None = None,
    last: int | None = None,
    actionable: bool = False,
) -> list[DecisionState]:
    """Recorded runs, newest first.

    ``run_id`` matches by prefix, so a timestamp fragment is enough to find a
    run without copying the whole id. ``actionable`` keeps only runs whose
    final decision was a BUY or SELL -- the same definition the cadence check
    uses, so "what did the desk actually do today" has one answer.

    A row that no longer validates against the current schema is skipped with a
    warning rather than raising. That is the honest behaviour for an
    append-only log the code has moved past, and it is also the signal that a
    schema change broke replay -- which is why it warns.
    """
    wanted_symbol = symbol.strip().upper() if symbol else None
    out: list[DecisionState] = []

    for path in journal_files(journal_dir, on=on):
        day: list[DecisionState] = []
        for row in _rows(path):
            if wanted_symbol and str(row.get("symbol", "")).upper() != wanted_symbol:
                continue
            if run_id and not str(row.get("run_id", "")).startswith(run_id):
                continue
            if actionable:
                final = row.get("final_decision") or {}
                if final.get("action") not in {"BUY", "SELL"}:
                    continue
            try:
                day.append(DecisionState.model_validate(row))
            except Exception as exc:  # noqa: BLE001 -- a log the code outgrew
                logger.warning(
                    "%s: run %s does not validate against the current schema "
                    "(%s). Skipped. This is what a schema change that breaks "
                    "replay looks like.",
                    path.name, row.get("run_id", "?"), type(exc).__name__,
                )
        # Within a file, later lines are later runs.
        out.extend(reversed(day))
        if last is not None and len(out) >= last:
            break

    return out[:last] if last is not None else out


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _flat(text: str) -> str:
    """Collapse the whitespace a model's multi-line string arrives with."""
    return " ".join((text or "").split())


def render_summary(state: DecisionState) -> str:
    """One line per run, for a listing."""
    decision = state.final_decision
    plan = state.order_plan

    action = decision.action if decision else "-"
    shares = f"{plan.quantity:+d}" if plan and plan.quantity else ""

    flags = []
    if decision is not None and decision.degraded:
        flags.append("DEGRADED")
    if plan is not None and plan.blocked:
        flags.append("VETOED:" + ",".join(v.rule for v in plan.blocking))
    elif plan is not None and plan.warnings:
        flags.append("warn:" + ",".join(v.rule for v in plan.warnings))

    return (
        f"{state.as_of}  {state.run_id:<22} {state.symbol:<6} "
        f"{action:<4} {shares:>6}  {' '.join(flags)}"
    ).rstrip()


def render_run(state: DecisionState, *, full: bool = False) -> str:
    """The whole run, as `desk decide` prints it live.

    ``full`` adds what a live run does not bother showing because you have just
    watched it happen: every analyst rather than only the market one, and the
    raw model calls.
    """
    out: list[str] = [
        f"{state.symbol}  as_of={state.as_of}  run={state.run_id}",
        f"prompts={state.prompt_pack_version}  config={state.config_hash}  "
        f"mode={state.mode.value}",
        "",
    ]

    for note in state.notes:
        out.append(f"  {note}")
    out.append("")

    kinds = sorted(state.analyst_reports) if full else ["market"]
    for kind in kinds:
        report = state.analyst_reports.get(kind)
        if report is None:
            continue
        out.append(f"ANALYST ({kind})  {report.stance} @ {report.confidence:.2f}")
        out.append(f"  {_flat(report.summary)}")
        out += [f"  - {_flat(point)}" for point in report.key_points]
        if report.data_gaps:
            out.append(f"  gaps: {'; '.join(report.data_gaps)}")
        out.append("")

    debate = state.research_debate
    if debate.turns:
        out.append(
            f"DEBATE  {debate.rounds_completed} round(s), "
            f"stopped: {debate.stop_reason}"
        )
        for turn in debate.turns:
            flag = "" if turn.new_information else "  [no new information]"
            out.append(
                f"  r{turn.round} {turn.speaker:<5} "
                f"{_flat(turn.claim)[:CLAIM_WIDTH]}{flag}"
            )
        verdict = state.research_verdict
        if verdict is not None:
            out.append(
                f"  verdict: {verdict.winner} @ {verdict.confidence:.2f}"
                + (f" (converged: {verdict.converged_reason})"
                   if verdict.converged else "")
            )
        out.append("")

    risk = state.risk_debate
    if risk.turns:
        out.append(
            f"RISK COMMITTEE  {risk.rounds_completed} round(s), "
            f"stopped: {risk.stop_reason}"
        )
        for turn in risk.turns:
            out.append(
                f"  r{turn.round} {turn.speaker:<8} "
                f"{_flat(turn.claim)[:CLAIM_WIDTH]}"
            )
        rv = state.risk_verdict
        if rv is not None:
            out.append(
                f"  fund manager: {rv.decision.upper()} -> {rv.action} "
                f"{rv.target_weight_pct:.1f}% @ {rv.conviction:.2f}"
            )
            if rv.adjustment:
                out.append(f"    adjusted: {_flat(rv.adjustment)}")
        out.append("")

    decision = state.final_decision
    if decision is None:
        out.append("No decision was produced.")
        return "\n".join(out)

    out.append(
        f"DECISION  {decision.action}  {decision.target_weight_pct:.1f}% of equity"
        f"  conviction {decision.conviction:.2f}  horizon {decision.horizon_days}d"
    )
    out.append(f"  rationale    {_flat(decision.rationale)}")
    out.append(f"  invalidation {_flat(decision.invalidation)}")
    out.append(f"  dissent      {_flat(decision.dissent)}")
    out.append(f"  expires      {decision.expires_at:%Y-%m-%d %H:%M}")
    out.append("")

    if state.order_plan is not None:
        out.append(state.order_plan.render())
        if state.order_plan.blocked:
            out += [
                "",
                "  VETOED BY PYTHON. The fund manager approved this and the",
                "  compliance node blocked it -- which is the design (§9),",
                "  not a malfunction.",
            ]
        out.append("")

    if state.portfolio is not None:
        book = state.portfolio
        out.append(
            f"BOOK AT DECISION TIME  marked {book.as_of}  "
            f"equity {book.equity:,.0f} USD  {book.position_count} position(s)"
        )
        out.append("")

    out.append(_render_cost(state))

    if full and state.llm_calls:
        out += ["", "MODEL CALLS"]
        for call in state.llm_calls:
            out.append(
                f"  {call.node:<22} {call.model:<14} attempt {call.attempt}"
                f"  {call.latency_ms or 0:>6}ms"
                f"  {call.prompt_tokens or 0}->{call.completion_tokens or 0} tok"
                + ("  PARSE FAILED" if call.parse_failed else "")
            )

    if decision.degraded:
        out += [
            "",
            "DEGRADED -- this HOLD is a failure, not a judgement.",
            f"  {state.degraded_reason()}",
        ]

    return "\n".join(out)


def _render_cost(state: DecisionState) -> str:
    """Spend, without implying a budget cap that may not have been in force.

    The caps live in ``models.yaml``, not in the journal, so a replayed run
    cannot honestly print "13 of 20". Reading today's config and showing it
    next to a month-old run would assert something unknown.
    """
    attempts = len(state.llm_calls)
    repairs = sum(1 for r in state.llm_calls if r.parse_failed)
    wall = sum(r.latency_ms or 0 for r in state.llm_calls) / 1000
    spend = sum(r.usd for r in state.llm_calls)
    return (
        f"{attempts} model call(s), {repairs} repair turn(s), "
        f"{wall:.1f}s of inference, ${spend:.4f}"
    )


__all__ = [
    "CLAIM_WIDTH",
    "journal_files",
    "read_runs",
    "render_run",
    "render_summary",
]
