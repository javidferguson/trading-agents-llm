"""The research debate (§3 nodes 6-8) and its four termination conditions.

> *"Four conditions. **Budget always wins.**
>  1. ``rounds_completed >= cfg.max_rounds`` (default 1, not 2 -- start cheap)
>  2. Facilitator returns ``converged=True`` with a reason
>  3. **Novelty guard**: every turn this round has ``new_information=False``,
>     or a speaker's claim has >0.85 Jaccard overlap with its previous round
>  4. ``budget.exhausted()``
>  Conditions 1 and 4 are hard caps enforced in Python. Never let the model
>  alone decide when to stop."* -- §5

Condition 2 is the only one the model influences, and even then it is advice:
``should_continue`` checks 1 and 4 before it is consulted, so a facilitator
insisting on another round cannot get one it has no budget for.
"""

from __future__ import annotations

import logging
from typing import Any

from ...context import NodeContext
from ...llm.structured import structured
from ...metrics.novelty import DEFAULT_THRESHOLD, is_restatement
from ...models.state import (
    DebateTranscript,
    DebateTurn,
    DecisionState,
    NodeError,
    NodePatch,
    ResearchVerdict,
)
from ...prompts import load_prompt

logger = logging.getLogger(__name__)


def render_reports(state: DecisionState) -> str:
    """Every analyst report, for the researchers.

    The researchers DO see all four -- that is the point of a debate. The
    analysts did not see each other, which is what makes the four reports
    independent evidence rather than one opinion held four times (§15.2).
    """
    if not state.analyst_reports:
        return "ANALYST REPORTS: none were produced.\n"

    lines = ["ANALYST REPORTS", ""]
    for kind in ("market", "news", "positioning", "fundamentals"):
        report = state.analyst_reports.get(kind)
        if report is None:
            lines.append(f"  [{kind}] not run.")
            continue
        if report.parse_failed:
            lines.append(
                f"  [{kind}] DEGRADED -- {report.degraded_reason}. "
                "This is absent evidence, not neutral evidence; do not build on it."
            )
            lines.append("")
            continue
        lines.append(f"  [{kind}] {report.stance} @ {report.confidence:.2f}")
        lines.append(f"    {' '.join(report.summary.split())}")
        for point in report.key_points:
            lines.append(f"      - {' '.join(point.split())}")
        if report.data_gaps:
            lines.append(f"    could not see: {'; '.join(report.data_gaps)}")
        lines.append("")
    return "\n".join(lines)


def render_debate(transcript: DebateTranscript) -> str:
    """What has been argued so far."""
    if not transcript.turns:
        return "DEBATE SO FAR: nothing yet; you are opening.\n"
    lines = ["DEBATE SO FAR", ""]
    for turn in transcript.turns:
        rebut = f" (rebutting: {turn.rebuts})" if turn.rebuts else ""
        lines.append(f"  round {turn.round} -- {turn.speaker}{rebut}")
        lines.append(f"    {' '.join(turn.claim.split())}")
        if turn.supporting_report_kinds:
            lines.append(f"    resting on: {', '.join(turn.supporting_report_kinds)}")
        lines.append("")
    return "\n".join(lines)


def make_researcher(side: str):
    """Build the bull or bear researcher node."""
    node = f"{side}_researcher"

    async def researcher(state: DecisionState, ctx: NodeContext) -> NodePatch:
        router = ctx.extras.get("router")
        if router is None:
            return {"errors": [NodeError(node=node, kind="misconfigured",
                                         message="no LLM router on the context")]}

        budget = ctx.extras.get("budget")
        if budget is not None and (spent := budget.exhausted()):
            # Budget always wins, and it is checked before the call rather
            # than after -- a check after the fact is an audit, not a cap.
            return {"notes": [f"{node}: skipped, budget exhausted ({spent})"]}

        round_number = state.research_debate.rounds_completed + 1

        result = await structured(
            router, node, DebateTurn,
            [
                {"role": "system", "content": load_prompt(node)},
                {"role": "user", "content": (
                    f"{render_reports(state)}\n"
                    f"{render_debate(state.research_debate)}\n"
                    f"This is round {round_number}. Make your argument about "
                    f"{state.symbol}."
                )},
            ],
            degraded_fields={"round": round_number, "speaker": side},
        )
        if budget is not None:
            budget.record(calls=len(result.records), usd=result.usd)

        turn = result.value
        # The speaker and round are ours to set, not the model's: a model that
        # mislabels its own round corrupts the novelty comparison.
        turn = turn.model_copy(update={"round": round_number, "speaker": side})

        return {
            "debate_turns": [turn],
            "llm_calls": result.records,
            "notes": [f"{node} r{round_number}: {turn.claim[:70]}"],
        }

    researcher.__name__ = node
    return researcher


async def research_facilitator(state: DecisionState, ctx: NodeContext) -> NodePatch:
    """Judge the round, verify novelty, and emit a verdict (§3 node 8)."""
    node = "research_facilitator"
    router = ctx.extras.get("router")
    if router is None:
        return {"errors": [NodeError(node=node, kind="misconfigured",
                                     message="no LLM router on the context")]}

    round_number = state.research_debate.rounds_completed + 1
    this_round = [t for t in state.debate_turns if t.round == round_number]

    budget = ctx.extras.get("budget")
    if budget is not None and (spent := budget.exhausted()):
        return _close(state, this_round, round_number, "budget",
                      verdict=None, note=f"{node}: skipped, budget exhausted ({spent})")

    result = await structured(
        router, node, ResearchVerdict,
        [
            {"role": "system", "content": load_prompt(node)},
            {"role": "user", "content": (
                f"{render_reports(state)}\n"
                f"{render_debate(_transcript_with(state, this_round))}\n"
                f"Round {round_number} is complete. Give your verdict on "
                f"{state.symbol}."
            )},
        ],
    )
    if budget is not None:
        budget.record(calls=len(result.records), usd=result.usd)

    verdict = result.value
    scored = _score_novelty(state, this_round)
    stop = _stop_reason(state, ctx, scored, verdict, round_number)

    patch = _close(state, scored, round_number, stop, verdict=verdict,
                   note=f"{node} r{round_number}: {verdict.winner} "
                        f"@ {verdict.confidence:.2f}"
                        + (f", stopping ({stop})" if stop else ", continuing"))
    patch["llm_calls"] = result.records
    if result.degraded:
        patch["errors"] = [NodeError(node=node, kind="degraded",
                                     message=verdict.degraded_reason or "unknown")]
    return patch


def _transcript_with(state: DecisionState, turns: list[DebateTurn]) -> DebateTranscript:
    return DebateTranscript(
        turns=[*state.research_debate.turns, *turns],
        rounds_completed=state.research_debate.rounds_completed,
    )


def _score_novelty(
    state: DecisionState,
    turns: list[DebateTurn],
    *,
    transcript: DebateTranscript | None = None,
) -> list[DebateTurn]:
    """Overwrite ``new_information`` with the Jaccard check where it disagrees.

    Two signals, deliberately. The facilitator catches a speaker who changes
    the words and not the point; the overlap score catches a facilitator too
    agreeable to call a repeat a repeat. Either one saying "restatement" is
    enough, because a false continue costs a round and a false stop costs
    nothing we can measure.
    """
    history = transcript if transcript is not None else state.research_debate
    threshold = DEFAULT_THRESHOLD
    scored: list[DebateTurn] = []
    for turn in turns:
        earlier = [
            t.claim for t in history.turns
            if t.speaker == turn.speaker and t.round < turn.round
        ]

        # AN OPENING ARGUMENT IS NEW BY DEFINITION. Nothing can restate
        # nothing, so round 1 is forced true whatever the facilitator said.
        #
        # Observed live: told to be strict about novelty, the facilitator
        # marked BOTH round-1 turns `new_information=False`. With
        # max_research_rounds at 1 that was harmless -- condition 1 stopped
        # the debate first -- but at 2 it would have fired `no_novelty` before
        # either side had spoken twice, ending a debate that had not started.
        # A latent bug that only appears when someone raises a config value.
        if not earlier:
            scored.append(turn.model_copy(update={"new_information": True}))
            continue

        repeated, score = is_restatement(turn.claim, earlier, threshold)
        if repeated and turn.new_information:
            logger.info(
                "%s r%d: facilitator called this new, but it overlaps a previous "
                "claim at %.2f -- treating as a restatement",
                turn.speaker, turn.round, score,
            )
        scored.append(turn.model_copy(
            update={"new_information": turn.new_information and not repeated}
        ))
    return scored


def _stop_reason(
    state: DecisionState,
    ctx: NodeContext,
    turns: list[DebateTurn],
    verdict: ResearchVerdict | None,
    round_number: int,
    *,
    transcript: DebateTranscript | None = None,
    rounds_key: str = "max_research_rounds",
) -> str | None:
    """The four conditions of §5, in priority order. ``None`` means continue."""
    budget = ctx.extras.get("budget")

    # 4 first: budget always wins, so it is checked before anything the model
    # had an opinion about.
    if budget is not None and budget.exhausted():
        return "budget"

    # 1: a hard cap in Python.
    max_rounds = int(ctx.extras.get(rounds_key, 1))
    if round_number >= max_rounds:
        return "max_rounds"

    # 3: the novelty guard. Nobody said anything new this round.
    if turns and not any(t.new_information for t in turns):
        return "no_novelty"

    # 2: the only one the model influences, and only after the caps. The
    # risk committee has no convergence signal of its own -- the fund manager
    # decides rather than reporting agreement -- so it passes None.
    if verdict is not None and verdict.converged:
        return "converged"

    return None


def _close(
    state: DecisionState,
    turns: list[DebateTurn],
    round_number: int,
    stop: str | None,
    *,
    verdict: ResearchVerdict | None,
    note: str,
) -> NodePatch:
    transcript = DebateTranscript(
        turns=[*state.research_debate.turns, *turns],
        rounds_completed=round_number,
        stop_reason=stop,
    )
    patch: NodePatch = {"research_debate": transcript, "notes": [note]}
    if verdict is not None:
        patch["research_verdict"] = verdict
    return patch
