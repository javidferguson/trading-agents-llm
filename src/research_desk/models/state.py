"""``DecisionState`` and the records that accumulate inside it.

**This is the Stage 0 subset**, not the finished §4 schema. What is here is what
Stage 0 actually exercises, plus the two accumulating record types that must be
correct from the very first LLM call. Analyst reports, debate transcripts,
proposals and the final decision arrive in Stages 3–5.

Two things are load-bearing already and should not be treated as scaffolding:

* ``LLMCallRecord.raw_response`` -- ``mode="replay_llm"`` is worthless
  retroactively, so the raw text gets recorded from call number one.
* The **reducer annotations** on ``llm_calls`` and ``errors``. Nodes that run in
  a parallel fan-out append to both. Without an add-reducer LangGraph applies
  last-write-wins and the four analysts silently overwrite each other's records.
  This is the single nastiest failure mode in the design because nothing errors
  -- you just quietly lose three of four. ``tests/graph/test_reducers.py``
  asserts it.

> **Rule (architecture §1).** This JSONL is the source of truth for replay and
> audit. The LangGraph checkpointer exists *only* for crash-resume within a
> single live run. Two mechanisms, two jobs. Do not merge them, and do not let a
> future refactor "simplify" by deleting the JSONL.
"""

from __future__ import annotations

import operator
from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .modes import RunMode
from .orders import OrderPlan, Violation
from .portfolio import PortfolioSnapshot


class Degradable(BaseModel):
    """A schema that can always produce a safe, explicitly-marked fallback.

    Architecture §5: *"Any node error, budget exhaustion, or schema-parse
    failure after one repair turn -> FinalDecision(action='HOLD', ...). Never
    fail toward a trade."* ``structured()`` implements the mechanism; this base
    is where each schema says what its own safe answer looks like.

    Two fields do the work, and both are deliberately **not** shown to the
    model -- ``structured()`` strips them from the JSON schema it sends, so a
    model can never declare its own output degraded or, worse, un-degraded:

    * ``parse_failed`` -- a boolean that downstream nodes and the eval harness
      can filter on.
    * ``degraded_reason`` -- why, in words, so a run that quietly produced
      nothing useful says so in its own JSONL rather than looking like a
      confident neutral.

    The rule for ``_degraded_defaults`` is that the fallback must be the
    *least actionable* value available: neutral rather than bullish, zero
    confidence rather than average, empty evidence rather than invented. A
    degraded report that still reads as a mild recommendation is worse than a
    crash, because it is indistinguishable from a real one downstream.
    """

    parse_failed: bool = False
    degraded_reason: str | None = None

    @classmethod
    def _degraded_defaults(cls) -> dict[str, Any]:
        """The least actionable instance of this schema, as a field dict."""
        raise NotImplementedError(
            f"{cls.__name__} is Degradable but does not define "
            "_degraded_defaults(). Every schema reachable from structured() "
            "needs a safe fallback -- see architecture §5."
        )

    @classmethod
    def degraded(cls, reason: str, **overrides: Any) -> "Degradable":
        """Build the safe fallback, marked as such.

        ``overrides`` carries the fields the caller knows but the schema cannot
        default -- an ``AnalystReport`` still needs its ``kind``, because a
        degraded market report and a degraded news report are different facts.
        """
        values = {
            **cls._degraded_defaults(),
            **overrides,
            "parse_failed": True,
            "degraded_reason": reason,
        }
        return cls.model_validate(values)


class Evidence(BaseModel):
    """One cited fact. Short on purpose -- a quote, not an article."""

    source: str
    url: str | None = None
    as_of: datetime
    excerpt: str = Field(max_length=300)


class AnalystReport(Degradable):
    """One analyst's read. The §4 schema, and Stage 1's smoke-test target.

    Several constraints here are deliberately **not expressible in the JSON
    schema** handed to Ollama -- the 200-word summary cap most obviously, but
    also the 3-6 key points and the 0..1 confidence bound, which Ollama's
    grammar does not reliably enforce. That is a feature, not an oversight: it
    is what makes the repair turn (§6) a real code path exercised on ordinary
    runs rather than a branch that is written once and never taken.
    """

    kind: Literal["market", "news", "positioning", "fundamentals", "social"]
    stance: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0, le=1)
    summary: str
    key_points: list[str] = Field(min_length=3, max_length=6)
    evidence: list[Evidence] = Field(default_factory=list)
    #: What it could NOT see. Feeds calibration, and gives a small model an
    #: honest alternative to inventing the missing number.
    data_gaps: list[str] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def _summary_is_short(cls, value: str) -> str:
        words = len(value.split())
        if words > 200:
            # This message is pasted verbatim into the repair prompt, so it is
            # written as an instruction to a model, not as a diagnosis for a
            # human -- and deliberately NOT as a counting task.
            #
            # Measured: told "summary is 412 words; the limit is 200", qwen3:8b
            # came back with 304. It shortened and still failed, because
            # counting words is arithmetic and §2 says models cannot do
            # arithmetic -- a rule that applies to the constraints we hand them
            # just as much as to the numbers we ask them for. A structural
            # target ("at most 6 sentences") is something a model can actually
            # self-check while writing.
            raise ValueError(
                f"summary is too long at {words} words. Rewrite it as AT MOST "
                "6 short sentences, keeping the specific numbers and dropping "
                "the commentary."
            )
        return value

    @field_validator("key_points")
    @classmethod
    def _key_points_are_real(cls, value: list[str]) -> list[str]:
        """Reject blank or placeholder entries.

        Defence in depth. The padding that originally motivated this is now
        prevented upstream: `structured.schema_for` strips `minItems` before
        the schema reaches Ollama, precisely because enforcing a count in the
        grammar made a model satisfy it with junk -- observed returning
        `['real point', 'another', '\n\n']` and, on another run,
        `['...', '...', 'key_points_count']`.

        With the count stripped the model returns however many it has and this
        validator rarely fires. It stays because a blank string is still
        possible, and because an empty "key point" reaching a debate prompt is
        the kind of thing nobody notices until the transcript looks strange.
        """
        blank = [i for i, point in enumerate(value) if not point.strip()]
        if blank:
            raise ValueError(
                f"key_points entries {blank} are empty or whitespace. Every "
                "key point must be a real sentence. If you genuinely have "
                "fewer points to make, say less in each of the others rather "
                "than padding the list."
            )
        return value

    @classmethod
    def _degraded_defaults(cls) -> dict[str, Any]:
        return {
            # Neutral and zero-confidence: the least actionable report this
            # schema can express. Never bullish, never a usable confidence.
            "kind": "market",
            "stance": "neutral",
            "confidence": 0.0,
            "summary": "No usable report: the model did not return valid output.",
            "key_points": [
                "Analyst output could not be parsed.",
                "Treat this report as absent, not as neutral evidence.",
                "Downstream nodes should not weight this.",
            ],
            "evidence": [],
            "data_gaps": ["entire report unavailable"],
        }


class DebateTurn(BaseModel):
    """One speaker's contribution to one round (§4)."""

    round: int = Field(ge=1)
    speaker: str
    claim: str
    #: Which analyst reports this claim rests on. Named so the facilitator can
    #: see when two speakers are citing the same single report and calling it
    #: corroboration -- §15.2's failure in miniature.
    supporting_report_kinds: list[str] = Field(default_factory=list)
    rebuts: str | None = None

    #: Facilitator-verified, and it drives the novelty stop. Self-reported
    #: novelty would be worthless: a model restating itself believes it is
    #: making a new point.
    new_information: bool = False


class DebateTranscript(BaseModel):
    """Every turn, plus why the debate stopped (§4)."""

    turns: list[DebateTurn] = Field(default_factory=list)
    rounds_completed: int = 0
    stop_reason: Literal[
        "max_rounds", "converged", "no_novelty", "budget", "error"
    ] | None = None

    def turns_in(self, round_number: int) -> list[DebateTurn]:
        return [t for t in self.turns if t.round == round_number]

    def claims_by(self, speaker: str) -> list[str]:
        return [t.claim for t in self.turns if t.speaker == speaker]


class ResearchVerdict(Degradable):
    """The facilitator's call at the end of the research debate (§4)."""

    winner: Literal["bull", "bear", "tie"]
    thesis: str
    #: Required, and never dropped. §4 puts it here because this is where the
    #: paper's explainability claim cashes out: it becomes FinalDecision.dissent
    #: and surfaces in the confirmation prompt a human reads before approving.
    strongest_counterargument: str
    confidence: float = Field(ge=0, le=1)

    #: The facilitator's own read on whether another round would add anything.
    #: Advisory only -- conditions 1 and 4 are hard caps in Python, and §5 is
    #: explicit that the model never decides alone when to stop.
    converged: bool = False
    converged_reason: str = ""

    @classmethod
    def _degraded_defaults(cls) -> dict[str, Any]:
        return {
            "winner": "tie",
            "thesis": "No usable verdict: the facilitator returned nothing valid.",
            "strongest_counterargument":
                "No argument was recorded on either side.",
            "confidence": 0.0,
            "converged": False,
            "converged_reason": "",
        }


class RiskVerdict(Degradable):
    """The fund manager's decision (§3 node 13). Approve, adjust or veto.

    The last judgement before a human sees the order. Everything upstream was
    advice; this is the decision, and §4 requires it to carry its own dissent
    because that is the text the confirmation gate displays.
    """

    decision: Literal["approve", "adjust", "veto"]
    action: Literal["BUY", "SELL", "HOLD"]
    conviction: float = Field(ge=0, le=1)
    target_weight_pct: float = Field(ge=0, le=100)
    horizon_days: int = Field(ge=1, le=750)

    rationale: str
    #: The strongest SURVIVING argument against this decision. §4: this is
    #: where the paper's explainability claim cashes out.
    dissent: str
    invalidation: str
    #: What was changed and why. Empty when approved unchanged.
    adjustment: str = ""

    @field_validator("rationale", "dissent", "invalidation")
    @classmethod
    def _substantive(cls, value: str) -> str:
        if len(value.split()) < 4:
            raise ValueError(
                "too short to be useful -- write a full sentence naming the "
                "specific condition, not a word"
            )
        return value

    @model_validator(mode="after")
    def _veto_means_hold(self) -> "RiskVerdict":
        """A veto cannot carry a trade.

        Enforced in the schema rather than downstream because a vetoed BUY is
        the single most dangerous shape this object could take -- §5's
        "never fail toward a trade" applies doubly when the failure is a model
        contradicting itself.
        """
        if self.decision == "veto" and self.action != "HOLD":
            raise ValueError(
                f"decision is 'veto' but action is {self.action!r}. A veto means "
                "the trade is not made, so the action must be HOLD."
            )
        if self.decision == "veto" and self.target_weight_pct > 0:
            raise ValueError(
                "a vetoed proposal cannot carry a target weight above zero"
            )
        return self

    @classmethod
    def _degraded_defaults(cls) -> dict[str, Any]:
        return {
            "decision": "veto",
            "action": "HOLD",
            "conviction": 0.0,
            "target_weight_pct": 0.0,
            "horizon_days": 1,
            "rationale": "Degraded: the fund manager produced no usable decision.",
            "dissent": "No argument was recorded, so none survives.",
            "invalidation": "Not applicable; no position is being taken.",
            "adjustment": "",
        }


class TraderProposal(Degradable):
    """What the trader wants to do (§4).

    Note ``target_weight_pct``: **a desired percentage of equity, not a share
    count.** Share counts come from ``intent/sizing.py`` at Stage 6, which
    takes the minimum of four caps. A model that names a share count is doing
    position sizing in its head, which is arithmetic, which §2 forbids.
    """

    action: Literal["BUY", "SELL", "HOLD"]
    conviction: float = Field(ge=0, le=1)
    target_weight_pct: float = Field(ge=0, le=100)
    horizon_days: int = Field(ge=1, le=750)
    rationale: str
    #: "What would prove this wrong." Required, and it is half of what makes
    #: the confirmation prompt at Stage 7 worth reading.
    invalidation: str

    #: The strongest case AGAINST this proposal, in the trader's own words.
    #:
    #: At Stage 4 this comes from the bear researcher, and §4 is emphatic that
    #: dissent is "REQUIRED, never dropped". There is no bear researcher yet,
    #: so the trader has to argue against itself -- which is weaker, and far
    #: better than a FinalDecision carrying an empty `dissent` through to a
    #: human gate that was built to show it.
    strongest_counterargument: str

    #: Which theme or gap this closes. Blank until Stage 6 wires the drift
    #: table; the field exists now so the prompt can ask for it once it does.
    intent_alignment: str = ""

    @field_validator("rationale", "invalidation", "strongest_counterargument")
    @classmethod
    def _substantive(cls, value: str) -> str:
        """Reject a field that technically parses but says nothing.

        A one-word invalidation ("price") is indistinguishable from a missing
        one downstream, and the confirmation gate exists to show a human this
        text. Caught here so the repair turn asks for a real answer.
        """
        if len(value.split()) < 4:
            raise ValueError(
                "too short to be useful -- write a full sentence naming the "
                "specific condition, not a word"
            )
        return value

    @classmethod
    def _degraded_defaults(cls) -> dict[str, Any]:
        return {
            # HOLD, zero conviction, no position. §5: never fail toward a trade.
            "action": "HOLD",
            "conviction": 0.0,
            "target_weight_pct": 0.0,
            "horizon_days": 1,
            "rationale": "Degraded: the trader produced no usable proposal.",
            "invalidation": "Not applicable; no position is being taken.",
            "strongest_counterargument":
                "No analysis was produced, so no case either way exists.",
            "intent_alignment": "",
        }


class FinalDecision(BaseModel):
    """The artefact `execute` reads (§4). Broker-agnostic on purpose.

    ``FinalDecision -> OrderPlan -> ib_async Contract/Order`` (§9), so the
    first two layers run with no broker at all. Nothing here names a share
    count, a contract or an exchange.
    """

    schema_version: int = 1
    action: Literal["BUY", "SELL", "HOLD"]
    symbol: str
    conviction: float = Field(ge=0, le=1)
    target_weight_pct: float = Field(ge=0, le=100)

    #: Marketable limit, never market. With delayed data you are looking at a
    #: 15-minute-old price (§9).
    order_type: Literal["LMT", "MKT"] = "LMT"
    limit_offset_bps: int = 10
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    horizon_days: int

    rationale: str
    #: The preserved case against. §4 lists `dissent` and `invalidation` as
    #: REQUIRED because "this is where the paper's explainability claim
    #: actually cashes out -- both surface in the confirmation prompt."
    dissent: str
    invalidation: str

    #: A stale proposal cannot be executed. `render_decision()` at Stage 7
    #: refuses outright rather than prompting: a proposal generated after
    #: Tuesday's close must not be executable on Thursday.
    expires_at: datetime
    fund_manager_adjustment: str | None = None

    #: True when any node degraded. Carried so `execute` can see that a HOLD
    #: was a failure rather than a judgement.
    degraded: bool = False

    @classmethod
    def from_proposal(
        cls,
        proposal: "TraderProposal",
        symbol: str,
        *,
        expires_at: datetime,
        stop_loss_pct: float | None = None,
        degraded: bool = False,
    ) -> "FinalDecision":
        """Promote a trader proposal with no fund manager in between.

        **A Stage 3 shim.** At Stage 5 the fund manager approves, adjusts or
        vetoes and produces this itself; until then the promotion is
        deterministic so the vertical slice reaches `proposal.json`. It is
        deliberately a plain copy with no judgement of its own -- anything
        cleverer here would be a fund manager written by accident, in the wrong
        file, untested.
        """
        return cls(
            action=proposal.action,
            symbol=symbol.upper(),
            conviction=proposal.conviction,
            target_weight_pct=proposal.target_weight_pct,
            horizon_days=proposal.horizon_days,
            rationale=proposal.rationale,
            dissent=proposal.strongest_counterargument,
            invalidation=proposal.invalidation,
            stop_loss_pct=stop_loss_pct,
            expires_at=expires_at,
            fund_manager_adjustment=None,
            degraded=degraded or proposal.parse_failed,
        )


def _final_from_verdict(
    verdict: "RiskVerdict",
    symbol: str,
    *,
    expires_at: datetime,
    stop_loss_pct: float | None = None,
    degraded: bool = False,
) -> "FinalDecision":
    """Promote the fund manager's verdict. The Stage 5 replacement for the shim.

    At Stage 3 ``FinalDecision.from_proposal`` promoted the trader directly
    because there was no fund manager. There is one now, so the trader's
    proposal is advice and this is the decision -- which is the whole point of
    §3 node 13 and the reason `adjust` exists as a distinct outcome.
    """
    return FinalDecision(
        action=verdict.action,
        symbol=symbol.upper(),
        conviction=verdict.conviction,
        target_weight_pct=verdict.target_weight_pct,
        horizon_days=verdict.horizon_days,
        rationale=verdict.rationale,
        dissent=verdict.dissent,
        invalidation=verdict.invalidation,
        stop_loss_pct=stop_loss_pct,
        expires_at=expires_at,
        fund_manager_adjustment=verdict.adjustment or None,
        degraded=degraded or verdict.parse_failed,
    )


class LLMCallRecord(BaseModel):
    """One model call, recorded in full enough detail to replay it."""

    node: str
    profile: str
    provider: str
    model: str
    #: Provider-reported model digest. Two runs of "qwen3:8b" are not
    #: necessarily the same weights; the digest is what makes a cache key honest.
    model_digest: str | None = None
    prompt_hash: str
    #: The RAW response text, before parsing. This is what `replay_llm` feeds
    #: back in. Never replace it with the parsed object.
    raw_response: str
    #: Reasoning tokens, when the model emits them separately. qwen3 and other
    #: hybrid thinking models return these in their own field. Kept for the same
    #: reason as `raw_response`: it explains *why* a model said something and
    #: cannot be recovered later. Not part of replay -- `raw_response` alone
    #: reproduces the parse.
    thinking: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    usd: float = 0.0
    #: 1 on the first try, 2 on the repair turn (architecture §6).
    attempt: int = 1
    parse_failed: bool = False
    at: datetime = Field(default_factory=datetime.now)


class NodeError(BaseModel):
    """A node that failed. Recorded rather than raised, so the run degrades.

    Failure direction is always HOLD (architecture §5) -- never fail toward a
    trade.
    """

    node: str
    kind: str
    message: str
    at: datetime = Field(default_factory=datetime.now)


class DecisionState(BaseModel):
    """The graph state. Used directly as the LangGraph state type.

    Fields that accumulate need reducer annotations; everything else is
    last-write-wins.
    """

    schema_version: int = 1

    run_id: str
    symbol: str
    #: The decision date, and therefore the replay anchor. Every provider call
    #: made during this run is pinned to it.
    as_of: date
    created_at: datetime = Field(default_factory=datetime.now)
    mode: RunMode = RunMode.LIVE

    #: Hash of the resolved config and the prompt pack. Two runs that disagree
    #: on either are not comparable, and at Stage 8 that matters.
    config_hash: str = ""
    prompt_pack_version: str = "0"

    # --- accumulating: reducers are mandatory, see the module docstring -------
    llm_calls: Annotated[list[LLMCallRecord], operator.add] = Field(default_factory=list)
    errors: Annotated[list[NodeError], operator.add] = Field(default_factory=list)

    #: Free-form breadcrumbs. Present from Stage 0 so the toy graph has
    #: something to write; harmless to keep.
    notes: Annotated[list[str], operator.add] = Field(default_factory=list)

    #: The deterministic prefetch every LLM node reads (§2). Typed as Any to
    #: keep models/state.py free of a market-data import cycle.
    snapshot: Any | None = None

    #: Written by FOUR PARALLEL NODES, one key each, so it needs a reducer
    #: exactly like the lists below. `operator.or_` is dict merge.
    #:
    #: §4 warns that parallel appends "will silently overwrite each other"
    #: without this -- and the list fields were annotated at Stage 0 for that
    #: reason. This dict was not, and the fan-out failed on the first live run
    #: with "At key 'analyst_reports': Can receive only one value per step."
    #:
    #: Worth recording: LangGraph 1.2 RAISED rather than silently keeping one
    #: of four. §4 was written against the silent behaviour, so the trap is
    #: now loud -- but only for fields it can see are concurrent, which is why
    #: the annotation is still the actual fix.
    analyst_reports: Annotated[dict[str, AnalystReport], operator.or_] = Field(
        default_factory=dict
    )

    #: Accumulating across rounds, so it needs a reducer like llm_calls. The
    #: bull and bear both append within one round.
    debate_turns: Annotated[list[DebateTurn], operator.add] = Field(default_factory=list)
    research_debate: DebateTranscript = Field(default_factory=DebateTranscript)
    research_verdict: ResearchVerdict | None = None

    trader_proposal: TraderProposal | None = None

    #: Appended to by the risk trio, so it needs a reducer like debate_turns.
    risk_turns: Annotated[list[DebateTurn], operator.add] = Field(default_factory=list)
    risk_debate: DebateTranscript = Field(default_factory=DebateTranscript)
    risk_verdict: RiskVerdict | None = None

    final_decision: FinalDecision | None = None

    # --- Stage 6: intent, portfolio, sizing, the veto ------------------------

    #: The book sizing ran against.
    #:
    #: Properly typed, unlike ``snapshot`` and ``drift``, because it costs
    #: nothing: ``models/portfolio.py`` imports nothing from this package, so
    #: there is no cycle to avoid. The payoff is that a run read back out of the
    #: journal has a real ``PortfolioSnapshot`` with its derived ``equity`` and
    #: ``weight_pct`` rather than a bare dict -- ``desk review`` tripped over
    #: exactly that on its first run.
    #:
    #: Recorded in full rather than as a reference to ``data/portfolio.yaml``,
    #: because that file is overwritten every time the book is re-marked and a
    #: replay must see the weights the decision was actually made against.
    portfolio: PortfolioSnapshot | None = None

    #: The drift table the trader was given (``intent.engine.DriftTable``).
    #: Channel 2's output, kept so that at Stage 8 a decision can be read
    #: against the gaps that were open when it was made -- "did it close the
    #: gap it said it was closing" is not answerable without it.
    drift: Any | None = None

    #: Channel 4's output: the sized, checked order. Present for every run,
    #: including HOLDs and blocked orders -- the reason no order was placed is
    #: the most useful field in the file.
    order_plan: OrderPlan | None = None

    # --- arriving in later stages --------------------------------------------
    # Stage 9: memory_hits, regime

    def degraded_reason(self) -> str | None:
        """Why this run should fail toward HOLD, or ``None`` if it is healthy."""
        if not self.errors:
            return None
        return "; ".join(f"{e.node}: {e.kind}" for e in self.errors)


#: What a node returns: a partial-state patch, never a mutated state object.
#: Keeping node bodies to this signature is what makes them unit-testable,
#: replayable, and portable if LangGraph is ever un-adopted (architecture §1).
NodePatch = dict[str, Any]

__all__ = [
    "AnalystReport",
    "DebateTranscript",
    "DebateTurn",
    "FinalDecision",
    "ResearchVerdict",
    "RiskVerdict",
    "TraderProposal",
    "DecisionState",
    "Degradable",
    "Evidence",
    "LLMCallRecord",
    "NodeError",
    "NodePatch",
    "OrderPlan",
    "PortfolioSnapshot",
    "RunMode",
    "Violation",
]
