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

from pydantic import BaseModel, Field, field_validator

from .modes import RunMode


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
        """Reject blank entries.

        Not hypothetical. Ollama's constrained decode *does* enforce
        `minItems`, and when a model wants to emit fewer items than the schema
        demands it satisfies the count with padding -- observed returning
        `['...', '...', '\n\n']`. The grammar is happy; the report is not.
        Schema-valid is not the same as usable, and this is the clearest case
        of the difference.
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

    # --- arriving in later stages --------------------------------------------
    # Stage 2: intent, portfolio, snapshot, regime
    # Stage 3: analyst_reports, trader_proposal, final_decision
    # Stage 4: research_debate, research_verdict
    # Stage 5: risk_debate
    # Stage 6: violations
    # Stage 9: memory_hits

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
    "DecisionState",
    "Degradable",
    "Evidence",
    "LLMCallRecord",
    "NodeError",
    "NodePatch",
    "RunMode",
]
