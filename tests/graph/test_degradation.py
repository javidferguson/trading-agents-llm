"""**An absent input is not a broken pipeline.** The ``NodeError.severity`` split.

The rule here used to be one line -- ``degraded = bool(state.errors)`` -- and it
was faithful to §5, which says *"**Any** node error... -> HOLD"*. What §5 did not
anticipate is that "an analyst had nothing to read" is recorded as a node error,
so over the first 76 runs **31 of the 32 degraded runs had no problem other than
a missing news analyst**, and an approved BUY was rewritten to a degraded HOLD
every time. There was never a real machinery failure.

What these tests pin down, in both directions:

* a symbol with no news history produces a real decision, with the gap recorded
* a genuine failure -- a fatal error, or coverage collapsing -- still goes to
  HOLD, because the point was never to soften §5
* an unrecognised ``kind`` is **fatal**, so a future emit site that does not
  stop to think about this fails safe
* the absence stays visible in all three renderers, because the old behaviour at
  least had the virtue of being loud
"""

from __future__ import annotations

from datetime import date

from research_desk.models.state import (
    AnalystReport,
    DecisionState,
    NodeError,
)

AS_OF = date(2026, 10, 8)


def usable(kind: str) -> AnalystReport:
    return AnalystReport(
        kind=kind, stance="bullish", confidence=0.6,
        summary="momentum has improved against the sector since August",
        key_points=["price above the 200-day", "volume expanding", "sector lagging"],
    )


def absent(kind: str) -> AnalystReport:
    return AnalystReport.degraded(f"no {kind} data was available", kind=kind)


def state(reports: dict | None = None, errors: list | None = None) -> DecisionState:
    return DecisionState(
        run_id="t", symbol="AMD", as_of=AS_OF,
        analyst_reports=reports or {},
        errors=errors or [],
    )


def full(**overrides: AnalystReport) -> dict:
    """All four analysts usable, with named ones replaced."""
    reports = {k: usable(k) for k in
               ("market", "news", "positioning", "fundamentals")}
    reports.update(overrides)
    return reports


# --------------------------------------------------------------------------- #
# Severity itself
# --------------------------------------------------------------------------- #


def test_an_unrecognised_kind_is_fatal() -> None:
    """The fail-safe property, asserted directly rather than inferred.

    A new error site added in six months' time gets §5's behaviour by default.
    Marking something partial has to be a deliberate act, in the one place that
    knows whether an input was missing or the machinery broke.
    """
    error = NodeError(node="some_future_node", kind="who_knows", message="?")
    assert error.severity == "fatal"
    assert error.is_fatal
    assert state(full(), [error]).is_degraded()


def test_every_fatal_kind_the_codebase_emits_still_degrades() -> None:
    """The backward-compatibility claim. These are the kinds in the code today,
    and every one of them keeps its old meaning with no edit at its emit site."""
    for kind in ("misconfigured", "budget", "no_snapshot", "no_book", "degraded"):
        s = state(full(), [NodeError(node="n", kind=kind, message="m")])
        assert s.is_degraded(), kind
        assert kind in (s.degraded_reason() or "")


def test_prefetch_no_data_is_fatal_although_it_shares_a_kind() -> None:
    """``no_data`` means two different things in two places, which is exactly
    why severity cannot be derived from ``kind``. Prefetch failing means there
    are no prices at all; an empty analyst slice means one input was thin."""
    fatal = state(full(), [NodeError(node="prefetch", kind="no_data",
                                     message="the registry raised")])
    partial = state(full(news=absent("news")),
                    [NodeError(node="news_analyst", kind="no_data",
                               message="no news data", severity="partial")])
    assert fatal.is_degraded()
    assert not partial.is_degraded()


# --------------------------------------------------------------------------- #
# The coverage floor, so a real collapse still voids the run
# --------------------------------------------------------------------------- #


def test_one_missing_analyst_does_not_degrade_the_run() -> None:
    """The case that prompted all of this: 28 of 32 symbols have no GDELT
    history, and a sweep that reports 28 degraded HOLDs reports nothing."""
    s = state(full(news=absent("news")),
              [NodeError(node="news_analyst", kind="no_data",
                         message="no news data", severity="partial")])
    assert not s.is_degraded()
    assert s.absent_analysts() == ["news"]


def test_two_of_four_is_the_floor_and_passes() -> None:
    """Half, not a majority -- measured against the journal, where 7 of 76 runs
    had exactly two analysts and none had fewer."""
    s = state(full(news=absent("news"), positioning=absent("positioning")))
    assert not s.is_degraded()


def test_one_of_four_degrades() -> None:
    """A full fan-out that collapsed to a single analyst did not produce a
    narrower decision, it produced nothing to decide from."""
    s = state(full(news=absent("news"), positioning=absent("positioning"),
                   fundamentals=absent("fundamentals")))
    assert s.is_degraded()
    assert "1 of 4" in (s.degraded_reason() or "")


def test_the_market_analyst_is_required_when_it_ran() -> None:
    """It is the only slice built from prices. Without it the others are
    commentary on a number nobody has."""
    s = state(full(market=absent("market")))
    assert s.is_degraded()
    assert "market analyst" in (s.degraded_reason() or "")


def test_a_single_analyst_graph_is_not_degraded() -> None:
    """``--slice`` runs one analyst. Counting against four would make the
    shortest graph permanently degraded; the floor counts what ran."""
    assert not state({"market": usable("market")}).is_degraded()


def test_a_graph_with_no_analysts_is_not_degraded() -> None:
    """The Stage 0 linear and fan-out harnesses have none. An analyst that
    actually failed to run leaves a fatal ``misconfigured`` behind instead, so
    an empty dict is a statement about the graph, not about the data."""
    assert not state({}).is_degraded()


# --------------------------------------------------------------------------- #
# The HOLD rationale must not cite a cause that did not stop anything
# --------------------------------------------------------------------------- #


def test_degraded_reason_is_none_when_only_partial_errors_exist() -> None:
    """This string is written into the HOLD rationale. A run reading
    "degraded: news_analyst: no_data" while carrying an actionable BUY is a
    contradiction the human at the gate cannot resolve."""
    s = state(full(news=absent("news")),
              [NodeError(node="news_analyst", kind="no_data",
                         message="no news data", severity="partial")])
    assert s.degraded_reason() is None


def test_a_fatal_reason_does_not_mention_the_partial_one() -> None:
    s = state(full(news=absent("news")), [
        NodeError(node="news_analyst", kind="no_data", message="no news",
                  severity="partial"),
        NodeError(node="trader", kind="budget", message="exhausted"),
    ])
    reason = s.degraded_reason() or ""
    assert "trader: budget" in reason
    assert "news_analyst" not in reason


def test_a_parse_failure_is_reported_as_absent_too() -> None:
    """``absent_analysts`` keys on ``parse_failed``, so a report the model
    mangled counts as absent evidence -- which it is. The error it carries is
    ``kind="degraded"`` and stays fatal, so the run still holds; the field is
    about what was readable, not about what stopped the run."""
    s = state(full(fundamentals=absent("fundamentals")))
    assert s.absent_analysts() == ["fundamentals"]


# --------------------------------------------------------------------------- #
# persist answers the question itself, for the graphs with no compliance node
# --------------------------------------------------------------------------- #


async def test_persist_applies_the_same_rule_when_it_promotes_alone(tmp_path) -> None:
    """``persist`` cannot defer to ``compliance``: the Stage 0 harness shapes
    have no compliance node, so it promotes the proposal itself. Both nodes must
    reach the same answer -- which is why the rule lives on ``DecisionState``
    rather than being written out twice.
    """
    from research_desk.config import Settings
    from research_desk.context import NodeContext
    from research_desk.graph.nodes.persist import persist
    from research_desk.models.state import TraderProposal

    proposal = TraderProposal(
        action="BUY", conviction=0.8, target_weight_pct=5.0, horizon_days=60,
        rationale="relative strength is improving against the sector",
        invalidation="a close below the 200-day moving average would end this",
        strongest_counterargument="twelve-month momentum is still negative",
    )
    context = NodeContext(settings=Settings(data_dir=tmp_path), as_of=AS_OF,
                          extras={"registry": None})
    patch = await persist(
        DecisionState(
            run_id="t", symbol="AMD", as_of=AS_OF, trader_proposal=proposal,
            analyst_reports=full(news=absent("news")),
            errors=[NodeError(node="news_analyst", kind="no_data",
                              message="no news data", severity="partial")],
        ),
        context,
    )
    decision = patch["final_decision"]
    assert decision.action == "BUY"
    assert not decision.degraded
    assert decision.absent_analysts == ["news"]
