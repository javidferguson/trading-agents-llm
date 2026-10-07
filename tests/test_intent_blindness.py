"""§8's rule, enforced: **analysts never see portfolio intent.**

> *"Do not tell the fundamentals analyst 'we are bullish on AI
> infrastructure' before it reads the 10-K. The entire value of a bear
> researcher evaporates if every upstream report was primed with your thesis.
> Intent enters at the Trader node and no earlier. This deserves a code
> comment so nobody 'helpfully' fixes it later."*

The plan also asks for exactly this test: *"ideally a test that asserts intent
text is absent from analyst prompts."* It is the rule most likely to be broken
by a well-meaning refactor -- passing the intent everywhere looks like
consistency -- and the damage is invisible, because a primed analyst produces a
confident report that merely agrees with you.
"""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import pytest

from research_desk.config import Settings, load_yaml
from research_desk.context import NodeContext
from research_desk.llm.router import LLMResponse, LLMRouter, ModelProfile
from research_desk.metrics.render import render_intent, render_snapshot
from research_desk.models.market import MarketSnapshot
from research_desk.models.state import DecisionState

SRC = Path(__file__).resolve().parents[1] / "src" / "research_desk"
#: Every module that defines an analyst. `analysts.py` is the factory for all
#: four, which is precisely why they are one factory and not four near-copies:
#: this guard covers the whole family by covering one file.
ANALYST_MODULES = ("market_analyst", "analysts")

#: Nodes that ARE allowed to see intent, per §8: *"Objective description,
#: theme theses and free-text constraints go into the Trader and Fund
#: Manager system prompts only."* Listed so the positive control below can
#: assert the channel actually works -- a blindness test that passes
#: because nothing anywhere receives intent would be worthless.
INTENT_SIGHTED_NODES = ("trader", "risk")

#: Every analyst kind, checked live against what it actually sent.
ANALYST_KINDS = ("market", "news", "positioning", "fundamentals")


class Recorder:
    """Captures what a node actually sent, rather than what we think it sent."""

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    async def complete(self, profile, messages, schema=None):
        self.messages = [dict(m) for m in messages]
        return LLMResponse(
            text='{"kind":"market","stance":"neutral","confidence":0.5,'
                 '"summary":"Nothing notable.","key_points":["a","b","c"],'
                 '"evidence":[],"data_gaps":[]}',
            model="fake", profile=profile.name, provider=profile.provider,
        )

    async def aclose(self):
        return None


def snapshot() -> MarketSnapshot:
    return MarketSnapshot(symbol="MSFT", as_of=date(2026, 10, 6))


# --------------------------------------------------------------------------- #
# What the analyst actually receives
# --------------------------------------------------------------------------- #


async def test_the_analyst_prompt_contains_no_intent() -> None:
    """The live check: inspect the messages the node really sent."""
    from research_desk.graph.nodes.market_analyst import market_analyst

    recorder = Recorder()
    router = LLMRouter(
        profiles={"quick": ModelProfile(name="quick", provider="fake", model="m")},
        node_routes={"market_analyst": "quick"},
        clients={"fake": recorder},
    )
    ctx = NodeContext(settings=Settings(), as_of=date(2026, 10, 6),
                      extras={"router": router})
    state = DecisionState(run_id="t", symbol="MSFT", as_of=date(2026, 10, 6),
                          snapshot=snapshot())

    await market_analyst(state, ctx)
    sent = "\n".join(m["content"] for m in recorder.messages).lower()

    intent = load_yaml("portfolio-intent.yaml")

    # Every theme name and thesis must be absent.
    for theme in intent["themes"]:
        assert theme["name"].lower() not in sent, (
            f"theme {theme['name']!r} reached the analyst -- §8 is violated and "
            "the bear researcher is now arguing against a primed report"
        )
        # Four-word phrases, not single words. Single-word matching
        # false-positived on "compute": the analyst prompt legitimately says
        # the facts were "computed in Python", and the AI-infrastructure
        # thesis opens "Compute demand for training...". A real leak carries a
        # distinctive phrase; shared English does not.
        words = theme["thesis"].split()
        for start in range(max(len(words) - 3, 1)):
            phrase = " ".join(words[start:start + 4]).lower().strip(".,")
            if len(phrase) > 20:
                assert phrase not in sent, (
                    f"thesis phrase {phrase!r} reached the analyst"
                )

    # And every free-text constraint.
    for constraint in intent.get("constraints", []):
        snippet = " ".join(str(constraint).split()[:5]).lower()
        assert snippet not in sent

    for banned in ("portfolio intent", "target_weight", "max_position",
                   "conviction 0.", "exemplars"):
        assert banned not in sent, f"{banned!r} reached the analyst"


@pytest.mark.parametrize("kind", ANALYST_KINDS)
async def test_no_analyst_in_the_fanout_sees_intent(kind: str) -> None:
    """All four, live. Stage 4 added three analysts to a guard that covered one.

    Each is checked against what it really sent, not against what the factory
    looks like it sends -- the factory is shared, so a leak would hit all four
    at once and this is the test that would say so.
    """
    from research_desk.graph.nodes.analysts import make_analyst

    recorder = Recorder()
    node_name = {"market": "market_analyst", "news": "news_analyst",
                 "positioning": "positioning_analyst",
                 "fundamentals": "fundamentals_analyst"}[kind]
    router = LLMRouter(
        profiles={"quick": ModelProfile(name="quick", provider="fake", model="m")},
        node_routes={node_name: "quick"},
        clients={"fake": recorder},
    )
    ctx = NodeContext(settings=Settings(), as_of=date(2026, 10, 6),
                      extras={"router": router})

    # A snapshot with facts in this analyst's own slice, so the node does not
    # short-circuit on an empty slice and skip the call entirely.
    from research_desk.metrics.render import ANALYST_BLOCKS
    snap = MarketSnapshot(symbol="MSFT", as_of=date(2026, 10, 6))
    block_name = ANALYST_BLOCKS[kind][0]
    block_cls = type(getattr(snap, block_name))
    field = next(f for f in block_cls.model_fields if f != "gaps")
    populated = block_cls(**{field: 1.0}, gaps={
        f: "test" for f in block_cls.model_fields if f not in ("gaps", field)
    })
    snap = snap.model_copy(update={block_name: populated})

    await make_analyst(kind)(
        DecisionState(run_id="t", symbol="MSFT", as_of=date(2026, 10, 6),
                      snapshot=snap),
        ctx,
    )
    assert recorder.messages, f"{kind} analyst made no call; the test proved nothing"
    sent = "\n".join(m["content"] for m in recorder.messages).lower()

    intent = load_yaml("portfolio-intent.yaml")
    for theme in intent["themes"]:
        assert theme["name"].lower() not in sent, (
            f"theme {theme['name']!r} reached the {kind} analyst -- §8 violated"
        )
    for banned in ("portfolio intent", "target_weight", "max_position", "exemplars"):
        assert banned not in sent, f"{banned!r} reached the {kind} analyst"


@pytest.mark.parametrize("kind", ANALYST_KINDS)
def test_each_analyst_sees_a_disjoint_slice(kind: str) -> None:
    """§15.2: *"Three agents agreeing is not three pieces of evidence -- they
    read the same reports."* Disjoint slices are that rule enforced."""
    from research_desk.metrics.render import ANALYST_BLOCKS

    mine = set(ANALYST_BLOCKS[kind])
    for other, blocks in ANALYST_BLOCKS.items():
        if other == kind:
            continue
        assert not (mine & set(blocks)), (
            f"{kind} and {other} both read {mine & set(blocks)}; their reports "
            "would be one observation counted twice"
        )


def test_the_slices_cover_every_metric_block() -> None:
    """Nothing computed should be invisible to every analyst."""
    from research_desk.metrics.render import ANALYST_BLOCKS
    from research_desk.models.market import BLOCK_NAMES

    seen = {b for blocks in ANALYST_BLOCKS.values() for b in blocks}
    assert seen == set(BLOCK_NAMES), (
        f"blocks read by nobody: {set(BLOCK_NAMES) - seen}"
    )


async def test_the_analyst_prompt_does_contain_the_facts() -> None:
    """Guard against the test above passing because nothing was sent at all."""
    from research_desk.graph.nodes.market_analyst import market_analyst

    recorder = Recorder()
    router = LLMRouter(
        profiles={"quick": ModelProfile(name="quick", provider="fake", model="m")},
        node_routes={"market_analyst": "quick"},
        clients={"fake": recorder},
    )
    ctx = NodeContext(settings=Settings(), as_of=date(2026, 10, 6),
                      extras={"router": router})
    await market_analyst(
        DecisionState(run_id="t", symbol="MSFT", as_of=date(2026, 10, 6),
                      snapshot=snapshot()),
        ctx,
    )
    sent = "\n".join(m["content"] for m in recorder.messages)
    assert "MSFT" in sent and "2026-10-06" in sent
    assert "market analyst" in sent.lower()


# --------------------------------------------------------------------------- #
# The renderers are separate functions for this reason
# --------------------------------------------------------------------------- #


def test_render_snapshot_never_emits_intent() -> None:
    text = render_snapshot(snapshot())
    for banned in ("theme", "intent", "target_weight", "max_position", "benchmark:"):
        assert banned not in text.lower()


def test_render_intent_does_emit_intent() -> None:
    """The other half: giving intent to the trader has to actually work."""
    text = render_intent(load_yaml("portfolio-intent.yaml"))
    assert "PORTFOLIO INTENT" in text
    assert "AI infrastructure" in text


def test_a_theme_target_is_labelled_as_a_total_not_a_symbol_weight() -> None:
    """The failure this prevents, observed live.

    Reading the ten Stage 3 decisions, the trader proposed **35% of equity in
    AVGO** against a 10% hard cap, and said why in its own rationale: "the AI
    infrastructure theme is a key focus of the desk, with AVGO as a target at
    35.0% of equity." It had read the theme's total as the symbol's target.
    Four of ten proposals exceeded the cap that way.
    """
    text = render_intent(load_yaml("portfolio-intent.yaml"))
    assert "IN TOTAL" in text
    assert "NOT a target for any one symbol" in text
    assert "CEILING ON A SINGLE SYMBOL" in text
    # And the implied per-symbol figure is spelled out, because dividing is
    # arithmetic and §2 says models do not do arithmetic.
    assert "% each across" in text


# --------------------------------------------------------------------------- #
# Static: no analyst node may even import the intent renderer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("node", ANALYST_MODULES)
def test_no_analyst_node_imports_render_intent(node: str) -> None:
    """Catches the refactor before it runs.

    The live test above only covers nodes that exist. Stage 4 adds three more
    analysts, and this is the rule they will each be tempted to break.
    """
    path = SRC / "graph" / "nodes" / f"{node}.py"
    tree = ast.parse(path.read_text(), filename=str(path))

    for imported in ast.walk(tree):
        if isinstance(imported, ast.ImportFrom):
            names = {alias.name for alias in imported.names}
            assert "render_intent" not in names, (
                f"{node} imports render_intent. Analysts are intent-blind (§8); "
                "intent enters at the trader and no earlier."
            )
            assert "load_yaml" not in names or "portfolio" not in path.read_text(), (
                f"{node} may be reading portfolio-intent.yaml directly"
            )


@pytest.mark.parametrize("node", ANALYST_MODULES)
def test_no_analyst_node_mentions_portfolio_intent_at_all(node: str) -> None:
    source = (SRC / "graph" / "nodes" / f"{node}.py").read_text()
    assert "portfolio-intent.yaml" not in source


# --------------------------------------------------------------------------- #
# Stage 6: the drift table is intent too, and analysts must not see it either
# --------------------------------------------------------------------------- #


def drift_table():
    """The real drift table, from the real intent and the real seeded book."""
    from research_desk.intent.engine import compute_gaps, load_intent, load_portfolio

    return compute_gaps(load_intent(), load_portfolio(), as_of=date(2026, 10, 6))


@pytest.mark.parametrize("kind", ANALYST_KINDS)
async def test_no_analyst_sees_the_drift_table(kind: str) -> None:
    """**The Stage 6 regression this file exists for.**

    ``prefetch`` now puts the book and the drift table into ``DecisionState``,
    which is exactly the shape of accident §8 warns about: the data an analyst
    must not see is now sitting in the state object it reads. The analyst's path
    to facts is ``render_for_analyst``, which cannot reach it -- and this test
    is what would notice if that ever changed.

    An analyst that knows the book is overweight XOM is no longer reading XOM's
    fundamentals on their merits, and the bear researcher downstream is arguing
    against a report that was already primed.
    """
    from research_desk.graph.nodes.analysts import make_analyst
    from research_desk.metrics.render import ANALYST_BLOCKS

    recorder = Recorder()
    node_name = {"market": "market_analyst", "news": "news_analyst",
                 "positioning": "positioning_analyst",
                 "fundamentals": "fundamentals_analyst"}[kind]
    router = LLMRouter(
        profiles={"quick": ModelProfile(name="quick", provider="fake", model="m")},
        node_routes={node_name: "quick"},
        clients={"fake": recorder},
    )
    ctx = NodeContext(settings=Settings(), as_of=date(2026, 10, 6),
                      extras={"router": router})

    snap = MarketSnapshot(symbol="MSFT", as_of=date(2026, 10, 6))
    block_name = ANALYST_BLOCKS[kind][0]
    block_cls = type(getattr(snap, block_name))
    field = next(f for f in block_cls.model_fields if f != "gaps")
    populated = block_cls(**{field: 1.0}, gaps={
        f: "test" for f in block_cls.model_fields if f not in ("gaps", field)
    })
    snap = snap.model_copy(update={block_name: populated})

    from research_desk.intent.engine import load_portfolio

    drift = drift_table()
    state = DecisionState(
        run_id="t", symbol="MSFT", as_of=date(2026, 10, 6), snapshot=snap,
        # Both present in state, exactly as prefetch leaves them.
        drift=drift, portfolio=load_portfolio(),
    )

    await make_analyst(kind)(state, ctx)
    assert recorder.messages, f"{kind} analyst made no call; the test proved nothing"
    sent = "\n".join(m["content"] for m in recorder.messages).lower()

    for banned in ("portfolio drift", "gap usd", "free slot", "max_positions",
                   "gross exposure", "underweight", "overweight", "in band",
                   "equity"):
        assert banned not in sent, (
            f"{banned!r} reached the {kind} analyst -- the drift table has "
            "leaked into an intent-blind node (§8)"
        )

    # And no holding's actual weight, which is the specific number that would
    # let an analyst reason about the book rather than the symbol.
    book = load_portfolio()
    for held in book.positions:
        weight = f"{book.weight_pct(held.symbol):.2f}%"
        assert weight not in sent, (
            f"the book's {held.symbol} weight {weight} reached the {kind} analyst"
        )


@pytest.mark.parametrize("node", ANALYST_MODULES)
def test_no_analyst_node_imports_render_drift(node: str) -> None:
    """Catches the refactor before it runs, as with ``render_intent``."""
    path = SRC / "graph" / "nodes" / f"{node}.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    for imported in ast.walk(tree):
        if isinstance(imported, ast.ImportFrom):
            names = {alias.name for alias in imported.names}
            assert "render_drift" not in names, (
                f"{node} imports render_drift. The drift table is channel 2 of "
                "portfolio intent (§8) and reaches the trader and fund manager "
                "only."
            )


@pytest.mark.parametrize("node", ANALYST_MODULES)
def test_no_analyst_node_reaches_the_intent_package(node: str) -> None:
    """``intent/`` is channels 1, 2 and 4. None of them is an analyst's business."""
    path = SRC / "graph" / "nodes" / f"{node}.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    for imported in ast.walk(tree):
        if isinstance(imported, ast.ImportFrom) and imported.module:
            assert "intent" not in imported.module.split("."), (
                f"{node} imports from {imported.module}. Intent enters at the "
                "trader and no earlier (§8)."
            )


def test_render_drift_does_emit_the_book() -> None:
    """The positive control. A blindness test that passed because nothing
    anywhere rendered a drift table would be worthless."""
    from research_desk.metrics.render import render_drift

    text = render_drift(drift_table())
    assert "PORTFOLIO DRIFT" in text
    assert "gap USD" in text
    assert "free slot" in text


def test_render_drift_says_closing_a_gap_is_optional() -> None:
    """§8's claim is that the table converts "how much should we buy" into
    "close this gap or don't" -- the "or don't" has to be in the text."""
    from research_desk.metrics.render import render_drift

    text = render_drift(drift_table())
    assert "needs no action" in text
    assert "not an instruction" in text


def test_render_drift_uses_no_adjectives_about_the_book() -> None:
    """The §2 restraint the rest of ``render.py`` follows: percentiles and
    direction, never a judgement. ``overweight`` is a computed state; "badly
    overweight" or "dangerously concentrated" would be the renderer arguing."""
    from research_desk.metrics.render import render_drift

    text = render_drift(drift_table()).lower()
    for adjective in ("dangerous", "badly", "excessive", "too much", "risky",
                      "healthy", "comfortable", "should buy", "recommend"):
        assert adjective not in text


async def test_the_trader_does_receive_the_drift_table() -> None:
    """The other half of §8: channel 2 has to actually reach the trader.

    Both halves in one file on purpose. A test suite that only asserted the
    absence would be satisfied by deleting the feature.
    """
    from research_desk.graph.nodes.trader import trader

    class TraderRecorder(Recorder):
        async def complete(self, profile, messages, schema=None):
            self.messages = [dict(m) for m in messages]
            return LLMResponse(
                text='{"action":"HOLD","conviction":0.1,"target_weight_pct":0.0,'
                     '"horizon_days":30,"rationale":"Nothing compelling here today.",'
                     '"invalidation":"A break above the prior high would change this.",'
                     '"strongest_counterargument":"Momentum could continue regardless."}',
                model="fake", profile=profile.name, provider=profile.provider,
            )

    recorder = TraderRecorder()
    router = LLMRouter(
        profiles={"quick": ModelProfile(name="quick", provider="fake", model="m")},
        node_routes={"trader": "quick"},
        clients={"fake": recorder},
    )
    ctx = NodeContext(settings=Settings(), as_of=date(2026, 10, 6),
                      extras={"router": router})

    await trader(
        DecisionState(run_id="t", symbol="TSM", as_of=date(2026, 10, 6),
                      snapshot=snapshot(), drift=drift_table()),
        ctx,
    )
    sent = "\n".join(m["content"] for m in recorder.messages)
    assert "PORTFOLIO DRIFT" in sent
    assert "PORTFOLIO INTENT" in sent


async def test_the_trader_is_told_plainly_when_the_book_is_missing() -> None:
    """FOLLOWUPS.md recorded the trader *inferring* a position when it had no
    information. Silence is what produced that, so absence is stated."""
    from research_desk.graph.nodes.trader import trader

    class TraderRecorder(Recorder):
        async def complete(self, profile, messages, schema=None):
            self.messages = [dict(m) for m in messages]
            return LLMResponse(
                text='{"action":"HOLD","conviction":0.1,"target_weight_pct":0.0,'
                     '"horizon_days":30,"rationale":"Nothing compelling here today.",'
                     '"invalidation":"A break above the prior high would change this.",'
                     '"strongest_counterargument":"Momentum could continue regardless."}',
                model="fake", profile=profile.name, provider=profile.provider,
            )

    recorder = TraderRecorder()
    router = LLMRouter(
        profiles={"quick": ModelProfile(name="quick", provider="fake", model="m")},
        node_routes={"trader": "quick"},
        clients={"fake": recorder},
    )
    ctx = NodeContext(settings=Settings(), as_of=date(2026, 10, 6),
                      extras={"router": router})

    await trader(
        DecisionState(run_id="t", symbol="TSM", as_of=date(2026, 10, 6),
                      snapshot=snapshot(), drift=None),
        ctx,
    )
    sent = "\n".join(m["content"] for m in recorder.messages)
    assert "Do NOT infer a current position" in sent
