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
ANALYST_NODES = ("market_analyst",)


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


@pytest.mark.parametrize("node", ANALYST_NODES)
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


@pytest.mark.parametrize("node", ANALYST_NODES)
def test_no_analyst_node_mentions_portfolio_intent_at_all(node: str) -> None:
    source = (SRC / "graph" / "nodes" / f"{node}.py").read_text()
    assert "portfolio-intent.yaml" not in source
