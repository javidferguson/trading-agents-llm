"""Routing: node -> profile -> provider, and the presets Stage 8 flips."""

from __future__ import annotations

import pytest

from research_desk.config import Settings, load_yaml
from research_desk.llm.router import (
    LLMRouter,
    ModelProfile,
    ProfileNotConfigured,
)


def router_from_shipped_config(preset: str | None = None) -> LLMRouter:
    return LLMRouter.from_config(Settings(), config=load_yaml("models.yaml"), preset=preset)


def test_the_shipped_config_loads() -> None:
    router = router_from_shipped_config()
    assert {"quick", "deep_local", "deep_hosted"} <= set(router.profiles)


def test_every_routed_node_names_a_profile_that_exists() -> None:
    """A typo here surfaces as a crash 11 nodes into an 8-minute run."""
    router = router_from_shipped_config()
    for node in router.node_routes:
        assert router.profile_for(node) is not None


def test_nothing_is_routed_to_the_hosted_provider() -> None:
    """Revised at Stage 5: the fund manager runs locally for now.

    §6 calls it "the one node worth paying for" and routed it to
    `deep_hosted`. It now points at `judge_local` -- a quantized qwen3:32b
    with thinking enabled -- so a full run costs nothing and needs no API key.

    This asserts the direction of the change rather than just the current
    value: NOTHING may be routed to the hosted provider while that is the
    intent, because a single node quietly re-routed is a bill nobody expected.
    `test_the_hosted_path_stays_available` covers the other half.
    """
    router = router_from_shipped_config()
    hosted = [
        node for node in router.node_routes
        if router.profile_for(node).provider == "anthropic"
    ]
    assert hosted == [], (
        f"{hosted} route to the hosted provider. Stage 5 runs fully local; "
        "use the all_hosted preset to spend money deliberately."
    )


def test_the_hosted_path_stays_available_for_the_ablation() -> None:
    """§11's `trader->deep_hosted` ablation needs the path to exist.

    Deleting `deep_hosted` would be simpler and would discard the question
    "is the one paid node worth paying for" -- which the design treats as
    important enough to make a named ablation.
    """
    router = router_from_shipped_config()
    assert "deep_hosted" in router.profiles
    assert router.profiles["deep_hosted"].provider == "anthropic"

    hosted_preset = router_from_shipped_config(preset="all_hosted")
    assert hosted_preset.profile_for("fund_manager").provider == "anthropic"


def test_the_fund_manager_gets_its_own_model() -> None:
    """It must not share weights with another node.

    A fund manager running the same model as the trader is not a second
    opinion, and §3 node 13 exists to be a distinct judgement.
    """
    router = router_from_shipped_config()
    judge = router.profile_for("fund_manager")
    for other in ("trader", "market_analyst", "research_facilitator"):
        assert router.profile_for(other).model != judge.model, (
            f"fund_manager shares {judge.model} with {other}"
        )


def test_thinking_is_enabled_only_for_the_judgement_node() -> None:
    """§6: turn the dial on "for the nodes where reasoning is the product".

    It costs ~9x the output tokens (measured on qwen3:8b), and the fund manager
    runs ONCE per decision where the analysts run four times -- so it is the
    cheapest possible place to spend them, and the only place they are worth
    spending at this stage.
    """
    router = router_from_shipped_config()
    thinking = [n for n in router.node_routes if router.profile_for(n).think]
    assert thinking == ["fund_manager"], (
        f"{thinking} have thinking enabled; expected only fund_manager"
    )


def test_an_unrouted_node_fails_with_the_known_nodes_listed() -> None:
    router = router_from_shipped_config()
    with pytest.raises(ProfileNotConfigured, match="market_analyst"):
        router.profile_for("nonexistent_analyst")


def test_all_local_preset_overrides_every_node() -> None:
    """The ablation that makes a run free, and the escape hatch before Stage 5."""
    router = router_from_shipped_config(preset="all_local")
    for node in router.node_routes:
        profile = router.profile_for(node)
        assert profile.provider == "ollama", f"{node} escaped the preset"


def test_all_hosted_preset_overrides_every_node() -> None:
    router = router_from_shipped_config(preset="all_hosted")
    assert router.profile_for("market_analyst").provider == "anthropic"


def test_an_unknown_preset_is_rejected_at_load() -> None:
    with pytest.raises(ProfileNotConfigured, match="unknown preset"):
        LLMRouter.from_config(Settings(), config=load_yaml("models.yaml"), preset="all_free")


def test_thinking_is_off_by_default() -> None:
    """Measured on qwen3:8b: thinking on cost 579 eval tokens vs 64 for the
    same answer, 8.7s vs 2.7s. Across 14 nodes that decides whether the
    pipeline is pleasant to iterate on."""
    assert ModelProfile(name="x", provider="ollama", model="m").think is False
    router = router_from_shipped_config()
    assert router.profile_for("market_analyst").think is False


def test_a_profile_missing_provider_or_model_is_rejected() -> None:
    with pytest.raises(ProfileNotConfigured, match="missing"):
        ModelProfile.from_dict("broken", {"temperature": 0.3})


def test_base_url_comes_from_settings_not_from_the_yaml_placeholder() -> None:
    """models.yaml writes "${OLLAMA_BASE_URL}" for documentation. Expanding env
    vars inside config files is a second configuration language, and one is
    enough -- Settings already owns the environment."""
    settings = Settings(ollama_base_url="http://example.invalid:9999")
    router = LLMRouter.from_config(settings, config=load_yaml("models.yaml"))
    assert router.clients["ollama"].base_url == "http://example.invalid:9999"


async def test_the_hosted_provider_still_fails_loudly_when_reached() -> None:
    """Unimplemented, and it says so rather than producing plausible untested
    API code whose first run would be against a real bill.

    Reached through the `all_hosted` preset now, not through fund_manager --
    which is also why this no longer accidentally makes a live 32B call and
    turns an 18-second test suite back into a 1-second one.
    """
    from research_desk.llm.router import LLMError

    router = router_from_shipped_config(preset="all_hosted")
    with pytest.raises(LLMError, match="Stage 5"):
        await router.complete("fund_manager", [{"role": "user", "content": "x"}])
