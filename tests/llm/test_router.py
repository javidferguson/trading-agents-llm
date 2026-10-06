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


def test_the_fund_manager_is_the_one_node_routed_to_hosted() -> None:
    """Architecture §6 -- 'the one node worth paying for'.

    Asserted because quietly routing a second node to `deep_hosted` multiplies
    the bill, and §12's cost table assumes exactly one.
    """
    router = router_from_shipped_config()
    hosted = [
        node for node in router.node_routes
        if router.profile_for(node).provider == "anthropic"
    ]
    assert hosted == ["fund_manager"]


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


async def test_the_hosted_provider_fails_loudly_until_stage_5() -> None:
    """Reachable by config today, so a routing mistake must be obvious rather
    than producing plausible untested API code against a real bill."""
    from research_desk.llm.router import LLMError

    router = router_from_shipped_config()
    with pytest.raises(LLMError, match="Stage 5"):
        await router.complete("fund_manager", [{"role": "user", "content": "x"}])
