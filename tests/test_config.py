"""Config loading, and the one rule that must fail loudly.

The confirmation gate is not a setting. The ORB+GEX repo's options scanner had a
gate that was *inverted* -- ``require_confirmation: false`` still prompted, and
then returned True regardless of the answer. That bug shows up exactly once, on a
real order. So the only acceptable response to a config that asks for the gate to
be off is to refuse to start.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from research_desk.config import (
    CONFIG_DIR,
    CONFIG_FILES,
    ConfigError,
    Settings,
    config_hash,
    load_config,
    load_settings,
    load_yaml,
)
from research_desk.models.modes import RunMode


def test_the_shipped_config_loads() -> None:
    loaded = load_config()
    assert set(loaded) == set(CONFIG_FILES)


def test_config_hash_is_stable_and_order_independent() -> None:
    """Two runs that disagree on config are not comparable at Stage 8."""
    first = load_config()
    assert config_hash(first) == config_hash(load_config())

    reordered = {k: dict(reversed(list(v.items()))) for k, v in first.items()}
    assert config_hash(reordered) == config_hash(first)


def test_disabling_the_confirmation_gate_is_refused(tmp_path: Path) -> None:
    for name in CONFIG_FILES:
        (tmp_path / name).write_text((CONFIG_DIR / name).read_text())

    intent = yaml.safe_load((tmp_path / "portfolio-intent.yaml").read_text())
    intent["execution"]["require_confirmation"] = False
    (tmp_path / "portfolio-intent.yaml").write_text(yaml.safe_dump(intent))

    with pytest.raises(ConfigError, match="no off switch"):
        load_config(config_dir=tmp_path)


def test_the_shipped_intent_actually_enables_the_gate() -> None:
    """Guard against the test above passing vacuously."""
    intent = load_yaml("portfolio-intent.yaml")
    assert intent["execution"]["require_confirmation"] is True
    assert intent["execution"]["confirm_by"] == "ticker"


def test_missing_config_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=str(tmp_path)):
        load_config(config_dir=tmp_path)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def test_settings_default_to_live_mode() -> None:
    assert load_settings({}).mode is RunMode.LIVE


def test_unknown_run_mode_is_rejected_with_the_valid_set() -> None:
    with pytest.raises(ConfigError, match="live, replay, replay_llm"):
        load_settings({"RUN_MODE": "backtest"})


def test_tracing_needs_both_keys() -> None:
    assert not Settings().tracing_enabled
    assert not Settings(langfuse_public_key="pk").tracing_enabled
    assert Settings(langfuse_public_key="pk", langfuse_secret_key="sk").tracing_enabled


def test_default_client_id_is_the_allocated_slot() -> None:
    """11 is the research desk's `execute` slot.

    ib_async connections that collide on clientId do not error -- they silently
    fail to connect (migration plan §0). 2 belongs to the ORB+GEX engine.
    """
    assert load_settings({}).ib_client_id == 11
    assert load_settings({}).ib_port == 4004


def test_empty_env_strings_become_none_rather_than_empty_keys() -> None:
    """`FINNHUB_API_KEY=` in .env must read as absent, not as a key of ''."""
    settings = load_settings({"FINNHUB_API_KEY": "", "ANTHROPIC_API_KEY": ""})
    assert settings.finnhub_api_key is None
    assert settings.anthropic_api_key is None


# --------------------------------------------------------------------------- #
# universe.yaml -- the file whose cost rises with delay
# --------------------------------------------------------------------------- #


def test_every_tradeable_symbol_is_in_the_universe_file() -> None:
    """Anything tradeable must be collecting sentiment.

    The GDELT cache is keyed by symbol and its window is ~3 months, so a symbol
    added to `include` without a universe entry arrives at Stage 8 with no
    sentiment history and no way to get any.
    """
    universe = {s["symbol"] for s in load_yaml("universe.yaml")["symbols"]}
    tradeable = set(load_yaml("portfolio-intent.yaml")["universe"]["include"])
    assert tradeable <= universe, f"not collecting GDELT for: {tradeable - universe}"


def test_every_symbol_has_a_gdelt_query_and_a_sector_etf() -> None:
    known_etfs = set(load_yaml("universe.yaml")["sector_etfs"])
    for entry in load_yaml("universe.yaml")["symbols"]:
        symbol = entry["symbol"]
        assert entry.get("gdelt_query"), f"{symbol} has no gdelt_query"
        # GDELT matches text; a bare ticker matches almost nothing.
        assert entry["gdelt_query"].strip() != symbol, (
            f"{symbol}'s gdelt_query is just the ticker. GDELT matches text, "
            "not tickers -- see the notes in config/universe.yaml."
        )
        sector = entry.get("sector_etf")
        assert sector is None or sector in known_etfs, f"{symbol}: unknown sector ETF {sector}"


def test_benchmark_is_present_and_has_no_sector() -> None:
    universe = load_yaml("universe.yaml")
    benchmark = universe["benchmark"]
    entry = next(s for s in universe["symbols"] if s["symbol"] == benchmark)
    assert entry["sector_etf"] is None, "the benchmark is not measured against a sector"
    assert entry["has_fundamentals"] is False, (
        "SPY has no EDGAR companyfacts -- this is the ETF case that makes a "
        "naive fundamentals path throw at Stage 2."
    )
