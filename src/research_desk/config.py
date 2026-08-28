"""Settings and config loading.

Two rules carried from the ORB+GEX engine's ``config.py``:

1. **The human gate is not a setting.** ``require_confirmation: false`` is
   rejected at load time, loudly. The options scanner's gate was *inverted* --
   setting it false still prompted and then returned True regardless of the
   answer -- which is exactly the kind of bug that only shows up once, on a real
   order. There is deliberately no code path that turns the gate off.
2. **Make "which keys apply when" obvious from the file's shape.** Config is
   split across four small YAML files by concern rather than one large one.

Config that a *model* reads (theses, constraints, prose) lives in
``portfolio-intent.yaml``. Config that *Python* enforces lives under ``risk:``
in the same file and is never shown to a model as something negotiable.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models.modes import RunMode

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"

#: The four config files, and whether the run can proceed without each.
CONFIG_FILES: dict[str, bool] = {
    "models.yaml": True,
    "providers.yaml": True,
    "portfolio-intent.yaml": True,
    "universe.yaml": True,
}


ENV_FILE = REPO_ROOT / ".env"


class ConfigError(RuntimeError):
    """Configuration is missing, malformed, or asks for something forbidden."""


def read_dotenv(path: Path | None = None) -> dict[str, str]:
    """Parse ``.env`` into a dict. Deliberately not a dependency.

    The compose services get ``.env`` through ``env_file:``, but a local
    ``uv run desk ...`` gets nothing -- which produced a confusing first run
    where the keys were plainly there in the file and the app insisted they were
    not. Real environment variables still win, so ``RUN_MODE=replay uv run desk``
    behaves the way anyone would expect.

    Comment lines, blanks, ``export`` prefixes and surrounding quotes are
    handled; nothing else is. If this ever needs more, use python-dotenv.
    """
    target = path or ENV_FILE
    if not target.exists():
        return {}

    values: dict[str, str] = {}
    for line in target.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Environment-derived settings. Secrets stay here and never reach a prompt."""

    mode: RunMode = RunMode.LIVE
    log_level: str = "INFO"
    data_dir: Path = field(default=Path("data"))

    # Ollama. See architecture §10 -- the daemon binds 127.0.0.1 by default and
    # a container cannot reach it, which is the single most common hour lost on
    # this stack. `make check-ollama` diagnoses it from inside the container.
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_timeout_s: int = 300

    anthropic_api_key: str | None = None

    # Langfuse. Absent keys are not an error -- tracing degrades to no-op so the
    # graph still runs, which matters when you are debugging on a plane.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"

    # Providers.
    finnhub_api_key: str | None = None
    fred_api_key: str | None = None
    #: EDGAR 403s without a `User-Agent: Name email` header. This is the usual
    #: cause, and the error message it returns does not say so.
    sec_user_agent: str | None = None

    # IB. Read by `execute` only; `decide` must not even import ib_async.
    # 4004 from inside `trading-network`, 4002 from the host.
    ib_host: str = "ajj-ib-gateway"
    ib_port: int = 4004
    #: 11 is the research desk's `execute` slot. The ORB engine owns 2, and
    #: ib_async connections that collide on clientId do not error -- they
    #: silently fail to connect. See migration plan §0.
    ib_client_id: int = 11

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def journal_dir(self) -> Path:
        return self.data_dir / "journal"

    @property
    def proposals_dir(self) -> Path:
        return self.data_dir / "proposals"

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Build ``Settings`` from ``.env`` overlaid with the real environment.

    Pass ``env`` explicitly in tests to get a hermetic result -- doing so skips
    ``.env`` entirely, so a developer's local keys cannot change a test outcome.
    """
    if env is not None:
        src = dict(env)
    else:
        src = {**read_dotenv(), **os.environ}

    raw_mode = src.get("RUN_MODE", "live").strip().lower()
    try:
        mode = RunMode(raw_mode)
    except ValueError as exc:
        valid = ", ".join(m.value for m in RunMode)
        raise ConfigError(f"RUN_MODE={raw_mode!r} is not one of: {valid}") from exc

    return Settings(
        mode=mode,
        log_level=src.get("LOG_LEVEL", "INFO").upper(),
        data_dir=Path(src.get("DATA_DIR", "data")),
        ollama_base_url=src.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        ollama_timeout_s=int(src.get("OLLAMA_TIMEOUT_S", "300")),
        anthropic_api_key=src.get("ANTHROPIC_API_KEY") or None,
        langfuse_public_key=src.get("LANGFUSE_PUBLIC_KEY") or None,
        langfuse_secret_key=src.get("LANGFUSE_SECRET_KEY") or None,
        langfuse_host=src.get("LANGFUSE_HOST", "http://localhost:3000").rstrip("/"),
        finnhub_api_key=src.get("FINNHUB_API_KEY") or None,
        fred_api_key=src.get("FRED_API_KEY") or None,
        sec_user_agent=src.get("SEC_EDGAR_USER_AGENT") or None,
        ib_host=src.get("IB_HOST", "ajj-ib-gateway"),
        ib_port=int(src.get("IB_PORT", "4004")),
        ib_client_id=int(src.get("IB_CLIENT_ID", "11")),
    )


def load_yaml(name: str, *, config_dir: Path | None = None) -> dict[str, Any]:
    """Load one config file. Raises ``ConfigError`` with a path, not a traceback."""
    path = (config_dir or CONFIG_DIR) / name
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    try:
        loaded = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return loaded


def _assert_gate_intact(intent: dict[str, Any]) -> None:
    """Refuse to load a config that tries to disable the confirmation gate.

    Carried from ``config.py:154`` of the ORB engine. The gate is not a setting,
    so the only correct response to seeing it turned off is to refuse to start.
    """
    execution = intent.get("execution") or {}
    if execution.get("require_confirmation") is False:
        raise ConfigError(
            "portfolio-intent.yaml sets execution.require_confirmation: false. "
            "The human confirmation gate has no off switch -- every order on "
            "this path requires explicit approval, and there is deliberately no "
            "config flag to bypass it. Remove the key."
        )


def load_config(*, config_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load and validate all four config files."""
    loaded = {name: load_yaml(name, config_dir=config_dir) for name in CONFIG_FILES}
    _assert_gate_intact(loaded["portfolio-intent.yaml"])
    return loaded


def config_hash(loaded: dict[str, dict[str, Any]]) -> str:
    """A stable hash of the resolved config.

    Stamped into ``DecisionState`` so that at Stage 8 you can tell whether two
    runs are actually comparable. Sorted keys, so key order does not change it.
    """
    blob = json.dumps(loaded, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
