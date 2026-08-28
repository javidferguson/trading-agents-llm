"""Command line entry point for the ``decide`` side.

``execute`` gets its own entry point at Stage 7 and deliberately shares nothing
with this one but the filesystem. No TUI (§6 of the migration plan, open items).

Nothing in this module imports ``ib_async``, including ``doctor`` -- the Gateway
check is a bare TCP connect precisely so that the §0 split survives a
convenience.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import ConfigError, config_hash, load_config, load_settings
from .context import NodeContext
from .logging_setup import setup_logging
from .models.state import DecisionState
from .obs.tracing import configure_tracing, flush_tracing, tracing_healthy

OK = "  ok  "
WARN = " warn "
FAIL = " FAIL "


def _line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


def _check_ollama(settings: Any) -> bool:
    """Reachability, plus the hint for the trap that actually bites.

    Ollama binds ``127.0.0.1:11434`` by default, which a container cannot reach.
    Run this *from inside the container* (``make check-ollama``) -- from the host
    it will pass while the containerised run still fails. Architecture §10.
    """
    import httpx

    url = f"{settings.ollama_base_url}/api/tags"
    try:
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
    except Exception as exc:
        _line(FAIL, "ollama", f"{url}: {type(exc).__name__}")
        print("         if this passes on the host but fails in the container,")
        print("         the daemon is bound to 127.0.0.1. Set OLLAMA_HOST=0.0.0.0:11434")
        print("         on the *host* daemon and restart it.")
        return False

    models = [m.get("model", "?") for m in response.json().get("models", [])]
    _line(OK, "ollama", f"{len(models)} model(s): {', '.join(models[:4]) or 'none pulled'}")
    if not models:
        print("         no models pulled yet -- `ollama pull qwen3:8b`")
    return True


def _check_gateway(settings: Any) -> bool:
    """TCP-only. This repo never starts a Gateway; the ORB+GEX repo owns it.

    Two addresses, because there are genuinely two. ``ajj-ib-gateway:4004`` is
    the name from inside ``trading-network`` and cannot resolve from the host;
    ``127.0.0.1:4002`` is the host mapping and does not exist inside the network.
    Checking only the configured one makes `desk doctor` warn forever when run
    from a shell, which trains you to ignore it.
    """
    candidates = [(settings.ib_host, settings.ib_port), ("127.0.0.1", 4002)]

    for host, port in candidates:
        try:
            with socket.create_connection((host, port), timeout=3):
                pass
        except OSError:
            continue
        _line(OK, "ib gateway", f"{host}:{port} reachable")
        return True

    tried = ", ".join(f"{h}:{p}" for h, p in candidates)
    _line(WARN, "ib gateway", f"not reachable ({tried})")
    print("         not needed until Stage 7. The Gateway is owned by the")
    print("         ORB+GEX repo -- run `make gateway-start` *there*.")
    return False


def _check_langfuse(settings: Any) -> bool:
    if not settings.tracing_enabled:
        _line(WARN, "langfuse", "keys not set; runs will be untraced")
        print("         `make langfuse-up`, then copy the project keys into .env")
        return False
    configure_tracing(settings)
    if tracing_healthy():
        _line(OK, "langfuse", settings.langfuse_host)
        return True
    _line(FAIL, "langfuse", f"credentials rejected by {settings.langfuse_host}")
    return False


def _check_config() -> bool:
    try:
        loaded = load_config()
    except ConfigError as exc:
        _line(FAIL, "config", str(exc))
        return False
    _line(OK, "config", f"4 files, hash {config_hash(loaded)}")
    return True


def _check_dirs(settings: Any) -> bool:
    ok = True
    for path in (settings.cache_dir, settings.journal_dir, settings.proposals_dir):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".writable"
            probe.touch()
            probe.unlink()
        except OSError as exc:
            _line(FAIL, "data dirs", f"{path}: {exc}")
            ok = False
    if ok:
        _line(OK, "data dirs", str(settings.data_dir.resolve()))
    return ok


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report on everything the stack needs, in one screen."""
    settings = load_settings()
    print(f"research-desk {__version__}  mode={settings.mode.value}\n")

    _line(OK, "python", sys.version.split()[0])
    results = [
        _check_config(),
        _check_dirs(settings),
        _check_ollama(settings),
        _check_langfuse(settings),
    ]
    # The Gateway is a warning, never a failure, until Stage 7.
    _check_gateway(settings)

    print()
    if all(results):
        print("All required checks passed.")
        return 0
    print("Some checks failed. Stage 0 needs config, data dirs and Langfuse;")
    print("Ollama is needed from Stage 1, and the Gateway not until Stage 7.")
    return 1


# --------------------------------------------------------------------------- #
# config-check
# --------------------------------------------------------------------------- #


def cmd_config_check(args: argparse.Namespace) -> int:
    try:
        loaded = load_config()
    except ConfigError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    for name, body in loaded.items():
        print(f"  {name:<24} {len(body)} top-level key(s): {', '.join(sorted(body))}")
    print(f"\nconfig_hash = {config_hash(loaded)}")
    print("Confirmation gate: intact (require_confirmation cannot be disabled).")
    return 0


# --------------------------------------------------------------------------- #
# toy-graph -- the Stage 0 exit gate
# --------------------------------------------------------------------------- #


async def _run_toy(symbol: str, as_of: date) -> DecisionState:
    # Imported here rather than at module scope so that `desk doctor` and
    # `desk config-check` do not pay langgraph's import cost.
    from .graph.build import build_linear_graph
    from .graph.nodes.toy import toy_prefetch, toy_summarise

    settings = load_settings()
    configure_tracing(settings)

    try:
        hashed = config_hash(load_config())
    except ConfigError:
        hashed = "unconfigured"

    ctx = NodeContext(settings=settings, mode=settings.mode, as_of=as_of, config_hash=hashed)
    graph = build_linear_graph(
        [("toy_prefetch", toy_prefetch), ("toy_summarise", toy_summarise)],
        ctx,
    )

    initial = DecisionState(
        run_id=ctx.run_id,
        symbol=symbol,
        as_of=as_of,
        mode=ctx.mode,
        config_hash=hashed,
    )
    result = await graph.ainvoke(initial)
    return DecisionState.model_validate(result)


def cmd_toy_graph(args: argparse.Namespace) -> int:
    """Run the two-node graph. Both nodes should appear as spans in Langfuse."""
    setup_logging()
    settings = load_settings()
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    try:
        state = asyncio.run(_run_toy(args.symbol.upper(), as_of))
    finally:
        # Without this the process exits before the exporter has sent anything
        # and the trace you went looking for is simply not there.
        flush_tracing()

    print(json.dumps(state.model_dump(mode="json"), indent=2))
    print()

    # Report what actually happened, not what was configured. Claiming the gate
    # is met because two keys are present -- while the exporter was quietly
    # dropping every span into an unreachable host -- would make this command
    # worse than useless.
    if not settings.tracing_enabled:
        print("Ran UNTRACED: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set.")
        print("The Stage 0 exit gate is NOT met. `make langfuse-up`, then set them in .env.")
        return 1

    if not tracing_healthy():
        print(f"Ran UNTRACED: {settings.langfuse_host} did not accept those credentials.")
        print("The graph itself ran fine -- tracing is never load-bearing -- but the")
        print("Stage 0 exit gate is NOT met. Check `make langfuse-up` finished its")
        print("first-boot migrations, and that the keys in .env match the ones the")
        print("stack was initialised with.")
        return 1

    print(f"Traced. Two spans -- toy_prefetch, toy_summarise -- under run_id")
    print(f"{state.run_id} at {settings.langfuse_host}.")
    print("Open them and confirm both are there: that is the Stage 0 exit gate.")
    return 0


# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="desk", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check that the environment is wired up").set_defaults(
        func=cmd_doctor
    )
    sub.add_parser("config-check", help="parse and validate config/").set_defaults(
        func=cmd_config_check
    )

    toy = sub.add_parser("toy-graph", help="Stage 0 exit gate: two nodes, two spans")
    toy.add_argument("--symbol", default="SPY")
    toy.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today)")
    toy.set_defaults(func=cmd_toy_graph)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
