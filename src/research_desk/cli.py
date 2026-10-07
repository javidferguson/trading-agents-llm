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
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import (
    REPO_ROOT,
    ConfigError,
    config_hash,
    load_config,
    ib_endpoints,
    load_settings,
    load_yaml,
)
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


def _check_models(settings: Any) -> bool:
    """Are the models config/models.yaml names actually pulled?

    A warning, never a failure. The desk starts fine without them -- Stage 0
    needs no model at all -- but Stage 1's first run would otherwise die inside
    the Ollama client with a 404 on a model name, which reads like a bug in the
    router rather than a missing download.

    Prints the literal fix, because "pull the model" is less useful than the
    command that pulls it.
    """
    import httpx

    try:
        profiles = load_yaml("models.yaml").get("profiles") or {}
    except ConfigError:
        return True  # the config check already reported this

    # profile name -> model, for ollama profiles only. Hosted profiles have
    # nothing to pull.
    wanted: dict[str, str] = {
        name: body["model"]
        for name, body in profiles.items()
        if body.get("provider") == "ollama" and body.get("model")
    }
    if not wanted:
        return True

    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
        present = {m.get("model", "") for m in response.json().get("models", [])}
    except Exception:
        # Ollama being down is _check_ollama's story to tell, not ours.
        _line(WARN, "models", "could not ask Ollama which models are pulled")
        return True

    missing = {n: m for n, m in wanted.items() if m not in present}
    if not missing:
        _line(OK, "models", f"{len(wanted)} configured model(s) present: "
                            f"{', '.join(sorted(set(wanted.values())))}")
        return True

    _line(WARN, "models", f"{len(missing)} of {len(wanted)} configured model(s) not pulled")
    for profile, model in sorted(missing.items(), key=lambda kv: kv[1]):
        print(f"         ollama pull {model:<22} # profile: {profile}")
    print("         or just:  make models")
    return True


def _check_gateway(settings: Any) -> bool:
    """TCP-only, and it reports the ORB conflict as its own condition.

    Two addresses, because there genuinely are two. ``desk-ib-gateway:4004`` is
    the socat-paper port inside the container and cannot resolve from the host;
    ``127.0.0.1:4012`` is its published mapping and does not exist inside the
    network. Probing only the configured one makes `desk doctor` warn forever
    when run from a shell, which trains you to ignore it.

    A detected ORB Gateway is surfaced separately rather than folded into
    "unreachable", because the two have opposite fixes: one means *start*
    something, the other means *stop* something.
    """
    candidates = ib_endpoints(settings)

    for host, port in candidates:
        try:
            with socket.create_connection((host, port), timeout=3):
                pass
        except OSError:
            continue
        _line(OK, "ib gateway", f"{host}:{port} reachable (desk-ib-gateway)")
        return True

    # Ours is down. Is the other one up? That changes the advice entirely.
    conflict = _orb_gateway_detected()
    tried = ", ".join(f"{h}:{p}" for h, p in candidates)

    if conflict == ORB_UNKNOWN:
        # Not folded into "not reachable": that would claim the other Gateway
        # is absent on the strength of a check that never ran. This is the
        # normal result inside the container, where the guard cannot see the
        # host's Docker daemon or its loopback.
        _line(WARN, "ib gateway", f"ours is down ({tried})")
        print("         Whether the ORB+GEX Gateway is up could NOT be determined")
        print("         from here -- that check reads host state, so it does not")
        print("         work inside the container. On the host:")
        print("           make check-gateway")
        return False

    if conflict:
        _line(WARN, "ib gateway", f"ours is down, but the ORB+GEX Gateway IS up ({conflict})")
        print("         Both use the same IB credentials and one username supports")
        print("         one session, so they must never run together.")
        print("         Stop it first:  docker stop ajj-ib-gateway")
        return False

    _line(WARN, "ib gateway", f"not reachable ({tried})")
    print("         not needed until Stage 7. Start it with `make gateway-start`")
    print("         once IB_USERNAME / IB_PASSWORD are set in .env.")
    return False


#: What ``_orb_gateway_detected`` returns when the guard could not look.
#: A distinct value, because "no conflict" and "could not tell" have opposite
#: meanings for a safety check and were previously the same ``None``.
ORB_UNKNOWN = "__unknown__"


def _orb_gateway_detected() -> str | None:
    """Describe the ORB+GEX Gateway if it is running.

    Returns its description on a conflict, ``None`` when there is genuinely no
    conflict, and ``ORB_UNKNOWN`` when the guard could not determine either.

    Delegates to the standalone guard so there is one definition of "the other
    Gateway is up" rather than two that can drift.

    **The three-way return is the fix for a silent fail-open.** This used to
    map every non-conflict exit code to ``None``, which folded exit 2 ("could
    not tell") into exit 0 ("clear"). Inside the container that is always the
    outcome -- the guard has no Docker socket and its port probe reads the
    container's own loopback -- so `desk doctor`, which runs in the container by
    default, could never report an ORB conflict and said "not reachable"
    instead. A guard that cannot see must say so: the whole point is that two
    Gateways on one IB username evict each other.
    """
    script = REPO_ROOT / "scripts" / "check_gateway_exclusive.py"
    if not script.exists():
        return ORB_UNKNOWN
    try:
        result = subprocess.run(
            [sys.executable, str(script), "--quiet"],
            capture_output=True, text=True, timeout=20,
        )
    except (subprocess.SubprocessError, OSError):
        return ORB_UNKNOWN
    if result.returncode == 0:
        return None
    if result.returncode != 1:
        return ORB_UNKNOWN
    for line in result.stderr.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            return stripped[2:]
    return "see `make check-gateway`"


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
        _check_models(settings),
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


# --------------------------------------------------------------------------- #
# decide -- the Stage 3 exit gate
# --------------------------------------------------------------------------- #


async def _run_decide(
    symbol: str, as_of: date, preset: str | None,
    slice_only: bool = False, research_only: bool = False,
) -> int:
    from .graph.build import (
        build_decision_graph,
        build_full_graph,
        build_research_graph,
    )
    from .llm.router import LLMRouter
    from .models.budget import RunBudget
    from .models.state import DecisionState
    from .prompts import prompt_pack_version
    from .providers.registry import ProviderRegistry

    settings = load_settings()
    configure_tracing(settings)

    try:
        hashed = config_hash(load_config())
    except ConfigError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    router = LLMRouter.from_config(settings, preset=preset)
    registry = ProviderRegistry.from_config(settings)

    models_cfg = load_yaml("models.yaml")
    debate_cfg = models_cfg.get("debate") or {}
    budget = RunBudget.from_config(models_cfg)

    ctx = NodeContext(
        settings=settings, mode=settings.mode, as_of=as_of,
        config_hash=hashed, prompt_pack_version=prompt_pack_version(),
        extras={
            "router": router, "registry": registry, "budget": budget,
            "max_research_rounds": int(debate_cfg.get("max_research_rounds", 1)),
            "max_risk_rounds": int(debate_cfg.get("max_risk_rounds", 1)),
        },
    )

    initial = DecisionState(
        run_id=ctx.run_id, symbol=symbol, as_of=as_of, mode=ctx.mode,
        config_hash=hashed, prompt_pack_version=ctx.prompt_pack_version,
    )

    print(f"{symbol}  as_of={as_of}  run={ctx.run_id}")
    print(f"prompts={ctx.prompt_pack_version}  config={hashed}  mode={ctx.mode.value}\n")

    try:
        builder = (
            build_decision_graph if slice_only
            else build_research_graph if research_only
            else build_full_graph
        )
        graph = builder(ctx)
        result = await graph.ainvoke(initial, {"recursion_limit": 60})
    finally:
        await router.aclose()
        flush_tracing()

    state = DecisionState.model_validate(result)
    decision = state.final_decision

    for note in state.notes:
        print(f"  {note}")
    print()

    report = state.analyst_reports.get("market")
    if report is not None:
        print(f"ANALYST  {report.stance} @ {report.confidence:.2f}")
        print(f"  {' '.join(report.summary.split())}")
        for point in report.key_points:
            print(f"  - {' '.join(point.split())}")
        if report.data_gaps:
            print(f"  gaps: {'; '.join(report.data_gaps)}")
        print()

    debate = state.research_debate
    if debate.turns:
        print(f"DEBATE  {debate.rounds_completed} round(s), "
              f"stopped: {debate.stop_reason}")
        for turn in debate.turns:
            flag = "" if turn.new_information else "  [no new information]"
            print(f"  r{turn.round} {turn.speaker:<5} {' '.join(turn.claim.split())[:120]}{flag}")
        verdict = state.research_verdict
        if verdict is not None:
            print(f"  verdict: {verdict.winner} @ {verdict.confidence:.2f}"
                  + (f" (converged: {verdict.converged_reason})" if verdict.converged else ""))
        print()

    risk = state.risk_debate
    if risk.turns:
        print(f"RISK COMMITTEE  {risk.rounds_completed} round(s), "
              f"stopped: {risk.stop_reason}")
        for turn in risk.turns:
            print(f"  r{turn.round} {turn.speaker:<8} "
                  f"{' '.join(turn.claim.split())[:110]}")
        rv = state.risk_verdict
        if rv is not None:
            print(f"  fund manager: {rv.decision.upper()} -> {rv.action} "
                  f"{rv.target_weight_pct:.1f}% @ {rv.conviction:.2f}")
            if rv.adjustment:
                print(f"    adjusted: {' '.join(rv.adjustment.split())}")
        print()

    if decision is None:
        print("No decision was produced.", file=sys.stderr)
        return 1

    print(f"DECISION  {decision.action}  {decision.target_weight_pct:.1f}% of equity"
          f"  conviction {decision.conviction:.2f}  horizon {decision.horizon_days}d")
    print(f"  rationale    {' '.join(decision.rationale.split())}")
    print(f"  invalidation {' '.join(decision.invalidation.split())}")
    print(f"  dissent      {' '.join(decision.dissent.split())}")
    print(f"  expires      {decision.expires_at:%Y-%m-%d %H:%M}")
    print()

    # §9's middle layer. Printed after the decision and before the cost, because
    # that is the order in which it matters: what was decided, what that turns
    # into in shares, and what it cost to find out.
    plan = state.order_plan
    if plan is not None:
        print(plan.render())
        if plan.blocked:
            print()
            print("  VETOED BY PYTHON. The fund manager approved this and the")
            print("  compliance node blocked it -- which is the design (§9),")
            print("  not a malfunction.")
        print()

    spend = sum(r.usd for r in state.llm_calls)
    attempts = len(state.llm_calls)
    repairs = sum(1 for r in state.llm_calls if r.parse_failed)
    wall = sum(r.latency_ms or 0 for r in state.llm_calls) / 1000
    print(f"{attempts} model call(s), {repairs} repair turn(s), "
          f"{wall:.1f}s of inference, ${spend:.4f}")
    exhausted = budget.exhausted()
    print(f"budget: {budget.calls}/{budget.max_llm_calls} calls, "
          f"{budget.elapsed_s:.0f}/{budget.max_wall_s:.0f}s, "
          f"${budget.usd:.4f}/${budget.max_usd:.2f}"
          + (f"  EXHAUSTED ({exhausted})" if exhausted else ""))

    if decision.degraded:
        print()
        print("DEGRADED -- this HOLD is a failure, not a judgement.")
        print(f"  {state.degraded_reason()}")
        return 1

    if plan is not None and plan.blocked:
        # Exit 0: nothing failed. A veto is the system working, and the first
        # live Stage 6 run reporting one as DEGRADED was the bug, not the
        # veto. The block above already said so loudly on stdout.
        print("No order will be placed. Nothing failed.")
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    """Run the vertical slice and write proposal.json."""
    setup_logging()
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    return asyncio.run(_run_decide(
        args.symbol.upper(), as_of, args.preset, args.slice, args.research
    ))


# --------------------------------------------------------------------------- #
# data -- what is cached, from where, and how stale
# --------------------------------------------------------------------------- #


def cmd_data(args: argparse.Namespace) -> int:
    """Inventory the provider cache.

    Answers the question you actually have before trusting a snapshot: is this
    real IB data or the synthetic fixtures, and how old is it?
    """
    import json as _json

    settings = load_settings()
    universe = load_yaml("universe.yaml")
    wanted = sorted(
        {e["symbol"] for e in universe["symbols"]}
        | {universe["benchmark"]}
        | {e["sector_etf"] for e in universe["symbols"] if e.get("sector_etf")}
    )

    bars_dir = settings.cache_dir / "ib"
    found: dict[str, dict] = {}
    for path in sorted(bars_dir.glob("daily_bars-*.json")) if bars_dir.exists() else []:
        try:
            payload = (_json.loads(path.read_text()) or {}).get("payload") or {}
        except Exception:
            continue
        if payload.get("symbol"):
            found[payload["symbol"]] = payload

    today = date.today()
    print(f"{'symbol':<8}{'source':<10}{'bars':>6}  {'first':<12}{'last':<12}{'age':>5}")
    print("-" * 60)

    fixtures = 0
    missing = []
    for symbol in wanted:
        payload = found.get(symbol)
        if not payload:
            missing.append(symbol)
            print(f"{symbol:<8}{'-':<10}{0:>6}  {'-':<12}{'-':<12}{'-':>5}")
            continue
        rows = payload.get("bars") or []
        source = payload.get("source", "?")
        if source != "ib":
            fixtures += 1
        first = rows[0]["day"] if rows else "-"
        last = rows[-1]["day"] if rows else "-"
        age = (today - date.fromisoformat(last)).days if rows else 0
        print(f"{symbol:<8}{source:<10}{len(rows):>6}  {first:<12}{last:<12}{age:>4}d")

    print()
    if fixtures:
        print(f"!! {fixtures} symbol(s) hold SYNTHETIC fixture data, not market prices.")
        print("   Every metric computed from them is fiction. Replace with:")
        print("       make gateway-start && make bars")
    if missing:
        print(f"{len(missing)} symbol(s) have no bars: {', '.join(missing)}")
        print("   `make bars` fetches the whole universe.")
    if not fixtures and not missing:
        print("All universe symbols have real IB bars.")
    # Stale is a warning, not a failure: a long weekend is a legitimate 3 days.
    return 0


# --------------------------------------------------------------------------- #
# snapshot -- the Stage 2 exit gate
# --------------------------------------------------------------------------- #


async def _run_snapshot(symbol: str, as_of: date, show_gaps: bool) -> int:
    from .metrics.snapshot import build_market_snapshot, universe_context
    from .models.market import BLOCK_NAMES
    from .providers.base import ProviderError
    from .providers.registry import ProviderRegistry

    settings = load_settings()
    registry = ProviderRegistry.from_config(settings)
    benchmark, sector = universe_context(load_yaml("universe.yaml"), symbol)

    try:
        snapshot = await build_market_snapshot(
            registry, symbol, as_of,
            benchmark_symbol=benchmark, sector_symbol=sector,
        )
    except ProviderError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    populated, missing = snapshot.metric_count()

    # Loud, because synthetic prices mistaken for real ones would make every
    # number downstream fiction rather than merely wrong.
    if snapshot.sources.get("bars") != "ib":
        print(f"!! bars came from {snapshot.sources.get('bars')!r}, NOT from IB. "
              "These are not market prices.\n")

    print(f"{snapshot.symbol}  as_of={snapshot.as_of}  "
          f"bars={snapshot.bars_available} (latest {snapshot.last_bar_day})")
    print(f"sources: {snapshot.sources}")
    if snapshot.fundamentals_asof:
        print(f"fundamentals: {snapshot.fundamentals_form} filed "
              f"{snapshot.fundamentals_asof}")
    print(f"metrics: {populated} populated, {missing} unavailable\n")

    # Read from BLOCK_NAMES so a new metric block cannot be half-wired: the
    # CLI, all_gaps() and metric_count() all derive from the same list. This
    # printed only the OHLCV blocks after Stage 2b added three more.
    for name in BLOCK_NAMES:
        block = getattr(snapshot, name)
        print(f"  [{name}]")
        for key, value in block.available().items():
            shown = f"{value:,.4f}" if isinstance(value, float) else value
            print(f"    {key:<28} {shown}")
        if not block.available():
            print("    (nothing computed)")
        print()

    if missing:
        print(f"UNAVAILABLE ({missing}) -- every one states why, which is the gate:")
        for key, reason in sorted(snapshot.all_gaps().items()):
            print(f"    {key:<40} {reason}")
    else:
        print(f"Every §7.1 metric populated ({populated}).")
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Print a MarketSnapshot. No LLM involved -- that is the point."""
    setup_logging()
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
    return asyncio.run(_run_snapshot(args.symbol.upper(), as_of, args.gaps))


# --------------------------------------------------------------------------- #
# smoke -- the Stage 1 exit gate
# --------------------------------------------------------------------------- #


async def _run_smoke(node: str, preset: str | None) -> int:
    from .llm.router import LLMRouter
    from .llm.structured import structured
    from .models.state import AnalystReport

    settings = load_settings()
    router = LLMRouter.from_config(settings, preset=preset)
    profile = router.profile_for(node)
    print(f"{node} -> profile {profile.name} ({profile.provider}/{profile.model}, "
          f"think={profile.think})\n")

    try:
        result = await structured(
            router, node, AnalystReport,
            [
                {"role": "system", "content":
                 "You are a market analyst. You are given pre-computed facts. Do not "
                 "recalculate any number and do not invent numbers that are not shown. "
                 "Output only JSON matching the schema."},
                {"role": "user", "content":
                 "Symbol: SPY\n  Last close 678.42\n  Price above SMA 20/50/200\n"
                 "  RSI-14 71.2\n  52-week percentile 0.94\n  ATR-14 4.10\n"
                 "  Fundamentals: unavailable (SPY is an ETF)\n\n"
                 "Produce your market report."},
            ],
            degraded_fields={"kind": "market"},
        )
    finally:
        await router.aclose()

    for record in result.records:
        status = "parse FAILED" if record.parse_failed else "ok"
        print(f"  attempt {record.attempt}: {status}  "
              f"{record.latency_ms}ms  "
              f"{record.prompt_tokens}->{record.completion_tokens} tok  "
              f"digest={(record.model_digest or '?')[:12]}")

    report = result.value
    print()
    print(f"  stance      {report.stance}  (confidence {report.confidence})")
    print(f"  summary     {report.summary[:140]}")
    print(f"  key_points  {len(report.key_points)}")
    print(f"  data_gaps   {report.data_gaps}")
    print()

    if result.degraded:
        print("DEGRADED -- the model never produced a valid report.")
        print(f"  reason: {report.degraded_reason}")
        print("This is correct behaviour (architecture §5 -- never fail toward a")
        print("trade), but the Stage 1 gate is NOT met by a degraded result.")
        return 1

    print(f"Valid AnalystReport in {result.attempts} attempt(s). Stage 1 gate met.")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Ask a local model for a real AnalystReport and show what came back."""
    setup_logging()
    try:
        return asyncio.run(_run_smoke(args.node, args.preset))
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


# --------------------------------------------------------------------------- #
# portfolio / candidates -- Stage 6, and neither touches IB or a model
# --------------------------------------------------------------------------- #


def cmd_portfolio(args: argparse.Namespace) -> int:
    """The book, and the drift table computed from it. No LLM, no broker."""
    from .intent.engine import compute_gaps, load_intent, load_portfolio
    from .metrics.render import render_drift

    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    try:
        intent = load_intent()
        portfolio = load_portfolio()
    except Exception as exc:  # noqa: BLE001 -- a CLI message beats a traceback
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    stale = portfolio.staleness_reason(as_of)
    print(f"book {portfolio.as_of}  source={portfolio.source}  "
          f"equity {portfolio.equity:,.2f} USD")
    if stale:
        # Printed before the table rather than after: every number below is
        # computed from marks this message says not to trust.
        print(f"\n{FAIL} STALE: {stale}\n")
    print()
    print(render_drift(compute_gaps(intent, portfolio, as_of=as_of)))
    return 1 if stale else 0


def cmd_candidates(args: argparse.Namespace) -> int:
    """Channel 1: today's tradeable set, computed before any model runs (§8).

    > *"Universe minus exclusions minus earnings blackout minus
    > at-max-positions -> today's candidates... Saves money and removes a class
    > of hallucination."*
    """
    from .intent.engine import blackout_map, candidates, load_intent, load_portfolio
    from .providers.registry import ProviderRegistry

    setup_logging()
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    try:
        intent = load_intent()
        portfolio = load_portfolio()
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    settings = load_settings()
    registry = ProviderRegistry.from_config(settings)

    blackouts: dict[str, str] = {}
    if not args.no_earnings:
        blackouts = asyncio.run(blackout_map(
            registry, list(intent.universe.tradeable), as_of,
            days_before=intent.risk.earnings_blackout_days_before,
            days_after=intent.risk.earnings_blackout_days_after,
        ))

    result = candidates(intent, portfolio, as_of=as_of, blackouts=blackouts)

    print(f"candidates for {as_of.isoformat()}  "
          f"({len(result.eligible)} of {len(intent.universe.include)} eligible)")
    print(f"  book: {portfolio.position_count}/{intent.risk.max_positions} "
          f"positions, {portfolio.cash_pct:.1f}% cash\n")

    print("ELIGIBLE")
    for symbol in result.eligible:
        held = portfolio.get(symbol)
        mark = f"held {portfolio.weight_pct(symbol):5.2f}%" if held else "not held"
        print(f"  {symbol:7} {mark}")

    if result.rejected:
        print("\nFILTERED OUT")
        for symbol, reason in result.rejected.items():
            print(f"  {symbol:7} {' '.join(reason.split())}")

    if args.earnings:
        from .intent.earnings import estimate_next_report

        print("\nNEXT REPORT ESTIMATES")
        print("  EDGAR publishes no forward calendar, so these are projected")
        print("  from filing cadence and WARN rather than block. See")
        print("  intent/earnings.py.")

        async def _show() -> None:
            for symbol in intent.universe.tradeable:
                estimate = await estimate_next_report(registry, symbol, as_of)
                print(f"  {' '.join(estimate.describe().split())}")

        asyncio.run(_show())

    return 0


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

    dec = sub.add_parser("decide", help="Stage 3 exit gate: a real decision -> proposal.json")
    dec.add_argument("--symbol", default="MSFT")
    dec.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today)")
    dec.add_argument("--slice", action="store_true",
                     help="Stage 3 shape: one analyst, no debate")
    dec.add_argument("--research", action="store_true",
                     help="Stage 4 shape: analysts + research debate, no risk committee")
    dec.add_argument("--preset", default="all_local",
                     help="models.yaml preset (default all_local -- hosted is Stage 5)")
    dec.set_defaults(func=cmd_decide)

    sub.add_parser("data", help="what is in the provider cache, and is it real").set_defaults(
        func=cmd_data
    )

    snap = sub.add_parser("snapshot", help="Stage 2 exit gate: every §7.1 metric, no LLM")
    snap.add_argument("--symbol", default="SPY")
    snap.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today)")
    snap.add_argument("--gaps", action="store_true", help="(gaps always shown)")
    snap.set_defaults(func=cmd_snapshot)

    smoke = sub.add_parser("smoke", help="Stage 1 exit gate: a real AnalystReport from a local model")
    smoke.add_argument("--node", default="market_analyst")
    smoke.add_argument("--preset", default="all_local",
                       help="models.yaml preset (default all_local -- hosted is Stage 5)")
    smoke.set_defaults(func=cmd_smoke)

    pf = sub.add_parser(
        "portfolio",
        help="Stage 6: the book and the drift table, no LLM and no broker",
    )
    pf.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today)")
    pf.set_defaults(func=cmd_portfolio)

    cand = sub.add_parser(
        "candidates",
        help="Stage 6 channel 1: today's tradeable set, before any model runs",
    )
    cand.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today)")
    cand.add_argument("--earnings", action="store_true",
                      help="also print the next-report estimate for every symbol")
    cand.add_argument("--no-earnings", action="store_true",
                      help="skip the earnings filter (it reads EDGAR filing history)")
    cand.set_defaults(func=cmd_candidates)

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
