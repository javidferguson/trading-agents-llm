"""Turn a ``MarketSnapshot`` into the facts a model reads. The §2 boundary.

> *"Every LLM node receives rendered pre-computed facts and returns JSON
> constrained by a Pydantic schema."*

Four rules, and three of them are about what NOT to write.

**No formulas, no raw series.** The model gets finished numbers. It never sees
a price history, because a model handed 1,253 closes will try to compute a
moving average and will get it wrong (§2).

**Unavailable is stated, never omitted.** A missing metric appears with its
reason. Silently dropping it invites the model to assume a value, and §7.5 is
emphatic that a missing measurement must stay distinguishable from a real one.
It also gives the model an honest alternative to inventing a number, which is
what ``data_gaps`` on ``AnalystReport`` is for.

**No adjectives on sentiment. Percentiles and direction only.** §7.1: *"Tell an
8B model 'sentiment is very positive' and it will say BUY, confidently, every
time"* -- while the better-documented short-horizon reading of extreme bullish
sentiment is *contrarian*. So this module writes `0.74` and never "positive".
The same restraint applies to RSI, drawdown and short interest: no
"overbought", no "stretched", no "crowded". The researchers argue; the renderer
reports.

**No intent, ever (§8).** Analysts are intent-blind. Priming the fundamentals
analyst with "we are bullish on AI infrastructure" before it reads the 10-K
evaporates the entire value of a bear researcher. Intent enters at the trader
node and no earlier, which is why it reaches the prompt through a separate
function here.
"""

from __future__ import annotations

from typing import Any

from ..models.market import BLOCK_NAMES, MarketSnapshot

#: Human labels for the metric blocks, in reading order.
BLOCK_TITLES = {
    "trend": "Price and trend",
    "risk": "Risk and liquidity",
    "mean_reversion": "Mean reversion",
    "relative_strength": "Relative strength",
    "value": "Value (yields, not multiples)",
    "quality": "Quality",
    "growth": "Growth and dilution",
    "positioning": "Positioning (revealed, from regulated disclosure)",
    "sentiment": "News tone and attention (percentiles of its own history)",
}

#: Units and scale hints, so a bare number is not ambiguous. Deliberately
#: factual: "0 to 1" not "high is good".
UNITS = {
    # Observed: qwen3:8b read "momentum_12_1_pct" as "momentum over 12 days"
    # and reported it as such in a key point. The name is the academic
    # convention and the model has no way to know that, so the unit says it.
    # §2 says compute in Python and judge in the model -- a fact the model
    # misreads is not a fact it was given.
    "momentum_12_1_pct": "12-month return EXCLUDING the most recent month "
                         "(the standard momentum definition)",
    "reversal_1m_pct": "return over the last ~21 sessions, which opposes "
                       "12-1 momentum at short horizon",
    "realized_vol_20d_pct": "annualised, from 20 daily returns",
    "realized_vol_60d_pct": "annualised, from 60 daily returns",
    "vol_ratio_20_60": "20-day vol / 60-day vol; below 1 means contracting",
    "pct_52w_range": "0 to 1, where in its own 52-week range",
    "cross_state": "+1 = 50-day above 200-day, -1 = below",
    "bollinger_pct_b": "0 to 1 within the bands; outside means beyond them",
    "beta_vs_spy": "vs the benchmark",
    "accruals": "(net income - operating cash flow) / assets; negative means "
                "earnings are backed by cash",
    "gross_profitability": "gross profit / assets",
    "piotroski_f_score": "0 to 9",
    "amihud_illiquidity": "basis points of price move per $1bn traded",
    "news_tone_percentile_1y": "0 to 1, against this symbol's own trailing year",
    "coverage_volume_zscore_90d": "standard deviations vs its own prior 90 days",
    "short_volume_ratio": "0 to 1, short share of reported volume",
    "short_interest_age_days": "days since the settlement date; short interest "
                               "is semi-monthly and published ~8 business days late",
    "insider_net_usd": "discretionary open-market trades only; exercises, "
                       "awards and tax withholding excluded",
    "insider_nondiscretionary_count": "exercises, awards, gifts, tax withholding "
                                      "-- not decisions about price",
    "market_cap": "USD",
    "dollar_adv_20d": "USD, 20-day average daily dollar volume",
}


def _format(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) >= 1e9:
            return f"{value / 1e9:,.2f}bn"
        if abs(value) >= 1e6:
            return f"{value / 1e6:,.2f}m"
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_snapshot(snapshot: MarketSnapshot, *, include_gaps: bool = True) -> str:
    """The facts block handed to an analyst. Contains no intent and no advice."""
    lines = [
        f"Symbol: {snapshot.symbol}",
        f"As of: {snapshot.as_of.isoformat()}",
        f"Daily bars available: {snapshot.bars_available} "
        f"(most recent {snapshot.last_bar_day})",
    ]
    if snapshot.fundamentals_asof:
        lines.append(
            f"Fundamentals from: {snapshot.fundamentals_form} filed "
            f"{snapshot.fundamentals_asof.isoformat()}"
        )
    lines.append("")

    for name in BLOCK_NAMES:
        block = getattr(snapshot, name, None)
        if block is None:
            continue
        available = block.available()
        if not available:
            continue
        lines.append(f"{BLOCK_TITLES.get(name, name)}")
        for key, value in available.items():
            unit = UNITS.get(key)
            suffix = f"   [{unit}]" if unit else ""
            lines.append(f"  {key:<32} {_format(value)}{suffix}")
        lines.append("")

    if include_gaps:
        gaps = snapshot.all_gaps()
        if gaps:
            lines.append(
                f"NOT AVAILABLE for this symbol ({len(gaps)}). These are absent, "
                "not zero -- do not estimate them, and say so in data_gaps if "
                "they matter to your read:"
            )
            # Grouped by reason: thirty lines of the same sentence is noise, and
            # one line saying "these 22 share a cause" is information.
            by_reason: dict[str, list[str]] = {}
            for metric, reason in sorted(gaps.items()):
                by_reason.setdefault(reason, []).append(metric)
            for reason, metrics in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
                if len(metrics) == 1:
                    lines.append(f"  {metrics[0]}: {reason}")
                else:
                    lines.append(f"  {', '.join(metrics)}")
                    lines.append(f"      -> {reason}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_intent(intent: dict[str, Any]) -> str:
    """Portfolio intent, for the TRADER and FUND MANAGER only (§8).

    A separate function from ``render_snapshot`` so that giving intent to an
    analyst takes a deliberate, visible act rather than an accident of reuse.
    §8 asks for a code comment so nobody "helpfully" fixes it later; this
    docstring is it, and ``tests/test_intent_blindness.py`` enforces it.
    """
    objective = intent.get("objective") or {}
    risk = intent.get("risk") or {}

    max_position = risk.get("max_position_pct", "?")

    lines = [
        "PORTFOLIO INTENT",
        f"  Horizon: {objective.get('horizon_days', '?')} days "
        "(the desk's general horizon -- choose the horizon THIS thesis needs, "
        "which may be shorter or longer)",
        f"  Benchmark: {objective.get('benchmark', '?')}",
        "",
        "  Objective:",
        f"    {' '.join((objective.get('description') or '').split())}",
        "",
        "  >> CEILING ON A SINGLE SYMBOL: "
        f"target_weight_pct must not exceed {max_position}. <<",
        "",
        "  Hard limits (enforced in Python after you propose -- a violation",
        "  blocks the order regardless of what you decide):",
        f"    max position            {max_position}% of equity, PER SYMBOL",
        f"    per-trade risk          {risk.get('per_trade_risk_pct', '?')}% of equity",
        f"    default stop            {risk.get('default_stop_pct', '?')}%",
        f"    max positions           {risk.get('max_positions', '?')}",
        "",
    ]

    themes = intent.get("themes") or []
    if themes:
        lines += [
            "  Themes. A theme's target is the TOTAL across every symbol in it,",
            f"  NOT a target for any one symbol -- no single symbol may exceed",
            f"  {max_position}%. A theme targeting 35% across three exemplars",
            "  implies roughly 12% each, still subject to the ceiling above.",
            "",
        ]
        for theme in themes:
            exemplar_count = len(theme.get("exemplars") or []) or 1
            target = theme.get("target_weight_pct")
            implied = (
                f", so ~{float(target) / exemplar_count:.1f}% each across "
                f"{exemplar_count} exemplars"
                if isinstance(target, (int, float)) else ""
            )
            lines.append(
                f"    {theme.get('name')} -- {target}% of equity IN TOTAL"
                f"{implied}; conviction {theme.get('conviction')}"
            )
            thesis = " ".join((theme.get("thesis") or "").split())
            if thesis:
                lines.append(f"      {thesis}")
            exemplars = theme.get("exemplars") or []
            if exemplars:
                lines.append(f"      exemplars: {', '.join(exemplars)}")
        lines.append("")

    constraints = intent.get("constraints") or []
    if constraints:
        lines.append("  Constraints:")
        for item in constraints:
            lines.append(f"    - {' '.join(str(item).split())}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
