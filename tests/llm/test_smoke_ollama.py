"""THE STAGE 1 EXIT GATE, run against a real local model.

Migration plan, Stage 1: *"a smoke test asks a local 8B model for an
`AnalystReport` and gets a valid Pydantic object back -- including when the
first attempt is deliberately made to fail schema validation."*

Both halves matter. The first proves schema-constrained decode works against
real weights rather than against a stub that always complies. The second
proves the repair turn works against a model that has actually just made a
mistake, which is a different thing from a scripted reply.

These skip when Ollama is not running or the model is not pulled, so the suite
stays green in CI and on a laptop with no models. `make smoke` runs them
explicitly and fails loudly if they are skipped.
"""

from __future__ import annotations

import json

import httpx
import pytest

from research_desk.config import Settings, load_yaml
from research_desk.llm.router import LLMRouter
from research_desk.llm.structured import structured
from research_desk.models.state import AnalystReport

pytestmark = pytest.mark.smoke

#: The profile Stage 1's gate names. Not deep_local -- the claim being tested
#: is that an *8B* model is usable here, which is the whole §2 bet.
SMOKE_PROFILE = "quick"


def _ollama_state() -> tuple[bool, str]:
    settings = Settings()
    model = load_yaml("models.yaml")["profiles"][SMOKE_PROFILE]["model"]
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
    except Exception:
        return False, f"Ollama not reachable at {settings.ollama_base_url}"
    pulled = {m.get("model") for m in response.json().get("models", [])}
    if model not in pulled:
        return False, f"{model} not pulled -- `ollama pull {model}`"
    return True, model


AVAILABLE, REASON = _ollama_state()
pytestmark = [pytest.mark.smoke, pytest.mark.skipif(not AVAILABLE, reason=REASON)]


@pytest.fixture
async def router():
    r = LLMRouter.from_config(Settings(), config=load_yaml("models.yaml"), preset="all_local")
    yield r
    await r.aclose()


FACTS = """Symbol: SPY (SPDR S&P 500 ETF Trust)
As of: 2026-10-06

Price and trend
  Last close            678.42
  SMA 20 / 50 / 200     671.10 / 659.83 / 612.47   (price above all three)
  50/200 cross state    golden cross, 214 days ago
  12-1 momentum         +14.2%   (73rd percentile of its own 5y history)
  1-month reversal      +2.1%
  52-week percentile    0.94
  Max drawdown 1y       -8.3%

Risk and liquidity
  ATR-14                4.10   (0.60% of price)
  Realized vol 20d/60d  9.8% / 12.4%   (ratio 0.79, contracting)
  20d dollar ADV        $31.2bn

Mean reversion
  RSI-14                71.2
  Bollinger %B          0.96
  Distance from 20d VWAP +1.4%

Not available for this symbol
  Fundamentals: SPY is an ETF and has no EDGAR companyfacts.
"""

SYSTEM = (
    "You are a market analyst on a research desk. You are given pre-computed "
    "facts. Do not recalculate any number and do not invent numbers that are "
    "not shown. If something is marked unavailable, say so in data_gaps "
    "rather than estimating it. Output only JSON matching the schema."
)


async def test_an_8b_model_returns_a_valid_analyst_report(router) -> None:
    """Half one of the gate: real weights, schema-constrained, validates."""
    result = await structured(
        router, "market_analyst", AnalystReport,
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": FACTS + "\nProduce your market report."},
        ],
        degraded_fields={"kind": "market"},
    )

    assert not result.degraded, (
        f"degraded after {result.attempts} attempt(s): {result.value.degraded_reason}"
    )
    report = result.value
    assert report.kind == "market"
    assert report.stance in {"bullish", "bearish", "neutral"}
    assert 0.0 <= report.confidence <= 1.0
    assert 3 <= len(report.key_points) <= 6
    assert len(report.summary.split()) <= 200


async def test_the_record_is_complete_enough_to_replay(router) -> None:
    """`replay_llm` is worthless retroactively, so this is checked at Stage 1,
    not when the replay harness is built at Stage 8."""
    result = await structured(
        router, "market_analyst", AnalystReport,
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": FACTS + "\nProduce your market report."},
        ],
        degraded_fields={"kind": "market"},
    )

    record = result.records[-1]
    assert record.raw_response, "raw_response IS the replay substrate"
    if not result.degraded:
        # The successful attempt's raw text must round-trip on its own.
        assert AnalystReport.model_validate_json(record.raw_response)
    assert record.model_digest, "two runs of qwen3:8b are not necessarily same weights"
    assert record.prompt_tokens and record.completion_tokens
    assert record.latency_ms > 0
    assert record.prompt_hash


class FirstCallReturns:
    """Wraps the real client; serves a canned first reply, then gets out of the way.

    Why not just instruct the model to misbehave? Because the instruction stays
    in the conversation and outranks the correction. Measured: told to report
    confidence "as a PERCENTAGE", qwen3:8b returned 85; told it must be <= 1 it
    returned 82, then 80 on a later run. It was still obeying the earlier, more
    specific instruction -- so that test was measuring instruction-following
    precedence, not the repair turn, and no amount of prompt escalation would
    have made it a fair gate.

    This keeps the failure deterministic and the *repair* entirely real: a real
    8B model, the real repair prompt, real constrained decode.
    """

    def __init__(self, inner, canned: str):
        self._inner = inner
        self._canned = canned
        self.calls = 0

    async def complete(self, profile, messages, schema=None):
        self.calls += 1
        if self.calls == 1:
            from research_desk.llm.router import LLMResponse

            return LLMResponse(
                text=self._canned,
                model=profile.model,
                profile=profile.name,
                provider=profile.provider,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0,
            )
        return await self._inner.complete(profile, messages, schema)

    async def aclose(self):
        await self._inner.aclose()


async def test_a_deliberately_failed_first_attempt_is_repaired(router) -> None:
    """Half two of the gate: a real 8B model recovers a schema violation.

    The planted failure is `confidence: 85` -- a genuinely common real mistake
    (a model meaning 0.85), and one Ollama's decode does not catch, because
    measured against qwen3:8b it enforces enums, types and minItems but NOT
    minimum/maximum.
    """
    planted = json.dumps({
        "kind": "market",
        "stance": "bullish",
        "confidence": 85,          # the violation: must be 0..1
        "summary": "Trend is intact and momentum is stretched.",
        "key_points": ["above all three SMAs", "RSI-14 at 71.2", "52w percentile 0.94"],
        "evidence": [],
        "data_gaps": ["no fundamentals: SPY is an ETF"],
    })
    router.clients["ollama"] = FirstCallReturns(router.clients["ollama"], planted)

    result = await structured(
        router, "market_analyst", AnalystReport,
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": FACTS + "\nProduce your market report."},
        ],
        degraded_fields={"kind": "market"},
    )

    assert result.attempts == 2, "exactly one repair turn"
    assert result.records[0].parse_failed is True
    assert not result.degraded, (
        f"a real 8B model could not repair confidence=85 -> 0..1: "
        f"{result.value.degraded_reason}"
    )
    assert 0.0 <= result.value.confidence <= 1.0
    assert result.records[1].parse_failed is False
    # Both attempts billed: a budget counting only successes under-reports.
    assert len(result.records) == 2


async def test_an_unrecoverable_failure_degrades_rather_than_crashing(router) -> None:
    """The other half of §5, proven against real weights.

    The word cap is the hard case on purpose: a model cannot count, so even a
    well-phrased repair may not land. Measured -- asked for 400+ words then
    told to cut, qwen3:8b went 412 -> 304 and still failed. The requirement is
    not that the model always recovers; it is that failing produces a safe,
    clearly-marked report instead of an exception eight minutes into a run.
    """
    result = await structured(
        router, "market_analyst", AnalystReport,
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": FACTS + (
                    "\nProduce your market report. The summary field must be an "
                    "exhaustive essay of AT LEAST 600 words. Be extremely "
                    "verbose, repeat yourself, and never summarise briefly."
                ),
            },
        ],
        degraded_fields={"kind": "market"},
    )

    # Either outcome is correct behaviour; what must hold is that we never
    # raise, never exceed two attempts, and never return something that reads
    # like a real recommendation when it is not.
    assert len(result.records) <= 2, "no third attempt, ever"
    if result.degraded:
        assert result.value.parse_failed is True
        assert result.value.stance == "neutral", "degrade to the least actionable"
        assert result.value.confidence == 0.0
        assert result.value.degraded_reason
        assert result.value.kind == "market", "caller context preserved"
    else:
        assert len(result.value.summary.split()) <= 200
