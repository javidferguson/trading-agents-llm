"""The repair turn, degradation, and call accounting.

All offline against a scripted client. The point of these is the *policy* --
how many attempts, what gets recorded, what comes back when the model never
complies -- which must hold identically whatever the model does. The live
Stage 1 exit gate is `tests/llm/test_smoke_ollama.py`.
"""

from __future__ import annotations

import json

import pytest

from research_desk.llm.router import LLMError, LLMResponse, LLMRouter, ModelProfile
from research_desk.llm.structured import schema_for, structured
from research_desk.models.state import AnalystReport

PROFILE = ModelProfile(name="quick", provider="fake", model="fake-8b", temperature=0.3)

VALID = {
    "kind": "market",
    "stance": "bullish",
    "confidence": 0.7,
    "summary": "Trend intact, momentum stretched.",
    "key_points": ["above 200d", "RSI 71", "52w percentile 0.94"],
    "evidence": [],
    "data_gaps": [],
}


class ScriptedClient:
    """Returns a prepared reply per call, and records what it was asked."""

    def __init__(self, replies: list[str | Exception]):
        self._replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []
        self.schemas: list[dict | None] = []

    async def complete(self, profile, messages, schema=None):
        self.calls.append([dict(m) for m in messages])
        self.schemas.append(schema)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(
            text=reply,
            model=profile.model,
            profile=profile.name,
            provider=profile.provider,
            thinking="some reasoning",
            model_digest="deadbeef",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=1234,
        )

    async def aclose(self):
        return None


def make_router(replies) -> tuple[LLMRouter, ScriptedClient]:
    client = ScriptedClient(replies)
    router = LLMRouter(
        profiles={"quick": PROFILE},
        node_routes={"market_analyst": "quick"},
        clients={"fake": client},
    )
    return router, client


async def _run(replies, **kwargs):
    router, client = make_router(replies)
    result = await structured(
        router, "market_analyst", AnalystReport,
        [{"role": "user", "content": "read SPY"}],
        degraded_fields={"kind": "market"},
        **kwargs,
    )
    return result, client


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


async def test_a_valid_first_response_costs_one_call() -> None:
    result, client = await _run([json.dumps(VALID)])

    assert not result.degraded
    assert result.value.stance == "bullish"
    assert result.attempts == 1
    assert len(client.calls) == 1, "no repair turn should be spent on a good answer"


async def test_the_decode_is_schema_constrained() -> None:
    """Not 'please return JSON'. The sampler is constrained (§2)."""
    _, client = await _run([json.dumps(VALID)])

    sent = client.schemas[0]
    assert sent is not None
    assert sent["type"] == "object"
    assert "stance" in sent["properties"]


async def test_internal_fields_are_never_shown_to_the_model() -> None:
    """A model must not be able to mark its own output degraded -- or clean."""
    _, client = await _run([json.dumps(VALID)])

    sent = client.schemas[0]
    assert "parse_failed" not in sent["properties"]
    assert "degraded_reason" not in sent["properties"]
    assert "parse_failed" not in sent.get("required", [])


def test_developer_docstrings_do_not_leak_into_the_prompt() -> None:
    """Pydantic fills `description` from the class docstring.

    Ours talks about stages and repair turns. That is maintenance prose, not
    instruction, and it has no business in a model's context window.
    """
    sent = schema_for(AnalystReport)
    blob = json.dumps(sent)
    assert "description" not in sent
    assert "Stage 1" not in blob
    assert "repair turn" not in blob


# --------------------------------------------------------------------------- #
# The repair turn
# --------------------------------------------------------------------------- #


async def test_a_schema_violation_triggers_exactly_one_repair() -> None:
    """confidence=1.7 is valid JSON and valid per the loose schema, but not per
    Pydantic. This is the ordinary case, not an exotic one."""
    bad = json.dumps({**VALID, "confidence": 1.7})
    result, client = await _run([bad, json.dumps(VALID)])

    assert not result.degraded
    assert result.value.confidence == 0.7
    assert len(client.calls) == 2, "exactly one repair turn"
    assert result.attempts == 2


async def test_the_repair_prompt_carries_the_validation_error() -> None:
    """Telling the model only 'that was wrong' wastes the retry."""
    bad = json.dumps({**VALID, "confidence": 1.7})
    _, client = await _run([bad, json.dumps(VALID)])

    repair = client.calls[1]
    assert repair[-2]["role"] == "assistant", "the bad output must be in the history"
    assert repair[-2]["content"] == bad
    instruction = repair[-1]["content"]
    assert "confidence" in instruction, "the error must name the offending field"
    assert "less than or equal to 1" in instruction
    assert "COMPLETE corrected JSON" in instruction


async def test_the_word_limit_is_enforced_by_us_not_by_the_schema() -> None:
    """A 200-word cap cannot be expressed in JSON schema, so the validator
    catches it -- which is exactly what keeps the repair turn a live path."""
    windy = json.dumps({**VALID, "summary": "word " * 250})
    result, client = await _run([windy, json.dumps(VALID)])

    assert len(client.calls) == 2
    assert not result.degraded


async def test_length_errors_ask_for_structure_not_arithmetic() -> None:
    """§2 ("never let an LLM do arithmetic") applies to the constraints we hand
    a model, not only to the numbers we ask it for.

    Measured: told "summary is 412 words; the limit is 200", qwen3:8b returned
    304 -- it shortened and still failed, burning the only retry. The fix is to
    state the requirement structurally and to ask for margin.
    """
    windy = json.dumps({**VALID, "summary": "word " * 250})
    _, client = await _run([windy, json.dumps(VALID)])

    instruction = client.calls[1][-1]["content"]
    assert "AT MOST 6 short sentences" in instruction, "give a countable target"
    assert "go clearly inside it" in instruction, "ask for margin, not the limit"


def test_ollama_owns_shape_and_pydantic_owns_values() -> None:
    """The division of labour, asserted so it cannot drift back.

    Shape keywords a grammar is good at are kept; every value-level constraint
    is stripped and left to Pydantic. Measured justification: `minItems` made
    qwen3:8b pad a short list with 'key_points_count' to satisfy the count,
    and `maximum` was never enforced at all. A constraint the grammar
    half-enforces is worse than one it does not enforce, because the failure
    stops being visible.
    """
    sent = json.dumps(schema_for(AnalystReport))

    # Shape: kept.
    assert '"enum"' in sent, "enums work well and are worth constraining"
    assert "$defs" in sent, "nested models resolve correctly"
    assert '"required"' in sent
    assert '"type"' in sent

    # Values: Pydantic's job.
    for keyword in ("minItems", "maxItems", "minimum", "maximum",
                    "minLength", "maxLength", "pattern", "exclusiveMinimum"):
        assert keyword not in sent, f"{keyword} must not reach the model"


def test_value_constraints_are_stripped_inside_nested_defs() -> None:
    """A constraint hiding in $defs is still a constraint.

    Evidence.excerpt has max_length=300; stripping only the top level would
    leave it in and reintroduce exactly the half-enforcement being removed.
    """
    schema = schema_for(AnalystReport)
    evidence = schema["$defs"]["Evidence"]
    assert "maxLength" not in json.dumps(evidence)
    # ...and Pydantic still enforces it.
    with pytest.raises(Exception):
        AnalystReport.model_validate({
            **VALID,
            "evidence": [{"source": "x", "as_of": "2026-10-06T00:00:00",
                          "excerpt": "y" * 400}],
        })


async def test_whitespace_padded_key_points_are_rejected() -> None:
    """Observed from Ollama: it enforces minItems by padding with '\n\n'.

    The grammar is satisfied and the report is junk. Schema-valid is not the
    same as usable.
    """
    padded = json.dumps({**VALID, "key_points": ["real point", "another", "\n\n"]})
    result, client = await _run([padded, json.dumps(VALID)])

    assert len(client.calls) == 2, "padding must trigger the repair turn"
    assert "empty or whitespace" in client.calls[1][-1]["content"]
    assert not result.degraded


async def test_non_json_output_also_repairs() -> None:
    result, client = await _run(["I think SPY looks strong.", json.dumps(VALID)])

    assert len(client.calls) == 2
    assert not result.degraded


# --------------------------------------------------------------------------- #
# Degradation -- architecture §5, failure direction is always the safe one
# --------------------------------------------------------------------------- #


async def test_two_failures_degrade_rather_than_raise() -> None:
    """A degraded report beats a crashed 8-minute run."""
    bad = json.dumps({**VALID, "confidence": 9.0})
    result, client = await _run([bad, bad])

    assert result.degraded
    assert result.value.parse_failed is True
    assert result.value.degraded_reason
    assert len(client.calls) == 2, "no third attempt, ever"


async def test_the_degraded_report_is_the_least_actionable_one() -> None:
    """Neutral and zero confidence. A degraded report that still reads as a
    mild recommendation is worse than a crash."""
    bad = json.dumps({**VALID, "confidence": 9.0})
    result, _ = await _run([bad, bad])

    assert result.value.stance == "neutral"
    assert result.value.confidence == 0.0
    assert "unavailable" in " ".join(result.value.data_gaps)


async def test_degraded_fields_carry_the_callers_context() -> None:
    """A degraded news report and a degraded market report differ downstream."""
    router, _ = make_router(["nonsense", "nonsense"])
    result = await structured(
        router, "market_analyst", AnalystReport,
        [{"role": "user", "content": "x"}],
        degraded_fields={"kind": "news"},
    )
    assert result.value.kind == "news"


async def test_a_transport_failure_degrades_without_a_repair_turn() -> None:
    """There is no raw text to repair, so retrying the parse is pointless."""
    result, client = await _run([LLMError("connection refused")])

    assert result.degraded
    assert len(client.calls) == 1
    assert result.records == [], "a call that never returned produced no tokens"
    assert "connection refused" in result.value.degraded_reason


# --------------------------------------------------------------------------- #
# Accounting
# --------------------------------------------------------------------------- #


async def test_failed_attempts_are_recorded_too() -> None:
    """They burned tokens and wall-clock exactly like successful ones.

    Recording only successes would make the Stage 5 budget check and the §12
    cost table quietly optimistic, and a budget that under-counts is worse
    than no budget.
    """
    bad = json.dumps({**VALID, "confidence": 1.7})
    result, _ = await _run([bad, json.dumps(VALID)])

    assert len(result.records) == 2
    assert [r.attempt for r in result.records] == [1, 2]
    assert [r.parse_failed for r in result.records] == [True, False]
    assert sum(r.completion_tokens for r in result.records) == 100


async def test_raw_response_is_captured_from_the_very_first_call() -> None:
    """`replay_llm` is worthless retroactively."""
    result, _ = await _run([json.dumps(VALID)])

    raw = result.records[0].raw_response
    assert raw == json.dumps(VALID)
    # Round-trips: the record alone is enough to reproduce the parse.
    assert AnalystReport.model_validate_json(raw).stance == "bullish"


async def test_reasoning_tokens_are_kept_separately() -> None:
    result, _ = await _run([json.dumps(VALID)])
    assert result.records[0].thinking == "some reasoning"


async def test_the_prompt_hash_covers_schema_and_profile_not_just_prose() -> None:
    """Two calls with the same words but a different schema are different calls."""
    from research_desk.llm.structured import _prompt_hash

    messages = [{"role": "user", "content": "same words"}]
    base = _prompt_hash(messages, {"type": "object"}, "quick")
    assert base != _prompt_hash(messages, {"type": "string"}, "quick")
    assert base != _prompt_hash(messages, {"type": "object"}, "deep_local")
    assert base == _prompt_hash(messages, {"type": "object"}, "quick")
