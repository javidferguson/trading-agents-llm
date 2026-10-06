"""Schema-constrained decode, one repair turn, then degrade. Never raise upward.

Architecture §6: *"`structured()` handles: schema-constrained decode -> validate
-> on failure, one repair turn with the validation error appended -> on second
failure, return `schema.degraded()` with `parse_failed=True`. A degraded
analyst report beats a crashed 8-minute run."*

Three design points that are easy to get wrong and expensive to fix later.

**The repair turn is built now, at Stage 1, on purpose.** It is the piece that
gets deferred as "we'll add retries if we need them" and is then desperately
needed at Stage 4, when a four-way fan-out is failing intermittently and you
are trying to debate prompt quality at the same time. It costs ~40 lines here.

**Every attempt is recorded, including the ones that failed.** A failed call
burned tokens and wall-clock exactly like a successful one. Recording only the
successful attempt would make the Stage 5 budget check and the §12 cost table
quietly optimistic, and a budget that under-counts is worse than none.

**Degradation is never silent.** The fallback is marked `parse_failed=True`
with a reason, and the caller gets the records. Architecture §5's rule --
failure direction is always HOLD -- only works if downstream can tell a
degraded report from a genuinely neutral one.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import ValidationError

from ..models.state import Degradable, LLMCallRecord
from .router import LLMError, LLMRouter

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Degradable)

#: Fields `Degradable` adds for our own bookkeeping. They are stripped from the
#: schema handed to the model: a model must never be able to declare its own
#: output degraded, nor -- far worse -- declare a degraded output fine.
INTERNAL_FIELDS = ("parse_failed", "degraded_reason")

#: How much of a validation error to paste into the repair prompt. Pydantic is
#: verbose, and a 2kB error pushes the original request out of a small model's
#: attention just when it needs it most.
MAX_ERROR_CHARS = 800


@dataclass
class StructuredResult:
    """The parsed object, plus what it cost to get it."""

    value: Any
    records: list[LLMCallRecord] = field(default_factory=list)
    #: True when the value is `schema.degraded(...)` rather than model output.
    degraded: bool = False

    @property
    def usd(self) -> float:
        return sum(r.usd for r in self.records)

    @property
    def attempts(self) -> int:
        return len(self.records)


def schema_for(model: type[Degradable]) -> dict[str, Any]:
    """The JSON schema to constrain the decode with.

    Two edits to what Pydantic generates, both about what the *model* should
    see rather than what is technically correct:

    * ``parse_failed`` / ``degraded_reason`` are removed. They are ours.
    * ``title`` and ``description`` are removed. Pydantic fills ``description``
      from the class docstring, and our docstrings are written for whoever
      maintains this -- shipping "Stage 1's smoke-test target" into a prompt is
      noise at best and misdirection at worst. Field-level descriptions written
      *for* a model would be worth keeping; developer prose is not.

    ``$defs`` and ``$ref`` are left alone: Ollama resolves them correctly,
    including nested models, which was verified before this was written.
    """
    schema = model.model_json_schema()
    properties = schema.get("properties", {})
    for name in INTERNAL_FIELDS:
        properties.pop(name, None)
    if "required" in schema:
        schema["required"] = [r for r in schema["required"] if r not in INTERNAL_FIELDS]
    schema.pop("title", None)
    schema.pop("description", None)
    return schema


def _prompt_hash(messages: list[dict[str, str]], schema: dict[str, Any], profile: str) -> str:
    """Identifies the whole request, not just the prose.

    The schema and the profile change the output as surely as the prompt does,
    so a hash that ignored them would collide across genuinely different calls
    and make the Stage 8 "are these two runs comparable" check a lie.
    """
    blob = json.dumps(
        {"messages": messages, "schema": schema, "profile": profile},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _trim(text: str, limit: int = MAX_ERROR_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " ...(truncated)"


def _repair_prompt(bad_output: str, error: str) -> str:
    """What we say on the second and final attempt.

    Written as an instruction to a small model rather than as a bug report:
    state the fault, restate the requirement, and ask for the whole object
    back. Asking for a patch or a diff reliably confuses an 8B model.
    """
    return (
        "Your previous response did not satisfy the required schema.\n\n"
        f"Validation error:\n{_trim(error)}\n\n"
        "Return the COMPLETE corrected JSON object. Fix only what the error "
        "describes and keep every other field as you had it. Do not explain "
        "the fix, do not apologise, and do not wrap the JSON in markdown.\n"
        # Models overshoot limits rather than clearing them. Measured: asked to
        # cut a 412-word summary to 200, qwen3:8b returned 304 -- it shortened
        # and still failed, burning the only retry available. Asking for margin
        # costs nothing and converts a near-miss into a pass.
        "If the error is about a length, a count or a range, do not aim at the "
        "limit -- go clearly inside it.\n"
        # The repair turn competes with the original request, and without this
        # it loses. Measured: asked for confidence "as a PERCENTAGE", qwen3:8b
        # returned 85; told it must be <= 1, it returned 82. It was still
        # obeying the earlier, more specific instruction, because a Pydantic
        # error states a constraint without saying what to do about it. The
        # same thing happens for real whenever a system prompt and a schema
        # disagree, so the correction has to claim priority explicitly.
        "This correction OVERRIDES any earlier instruction in this conversation "
        "that conflicts with it, including any about formatting, scale or units. "
        "The schema is the authority. If a value is out of range, convert it to "
        "the required scale rather than nudging it."
    )


async def structured(
    router: LLMRouter,
    node: str,
    schema: type[T],
    messages: list[dict[str, str]],
    *,
    degraded_fields: dict[str, Any] | None = None,
) -> StructuredResult:
    """Get a validated ``schema`` instance from ``node``'s model. Never raises.

    ``degraded_fields`` carries what the caller knows but the schema cannot
    default -- an ``AnalystReport`` still needs its ``kind``, because a degraded
    market report and a degraded news report are different facts downstream.
    """
    json_schema = schema_for(schema)
    profile = router.profile_for(node)
    prompt_hash = _prompt_hash(messages, json_schema, profile.name)
    overrides = degraded_fields or {}

    records: list[LLMCallRecord] = []
    conversation = list(messages)
    last_error = "unknown"

    # Attempt 1 is the real one; attempt 2 is the repair turn. There is no
    # attempt 3 -- a model that cannot satisfy a schema twice is not going to
    # on the third try, and a 14-node run cannot afford open-ended retries.
    for attempt in (1, 2):
        try:
            response = await router.complete(node, conversation, json_schema)
        except LLMError as exc:
            # Transport failure, not a parse failure: there is no raw text to
            # record and nothing a repair turn could fix.
            last_error = str(exc)
            logger.warning("%s: model call failed on attempt %d: %s", node, attempt, exc)
            break

        records.append(
            LLMCallRecord(
                node=node,
                profile=profile.name,
                provider=profile.provider,
                model=response.model,
                model_digest=response.model_digest,
                prompt_hash=prompt_hash,
                raw_response=response.text,
                thinking=response.thinking,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_ms=response.latency_ms,
                usd=response.usd,
                attempt=attempt,
                parse_failed=False,  # provisional; corrected just below
            )
        )

        try:
            value = schema.model_validate_json(response.text)
        except ValidationError as exc:
            records[-1].parse_failed = True
            last_error = str(exc)
            logger.info(
                "%s: schema validation failed on attempt %d (%d error(s))",
                node, attempt, exc.error_count(),
            )
            if attempt == 1:
                conversation = [
                    *conversation,
                    {"role": "assistant", "content": response.text},
                    {"role": "user", "content": _repair_prompt(response.text, last_error)},
                ]
                continue
            break
        except ValueError as exc:
            # Not even JSON. Constrained decode makes this rare, but a model
            # that hit its token limit mid-object lands here.
            records[-1].parse_failed = True
            last_error = f"response was not valid JSON: {exc}"
            if attempt == 1:
                conversation = [
                    *conversation,
                    {"role": "assistant", "content": response.text},
                    {"role": "user", "content": _repair_prompt(response.text, last_error)},
                ]
                continue
            break

        return StructuredResult(value=value, records=records, degraded=False)

    logger.warning("%s: degrading after %d attempt(s)", node, len(records))
    return StructuredResult(
        value=schema.degraded(_trim(last_error, 300), **overrides),
        records=records,
        degraded=True,
    )
