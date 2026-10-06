"""Profiles -> provider clients. The only place a model call is actually made.

LangGraph orchestrates; it does not call models. A node body calls *this*,
never a ``langchain_core`` chat model -- that is where the abstraction tax
lives, and routing through it makes Ollama's ``format=<json_schema>`` awkward
for no gain (architecture §1, constraint 1).

Two things this module deliberately does **not** do:

* **No tool definitions, ever.** The paper's "~20 tool calls" are
  ``get_price_history``, ``get_news``, ``get_financials`` -- there is no
  judgment in choosing them, so they are prefetched in Python (§2).
* **No parsing.** It returns raw text plus accounting. Validation, the repair
  turn and degradation all live in ``structured.py``, so that the retry policy
  is in one readable place rather than spread across providers.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import Settings, load_yaml
from ..models.modes import RunMode

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """A model call failed in a way the caller cannot paper over."""


class ProfileNotConfigured(LLMError):
    """A node asked for routing that `config/models.yaml` does not define."""


@dataclass(frozen=True)
class ModelProfile:
    """One row of `profiles:` in models.yaml, resolved."""

    name: str
    provider: str
    model: str
    temperature: float = 0.3
    num_ctx: int | None = None
    max_tokens: int | None = None

    #: Hybrid reasoning models (qwen3 and friends) think by default, and it is
    #: NOT free. Measured on qwen3:8b against an identical prompt: thinking on
    #: produced 579 eval tokens in 8.7s; thinking off produced 64 in 2.7s --
    #: same answer, ~9x the tokens. Across 14 nodes that is the difference
    #: between a pipeline you iterate on and one you avoid running.
    #:
    #: Structured output stays valid either way; the reasoning comes back in a
    #: separate `thinking` field rather than contaminating the JSON. So this is
    #: purely a cost/quality dial, defaulted to off and raised deliberately for
    #: the nodes where reasoning is the product (the facilitator, the trader).
    think: bool = False

    prompt_caching: bool = False

    @classmethod
    def from_dict(cls, name: str, body: dict[str, Any]) -> "ModelProfile":
        missing = {"provider", "model"} - set(body)
        if missing:
            raise ProfileNotConfigured(
                f"profile {name!r} in models.yaml is missing {sorted(missing)}"
            )
        return cls(
            name=name,
            provider=body["provider"],
            model=body["model"],
            temperature=float(body.get("temperature", 0.3)),
            num_ctx=body.get("num_ctx"),
            max_tokens=body.get("max_tokens"),
            think=bool(body.get("think", False)),
            prompt_caching=bool(body.get("prompt_caching", False)),
        )


@dataclass
class LLMResponse:
    """What came back, plus everything `LLMCallRecord` needs.

    ``text`` is the raw body before any parsing. Keep it that way -- it is the
    substrate `mode="replay_llm"` feeds back in, and it cannot be reconstructed
    from the parsed object.
    """

    text: str
    model: str
    profile: str
    provider: str
    thinking: str | None = None
    model_digest: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int = 0
    usd: float = 0.0


class OllamaClient:
    """Local models over Ollama's /api/chat, with schema-constrained decode."""

    def __init__(self, base_url: str, timeout_s: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)
        #: model name -> digest, fetched once. Two runs of "qwen3:8b" are not
        #: necessarily the same weights, and the digest is what makes a cache
        #: key or a Stage 8 comparison honest.
        self._digests: dict[str, str] | None = None

    async def _digest(self, model: str) -> str | None:
        if self._digests is None:
            try:
                response = await self._client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                self._digests = {
                    m.get("model", ""): m.get("digest", "")
                    for m in response.json().get("models", [])
                }
            except Exception:
                # Never fail a run over provenance metadata.
                logger.debug("Could not read model digests from Ollama.", exc_info=True)
                self._digests = {}
        return self._digests.get(model) or None

    async def complete(
        self,
        profile: ModelProfile,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        options: dict[str, Any] = {"temperature": profile.temperature}
        if profile.num_ctx:
            options["num_ctx"] = profile.num_ctx
        if profile.max_tokens:
            options["num_predict"] = profile.max_tokens

        payload: dict[str, Any] = {
            "model": profile.model,
            "messages": messages,
            "stream": False,
            "think": profile.think,
            "options": options,
        }
        if schema is not None:
            # The whole basis of the "no tool-calling" rule: constrain the
            # decode rather than asking politely for JSON and hoping.
            payload["format"] = schema

        started = time.monotonic()
        try:
            response = await self._client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            # A 404 here is nearly always a model that was never pulled, and
            # the raw message does not say so.
            hint = (
                f"  Is it pulled?  ollama pull {profile.model}"
                if exc.response.status_code == 404
                else ""
            )
            raise LLMError(
                f"Ollama returned {exc.response.status_code} for {profile.model}: "
                f"{detail}{hint}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Could not reach Ollama at {self.base_url}: {exc}. "
                "Is the daemon running?"
            ) from exc

        elapsed_ms = int((time.monotonic() - started) * 1000)
        message = body.get("message") or {}

        return LLMResponse(
            text=message.get("content") or "",
            thinking=message.get("thinking") or None,
            model=body.get("model", profile.model),
            profile=profile.name,
            provider="ollama",
            model_digest=await self._digest(profile.model),
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
            latency_ms=elapsed_ms,
            usd=0.0,  # local inference is free; the wall clock is the cost
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class AnthropicClient:
    """The hosted path. **Stage 5**, deliberately not written yet.

    `fund_manager` routes to `deep_hosted` in models.yaml, so this profile is
    reachable by configuration today. Rather than ship plausible-looking
    untested API code that would first run against a real bill, it fails with
    a message naming the stage. Anything routed here before Stage 5 is a
    routing mistake and should be loud.
    """

    def __init__(self, api_key: str | None, timeout_s: float = 120.0):
        self._api_key = api_key

    async def complete(
        self,
        profile: ModelProfile,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        raise LLMError(
            f"profile {profile.name!r} routes to the hosted provider, which "
            "arrives at Stage 5 along with budget enforcement and prompt "
            "caching. Until then, run with the `all_local` preset:\n"
            "    LLM_PRESET=all_local desk ...\n"
            "or point this node at a local profile in config/models.yaml."
        )

    async def aclose(self) -> None:
        return None


class LLMRouter:
    """Resolves node -> profile -> provider client.

    Per-node routing is a dict and was always going to be a dict; the value is
    in having exactly one of them, loaded from config rather than scattered
    across node modules.
    """

    def __init__(
        self,
        profiles: dict[str, ModelProfile],
        node_routes: dict[str, str],
        clients: dict[str, Any],
        *,
        preset: str | None = None,
        presets: dict[str, dict[str, str]] | None = None,
    ):
        self.profiles = profiles
        self.node_routes = node_routes
        self.clients = clients
        self.preset = preset
        self._presets = presets or {}

    @classmethod
    def from_config(
        cls,
        settings: Settings,
        *,
        config: dict[str, Any] | None = None,
        preset: str | None = None,
    ) -> "LLMRouter":
        body = config if config is not None else load_yaml("models.yaml")

        profiles = {
            name: ModelProfile.from_dict(name, spec)
            for name, spec in (body.get("profiles") or {}).items()
        }
        if not profiles:
            raise ProfileNotConfigured("models.yaml defines no profiles")

        presets = body.get("presets") or {}
        if preset and preset not in presets:
            raise ProfileNotConfigured(
                f"unknown preset {preset!r}; models.yaml has {sorted(presets)}"
            )

        provider_cfg = body.get("providers") or {}
        ollama_cfg = provider_cfg.get("ollama") or {}
        clients: dict[str, Any] = {
            # base_url comes from Settings, not from the YAML's "${OLLAMA_BASE_URL}"
            # placeholder -- env expansion inside config files is a second
            # configuration language, and one is enough.
            "ollama": OllamaClient(
                settings.ollama_base_url,
                timeout_s=float(ollama_cfg.get("timeout_s", settings.ollama_timeout_s)),
            ),
            "anthropic": AnthropicClient(settings.anthropic_api_key),
        }

        return cls(
            profiles=profiles,
            node_routes=dict(body.get("nodes") or {}),
            clients=clients,
            preset=preset,
            presets=presets,
        )

    def profile_for(self, node: str) -> ModelProfile:
        """Which profile serves this node, honouring an active preset.

        A preset is a whole-run override for ablations -- `all_local` for a
        free run, `all_hosted` for strategy development. Stage 8 flips these,
        so resolution has to be one lookup rather than an edit.
        """
        if self.preset:
            mapping = self._presets.get(self.preset, {})
            name = mapping.get(node) or mapping.get("*")
            if name:
                if name not in self.profiles:
                    raise ProfileNotConfigured(
                        f"preset {self.preset!r} maps {node!r} to unknown "
                        f"profile {name!r}"
                    )
                return self.profiles[name]

        name = self.node_routes.get(node)
        if name is None:
            raise ProfileNotConfigured(
                f"node {node!r} has no entry under `nodes:` in models.yaml. "
                f"Known nodes: {sorted(self.node_routes)}"
            )
        if name not in self.profiles:
            raise ProfileNotConfigured(
                f"node {node!r} routes to profile {name!r}, which models.yaml "
                f"does not define. Known profiles: {sorted(self.profiles)}"
            )
        return self.profiles[name]

    async def complete(
        self,
        node: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        profile = self.profile_for(node)
        client = self.clients.get(profile.provider)
        if client is None:
            raise ProfileNotConfigured(
                f"profile {profile.name!r} names provider {profile.provider!r}, "
                f"which is not configured. Known: {sorted(self.clients)}"
            )
        return await client.complete(profile, messages, schema)

    async def aclose(self) -> None:
        for client in self.clients.values():
            await client.aclose()
