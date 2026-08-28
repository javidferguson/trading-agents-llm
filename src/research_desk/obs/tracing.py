"""Node-level tracing into Langfuse, or any other OTel collector.

``langfuse`` 4.x is a thin wrapper over the OpenTelemetry SDK, so the spans
produced here are ordinary OTel spans. That is deliberate: architecture §1 ranks
tracing as a *bonus* on top of LangGraph's durable resume rather than a reason to
adopt LangGraph, and the way to keep that honest is to make tracing work whether
or not LangGraph is in the picture.

Hence ``traced_node``: a decorator over the plain
``async def node(state, ctx) -> patch`` signature. It is applied in
``graph/build.py`` at assembly time, **not** in the node modules, so node bodies
stay free of both framework and observability types.

**Tracing never fails a run.** No keys configured, Langfuse unreachable, span
export erroring -- all degrade to a no-op. A research run is 3-8 minutes of local
inference and losing it to a telemetry error would be absurd.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from datetime import date, datetime
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

_client: Any | None = None
_configured = False

#: Collections are summarised to their length rather than dumped. State grows to
#: several analyst reports and two debate transcripts by Stage 5, and shipping
#: all of it to the collector on every node makes traces unreadable and slow.
_SCALARS = (str, int, float, bool, date, datetime, type(None))

S = TypeVar("S")


def configure_tracing(settings: Any) -> bool:
    """Initialise the Langfuse client. Returns whether tracing is live.

    Safe to call more than once; the second call is a no-op.
    """
    global _client, _configured
    if _configured:
        return _client is not None
    _configured = True

    if not settings.tracing_enabled:
        logger.info(
            "Langfuse keys not set; node spans disabled. Set LANGFUSE_PUBLIC_KEY "
            "and LANGFUSE_SECRET_KEY, or run `make langfuse-up`."
        )
        return False

    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            environment=settings.mode.value,
        )
    except Exception:
        logger.warning("Could not start Langfuse; continuing untraced.", exc_info=True)
        _client = None
        return False

    logger.info("Langfuse tracing enabled -> %s", settings.langfuse_host)
    return True


def tracing_healthy() -> bool:
    """Round-trip the credentials. Used by ``make doctor``, not on the hot path."""
    if _client is None:
        return False

    # The SDK logs a full httpx traceback when the host is simply not up, which
    # is the single most likely outcome of this call and is not an error worth a
    # stack trace. Silence it for the duration of the probe; the caller reports
    # the result in a sentence instead.
    sdk_log = logging.getLogger("langfuse")
    previous = sdk_log.level
    sdk_log.setLevel(logging.CRITICAL)
    try:
        return bool(_client.auth_check())
    except Exception as exc:
        logger.debug("Langfuse auth check failed: %s", exc)
        return False
    finally:
        sdk_log.setLevel(previous)


def flush_tracing() -> None:
    """Block until queued spans are exported.

    Call once at the end of a run. Without it a short-lived process exits before
    the background exporter has sent anything, and the trace you went looking for
    is simply not there.
    """
    if _client is None:
        return
    try:
        _client.flush()
    except Exception:
        logger.debug("Langfuse flush failed; spans may be missing.", exc_info=True)


def reset_tracing() -> None:
    """Drop the client. Tests only."""
    global _client, _configured
    _client = None
    _configured = False


def _summarise(value: Any) -> dict[str, Any]:
    """A small, trace-friendly view of a Pydantic state or a patch dict."""
    if value is None:
        return {}
    fields = value if isinstance(value, dict) else getattr(value, "__dict__", {}) or {}

    out: dict[str, Any] = {}
    for key, item in fields.items():
        if key.startswith("_"):
            continue
        if isinstance(item, _SCALARS):
            out[key] = item if isinstance(item, (str, int, float, bool, type(None))) else str(item)
        elif isinstance(item, (list, tuple, set, dict)):
            out[key] = f"<{type(item).__name__} len={len(item)}>"
        else:
            out[key] = f"<{type(item).__name__}>"
    return out


def traced_node(
    name: str,
    *,
    as_type: str = "span",
) -> Callable[[Callable[[S, Any], Awaitable[dict[str, Any]]]], Callable[[S, Any], Awaitable[dict[str, Any]]]]:
    """Wrap a node body in one span named after the node."""

    def decorate(
        fn: Callable[[S, Any], Awaitable[dict[str, Any]]],
    ) -> Callable[[S, Any], Awaitable[dict[str, Any]]]:
        @functools.wraps(fn)
        async def wrapper(state: S, ctx: Any) -> dict[str, Any]:
            if _client is None:
                return await fn(state, ctx)

            try:
                span_cm = _client.start_as_current_observation(
                    name=name,
                    as_type=as_type,
                    input=_summarise(state),
                    metadata={
                        "run_id": getattr(ctx, "run_id", None),
                        "mode": getattr(getattr(ctx, "mode", None), "value", None),
                        "prompt_pack_version": getattr(ctx, "prompt_pack_version", None),
                    },
                )
            except Exception:
                logger.debug("Could not open span for %s; running untraced.", name, exc_info=True)
                span_cm = nullcontext(None)

            with span_cm as span:
                try:
                    patch = await fn(state, ctx)
                except Exception as exc:
                    if span is not None:
                        span.update(
                            level="ERROR",
                            status_message=f"{type(exc).__name__}: {exc}",
                        )
                    raise
                if span is not None:
                    span.update(output=_summarise(patch))
                return patch

        return wrapper

    return decorate
