"""Observability. Tracing is a bonus on top of the design, never load-bearing."""

from __future__ import annotations

from .tracing import configure_tracing, flush_tracing, traced_node, tracing_healthy

__all__ = ["configure_tracing", "flush_tracing", "traced_node", "tracing_healthy"]
