"""Observability for the contract extractor serving layer.

Phase 9 — per-request tracing of latency, throughput, and token counts via
`Langfuse <https://langfuse.com/>`_. There is intentionally **no cost
tracking**: the model is self-hosted, so a per-token dollar figure would be
misleading.

The single public entry point is :func:`extractor.observability.langfuse_setup.get_observability`,
which returns a process-wide :class:`~extractor.observability.langfuse_setup.Observability`
handle. It degrades to a silent no-op when Langfuse credentials are absent or
the ``langfuse`` package is not installed, so importing this package and tracing
through it is always safe on a CPU box or in CI.
"""

from extractor.observability.langfuse_setup import (
    Observability,
    RequestMetrics,
    get_observability,
    reset_observability,
)

__all__ = [
    "Observability",
    "RequestMetrics",
    "get_observability",
    "reset_observability",
]
