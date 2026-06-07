"""Langfuse client factory + per-request metrics, with graceful no-op fallback.

Design goals
------------
1. **Never fail a request.** Observability is a side channel. Missing
   credentials, an uninstalled ``langfuse`` package, or an unreachable Langfuse
   server must all degrade to a silent no-op — the extraction still returns.
2. **No cost tracking.** The model is self-hosted; a per-token dollar figure
   would be misleading, so we deliberately omit it (see ROADMAP Phase 9).
3. **CPU/CI-safe imports.** ``langfuse`` is an optional dependency and is
   imported lazily inside :func:`_build_client`, so this module imports cleanly
   even when it is not installed.

Configuration (all read from the environment):

================================  ===========================================
Variable                          Meaning
================================  ===========================================
``LANGFUSE_PUBLIC_KEY``           Project public key (``pk-lf-...``).
``LANGFUSE_SECRET_KEY``           Project secret key (``sk-lf-...``).
``LANGFUSE_HOST``                 Base URL of the Langfuse instance
                                  (e.g. ``https://cloud.langfuse.com`` or a
                                  self-hosted URL). Optional; the client falls
                                  back to its own default when unset.
================================  ===========================================

If **either** key is missing we run in no-op mode and log a single warning.

Usage::

    obs = get_observability()
    metrics = RequestMetrics(input_length_chars=len(text))
    # ... run inference, fill metrics in ...
    obs.trace_extraction(metrics=metrics)
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Langfuse trace/span name used for every extraction request.
TRACE_NAME = "contract-extraction"

# How many characters of the input contract to attach to a trace. Full contracts
# can be very large (8000-token budget) and may contain sensitive text, so we
# only attach a short preview by default.
INPUT_PREVIEW_CHARS = 500


@dataclass
class RequestMetrics:
    """The per-request metrics tracked for one extraction.

    All numeric fields are optional so a caller can populate only what a given
    code path can measure (e.g. the synchronous endpoint cannot observe
    time-to-first-token, while the streaming endpoint can). Only non-``None``
    values are sent to Langfuse.

    No cost field by design — the model is self-hosted (see ROADMAP Phase 9).
    """

    input_length_chars: int
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    inference_time_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None

    def compute_throughput(self) -> None:
        """Derive ``tokens_per_second`` from output tokens + inference time.

        No-op when either input is missing or the elapsed time is zero, so it is
        always safe to call. Idempotent — recomputes from current values.
        """
        if (
            self.output_tokens is not None
            and self.inference_time_ms is not None
            and self.inference_time_ms > 0
        ):
            self.tokens_per_second = self.output_tokens / (self.inference_time_ms / 1000.0)

    def as_dict(self) -> dict[str, Any]:
        """Return only the populated (non-``None``) metrics."""
        return {k: v for k, v in asdict(self).items() if v is not None}


def _read_config(env: Optional[dict] = None) -> dict[str, Optional[str]]:
    """Read the three ``LANGFUSE_*`` variables from the environment."""
    env = os.environ if env is None else env
    return {
        "public_key": env.get("LANGFUSE_PUBLIC_KEY") or None,
        "secret_key": env.get("LANGFUSE_SECRET_KEY") or None,
        "host": env.get("LANGFUSE_HOST") or None,
    }


def _build_client(config: dict[str, Optional[str]]) -> Optional[Any]:
    """Construct a Langfuse client, or return ``None`` for no-op mode.

    Returns ``None`` (and logs one warning) when credentials are incomplete or
    the ``langfuse`` package is not installed. Any unexpected construction error
    is also swallowed into no-op mode — observability must never break startup.
    """
    if not (config["public_key"] and config["secret_key"]):
        logger.warning(
            "Langfuse credentials not set (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY); "
            "running with observability disabled (no-op)."
        )
        return None

    try:
        from langfuse import Langfuse  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "langfuse is not installed; running with observability disabled (no-op). "
            "Install it with `pip install langfuse` to enable tracing."
        )
        return None

    kwargs: dict[str, Any] = {
        "public_key": config["public_key"],
        "secret_key": config["secret_key"],
    }
    if config["host"]:
        kwargs["host"] = config["host"]

    try:
        client = Langfuse(**kwargs)
        logger.info(
            "Langfuse observability enabled (host=%s).",
            config["host"] or "default",
        )
        return client
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to construct Langfuse client (%s); observability disabled.", exc)
        return None


class Observability:
    """Thin wrapper over a Langfuse client that is safe to call unconditionally.

    When ``client`` is ``None`` every method is a no-op. When a client is
    present, :meth:`trace_extraction` records one trace per request; any error
    raised by the Langfuse SDK is caught and logged so a tracing failure can
    never surface as a request failure.
    """

    def __init__(self, client: Optional[Any] = None) -> None:
        self._client = client

    @property
    def enabled(self) -> bool:
        """True when a real Langfuse client is attached."""
        return self._client is not None

    def trace_extraction(
        self,
        metrics: RequestMetrics,
        *,
        input_preview: Optional[str] = None,
        output_preview: Optional[str] = None,
        status: str = "ok",
    ) -> None:
        """Record one extraction request as a Langfuse trace.

        Derives ``tokens_per_second`` if not already set, then emits the trace
        with the populated metrics as ``metadata``. A silent no-op when
        observability is disabled. Never raises.
        """
        metrics.compute_throughput()
        if self._client is None:
            return

        payload = metrics.as_dict()
        try:
            self._emit(payload, input_preview, output_preview, status)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Langfuse trace emit failed (%s); ignoring.", exc)

    def _emit(
        self,
        metrics: dict[str, Any],
        input_preview: Optional[str],
        output_preview: Optional[str],
        status: str,
    ) -> None:
        """Send the trace to Langfuse.

        Supports both the modern (``start_span``) and legacy (``trace``) SDK
        surfaces so a version bump on the optional dependency does not silently
        stop emitting. Wrapped by :meth:`trace_extraction`'s try/except.
        """
        client = self._client
        trace_input = {"contract_preview": input_preview} if input_preview is not None else None
        trace_output = (
            {"extraction_preview": output_preview, "status": status}
            if output_preview is not None
            else {"status": status}
        )

        # Modern SDK (>=2.x): context-managed span carrying metadata.
        start_span = getattr(client, "start_span", None)
        if callable(start_span):
            span = start_span(name=TRACE_NAME, input=trace_input, metadata=metrics)
            try:
                update = getattr(span, "update", None)
                if callable(update):
                    update(output=trace_output)
            finally:
                end = getattr(span, "end", None)
                if callable(end):
                    end()
            self.flush()
            return

        # Legacy SDK: top-level trace(...) call.
        legacy_trace = getattr(client, "trace", None)
        if callable(legacy_trace):
            legacy_trace(
                name=TRACE_NAME,
                input=trace_input,
                output=trace_output,
                metadata=metrics,
            )
            self.flush()

    def flush(self) -> None:
        """Flush buffered events to Langfuse. No-op/safe when disabled."""
        if self._client is None:
            return
        flush = getattr(self._client, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Langfuse flush failed (%s); ignoring.", exc)


# Process-wide singleton, built lazily on first use.
_OBSERVABILITY: Optional[Observability] = None


def get_observability(env: Optional[dict] = None) -> Observability:
    """Return the process-wide :class:`Observability`, building it once.

    Reads the ``LANGFUSE_*`` environment on first call. Pass ``env`` to build
    from an explicit mapping (used in tests). Always returns a usable handle —
    a disabled one when credentials/SDK are unavailable.
    """
    global _OBSERVABILITY
    if _OBSERVABILITY is None:
        _OBSERVABILITY = Observability(_build_client(_read_config(env)))
    return _OBSERVABILITY


def reset_observability() -> None:
    """Drop the cached singleton so the next call rebuilds it (test hook)."""
    global _OBSERVABILITY
    _OBSERVABILITY = None
