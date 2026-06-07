"""FastAPI service for the fine-tuned contract extractor.

Endpoints
---------
- ``GET  /health``         — liveness + whether the model is loaded.
- ``POST /extract``        — sync extraction; returns :class:`ExtractResponse`.
- ``POST /extract/stream`` — Server-Sent Events of partial generation.

Design
------
The model is loaded **once** at startup into ``app.state.generator`` and reused
across requests. Loading is GPU-only (Unsloth + bitsandbytes), so:

- Set ``EXTRACTOR_SKIP_MODEL_LOAD=1`` to start the app without loading a model
  (used by the test suite and for importing the app on a CPU box). ``/extract``
  then returns 503 until a generator is present.
- If the load fails (e.g. no GPU, missing adapter), the app still starts and
  ``/health`` reports ``model_loaded: false`` rather than crashing.

Tests inject a mock generator via FastAPI's dependency override on
:func:`get_generator`, so CI never needs a GPU.

Run::

    uvicorn extractor.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from extractor.inference.model_loader import DEFAULT_ADAPTER_PATH
from extractor.inference.prompt import build_messages
from extractor.observability import RequestMetrics, get_observability
from extractor.observability.langfuse_setup import INPUT_PREVIEW_CHARS
from extractor.schemas import ContractExtraction, ExtractRequest, ExtractResponse

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 2048


def _count_input_tokens(generator: Any, messages: list[dict]) -> int | None:
    """Best-effort prompt token count via the generator's tokenizer.

    Returns ``None`` when the generator exposes no tokenizer (e.g. the mocked
    generator used in tests) or when tokenization fails — observability metrics
    are optional and must never break a request.
    """
    tokenizer = getattr(generator, "tokenizer", None)
    if tokenizer is None:
        return None
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return len(ids)
    except Exception:  # noqa: BLE001
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the generator once at startup (unless skipped), degrade gracefully."""
    app.state.generator = None
    if os.environ.get("EXTRACTOR_SKIP_MODEL_LOAD") == "1":
        logger.info("EXTRACTOR_SKIP_MODEL_LOAD=1 — starting without a model.")
    else:
        adapter_path = os.environ.get("EXTRACTOR_ADAPTER_PATH", DEFAULT_ADAPTER_PATH)
        try:
            from extractor.inference.model_loader import load_generator  # noqa: PLC0415

            app.state.generator = load_generator(adapter_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Model load failed (%s); /extract will return 503 until a model is loaded.",
                exc,
            )
    yield
    app.state.generator = None


app = FastAPI(
    title="Contract Extractor API",
    version="0.1.0",
    description="Fine-tuned Llama 3.1 8B serving 12-field structured contract extraction.",
    lifespan=lifespan,
)


def get_generator() -> Any:
    """Dependency: return the loaded generator, or ``None`` if unavailable.

    Intentionally does **not** raise — returning ``None`` lets request-body
    validation (422) run first, so malformed input is reported as a client
    error even when the model is down. Handlers call :func:`_require_generator`
    to turn a missing model into a 503. Tests override this with
    ``app.dependency_overrides[get_generator]``.
    """
    return getattr(app.state, "generator", None)


def _require_generator(generator: Any) -> Any:
    """Raise 503 if the generator is not loaded; otherwise return it."""
    if generator is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return generator


def _parse_extraction(raw_output: str) -> ContractExtraction:
    """Parse a model output string into a validated extraction, or raise 502."""
    try:
        return ContractExtraction.model_validate(json.loads(raw_output))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail="Model output was not valid JSON for the extraction schema.",
        ) from exc


@app.get("/health")
def health() -> dict:
    """Liveness probe. 200 always; reports whether a model is loaded."""
    return {"status": "ok", "model_loaded": getattr(app.state, "generator", None) is not None}


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest, generator: Any = Depends(get_generator)) -> ExtractResponse:
    """Synchronous extraction: full generation, then parse + return.

    Emits one Langfuse trace per request (no-op when observability is disabled)
    with input/output token counts, latency, and throughput. The blocking
    generate path cannot observe time-to-first-token, so that metric is left to
    the streaming endpoint.
    """
    generator = _require_generator(generator)
    messages = build_messages(request.contract_text)
    start = time.perf_counter()
    raw_output, tokens_generated = generator.generate(messages, DEFAULT_MAX_NEW_TOKENS)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    metrics = RequestMetrics(
        input_length_chars=len(request.contract_text),
        input_tokens=_count_input_tokens(generator, messages),
        output_tokens=tokens_generated,
        inference_time_ms=elapsed_ms,
    )
    input_preview = request.contract_text[:INPUT_PREVIEW_CHARS]
    output_preview = raw_output[:INPUT_PREVIEW_CHARS]
    try:
        extraction = _parse_extraction(raw_output)
    except HTTPException:
        get_observability().trace_extraction(
            metrics, input_preview=input_preview, output_preview=output_preview, status="invalid_output"
        )
        raise
    get_observability().trace_extraction(
        metrics, input_preview=input_preview, output_preview=output_preview, status="ok"
    )
    return ExtractResponse(
        extraction=extraction,
        inference_time_ms=elapsed_ms,
        tokens_generated=tokens_generated,
    )


@app.post("/extract/stream")
def extract_stream(request: ExtractRequest, generator: Any = Depends(get_generator)) -> StreamingResponse:
    """Stream partial generation as Server-Sent Events (``text/event-stream``).

    Emits ``data: <chunk>`` events as tokens arrive, then a final
    ``data: [DONE]`` sentinel. Chunks are JSON-encoded so newlines in the
    model output don't break the SSE framing. After the stream completes, one
    Langfuse trace is recorded with time-to-first-token and total latency.
    """
    generator = _require_generator(generator)
    messages = build_messages(request.contract_text)
    input_tokens = _count_input_tokens(generator, messages)

    def event_stream():
        start = time.perf_counter()
        ttft_ms: float | None = None
        for chunk in generator.stream(messages, DEFAULT_MAX_NEW_TOKENS):
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000.0
            yield f"data: {json.dumps(chunk)}\n\n"
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics = RequestMetrics(
            input_length_chars=len(request.contract_text),
            input_tokens=input_tokens,
            inference_time_ms=elapsed_ms,
            time_to_first_token_ms=ttft_ms,
        )
        get_observability().trace_extraction(
            metrics, input_preview=request.contract_text[:INPUT_PREVIEW_CHARS], status="ok"
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
