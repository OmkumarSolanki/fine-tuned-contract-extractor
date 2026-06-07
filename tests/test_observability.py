"""Unit tests for ``extractor/observability`` (Phase 9 — Langfuse).

CPU-only and offline: no real Langfuse client is ever constructed. We exercise

- :class:`RequestMetrics` throughput derivation + ``as_dict`` filtering;
- config reading and the no-op fallback when credentials/SDK are missing;
- the process-wide singleton (``get_observability`` / ``reset_observability``);
- :class:`Observability` against fake *modern* (``start_span``) and *legacy*
  (``trace``) client surfaces, including that a client error never propagates;
- end-to-end through the FastAPI ``/extract`` endpoint with a fake client
  injected as the singleton, asserting a trace is recorded and the request
  still succeeds.
"""

from __future__ import annotations

import os

import pytest

# Skip the GPU model load when the API app is imported below.
os.environ["EXTRACTOR_SKIP_MODEL_LOAD"] = "1"

from extractor.observability import (  # noqa: E402
    Observability,
    RequestMetrics,
    get_observability,
    reset_observability,
)
from extractor.observability import langfuse_setup  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure each test starts and ends with a clean observability singleton."""
    reset_observability()
    yield
    reset_observability()


# ---------------------------------------------------------------------------
# RequestMetrics
# ---------------------------------------------------------------------------


def test_metrics_compute_throughput():
    m = RequestMetrics(input_length_chars=100, output_tokens=200, inference_time_ms=1000.0)
    m.compute_throughput()
    assert m.tokens_per_second == pytest.approx(200.0)


def test_metrics_throughput_noop_without_output_tokens():
    m = RequestMetrics(input_length_chars=100, inference_time_ms=1000.0)
    m.compute_throughput()
    assert m.tokens_per_second is None


def test_metrics_throughput_noop_on_zero_time():
    m = RequestMetrics(input_length_chars=100, output_tokens=50, inference_time_ms=0.0)
    m.compute_throughput()
    assert m.tokens_per_second is None


def test_metrics_as_dict_drops_none():
    m = RequestMetrics(input_length_chars=100, output_tokens=10, inference_time_ms=500.0)
    m.compute_throughput()
    d = m.as_dict()
    assert d["input_length_chars"] == 100
    assert d["output_tokens"] == 10
    assert "tokens_per_second" in d
    # Unpopulated optionals are omitted.
    assert "input_tokens" not in d
    assert "time_to_first_token_ms" not in d


# ---------------------------------------------------------------------------
# Config + no-op fallback
# ---------------------------------------------------------------------------


def test_read_config_picks_up_env():
    cfg = langfuse_setup._read_config(
        {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk", "LANGFUSE_HOST": "http://x"}
    )
    assert cfg == {"public_key": "pk", "secret_key": "sk", "host": "http://x"}


def test_read_config_empty_strings_become_none():
    cfg = langfuse_setup._read_config(
        {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": "", "LANGFUSE_HOST": ""}
    )
    assert cfg == {"public_key": None, "secret_key": None, "host": None}


def test_build_client_none_without_keys():
    client = langfuse_setup._build_client({"public_key": None, "secret_key": None, "host": None})
    assert client is None


def test_get_observability_disabled_when_no_keys():
    obs = get_observability(env={})
    assert isinstance(obs, Observability)
    assert obs.enabled is False
    # No client → trace is a silent no-op that never raises.
    obs.trace_extraction(RequestMetrics(input_length_chars=10))


def test_get_observability_is_singleton():
    a = get_observability(env={})
    b = get_observability(env={})
    assert a is b
    reset_observability()
    c = get_observability(env={})
    assert c is not a


# ---------------------------------------------------------------------------
# Fake client surfaces
# ---------------------------------------------------------------------------


class FakeSpan:
    def __init__(self) -> None:
        self.updated_with: dict | None = None
        self.ended = False

    def update(self, output=None):
        self.updated_with = output

    def end(self):
        self.ended = True


class FakeModernClient:
    """Mimics the modern Langfuse SDK surface (``start_span`` + ``flush``)."""

    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []
        self.start_span_calls: list[dict] = []
        self.flushed = 0

    def start_span(self, name=None, input=None, metadata=None):
        self.start_span_calls.append({"name": name, "input": input, "metadata": metadata})
        span = FakeSpan()
        self.spans.append(span)
        return span

    def flush(self):
        self.flushed += 1


class FakeLegacyClient:
    """Mimics the legacy Langfuse SDK surface (top-level ``trace``)."""

    def __init__(self) -> None:
        self.trace_calls: list[dict] = []
        self.flushed = 0

    def trace(self, name=None, input=None, output=None, metadata=None):
        self.trace_calls.append(
            {"name": name, "input": input, "output": output, "metadata": metadata}
        )

    def flush(self):
        self.flushed += 1


class FakeV4Span:
    """A langfuse>=3 (OTel) span: ``update`` accepts ``level``, plus ``end``."""

    def __init__(self) -> None:
        self.updated_with: dict | None = None
        self.level: str | None = None
        self.ended = False

    def update(self, output=None, level=None, **kwargs):
        self.updated_with = output
        self.level = level

    def end(self):
        self.ended = True


class FakeV4Client:
    """Mimics the langfuse 4.x surface (``start_observation`` + ``flush``).

    This is the surface the pinned ``langfuse>=2.0.0`` actually resolves to
    today (4.7.1), so it is what the API end-to-end tests exercise.
    """

    def __init__(self) -> None:
        self.spans: list[FakeV4Span] = []
        self.start_observation_calls: list[dict] = []
        self.flushed = 0

    def start_observation(self, name=None, input=None, metadata=None, **kwargs):
        self.start_observation_calls.append(
            {"name": name, "input": input, "metadata": metadata}
        )
        span = FakeV4Span()
        self.spans.append(span)
        return span

    # Provided so a test can assert start_observation is preferred over these.
    def start_span(self, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("start_observation must be preferred over start_span")

    def flush(self):
        self.flushed += 1


def test_trace_v4_surface():
    client = FakeV4Client()
    obs = Observability(client)
    assert obs.enabled is True

    m = RequestMetrics(input_length_chars=120, output_tokens=60, inference_time_ms=2000.0)
    obs.trace_extraction(m, input_preview="ACME...", output_preview='{"x":1}', status="ok")

    assert len(client.start_observation_calls) == 1
    call = client.start_observation_calls[0]
    assert call["name"] == langfuse_setup.TRACE_NAME
    assert call["input"] == {"contract_preview": "ACME..."}
    assert call["metadata"]["tokens_per_second"] == pytest.approx(30.0)
    span = client.spans[0]
    assert span.updated_with == {"extraction_preview": '{"x":1}', "status": "ok"}
    assert span.level == "DEFAULT"
    assert span.ended is True
    assert client.flushed == 1


def test_trace_v4_error_level_on_invalid_output():
    client = FakeV4Client()
    Observability(client).trace_extraction(
        RequestMetrics(input_length_chars=10), output_preview="not json", status="invalid_output"
    )
    assert client.spans[0].level == "ERROR"


def test_trace_modern_surface():
    client = FakeModernClient()
    obs = Observability(client)
    assert obs.enabled is True

    m = RequestMetrics(input_length_chars=120, output_tokens=60, inference_time_ms=2000.0)
    obs.trace_extraction(m, input_preview="ACME...", output_preview='{"x":1}', status="ok")

    assert len(client.start_span_calls) == 1
    call = client.start_span_calls[0]
    assert call["name"] == langfuse_setup.TRACE_NAME
    assert call["input"] == {"contract_preview": "ACME..."}
    # Throughput was derived and passed through as metadata.
    assert call["metadata"]["tokens_per_second"] == pytest.approx(30.0)
    assert client.spans[0].updated_with == {"extraction_preview": '{"x":1}', "status": "ok"}
    assert client.spans[0].ended is True
    assert client.flushed == 1


def test_trace_legacy_surface():
    client = FakeLegacyClient()
    obs = Observability(client)
    obs.trace_extraction(
        RequestMetrics(input_length_chars=10), input_preview="p", output_preview="o", status="ok"
    )
    assert len(client.trace_calls) == 1
    assert client.trace_calls[0]["name"] == langfuse_setup.TRACE_NAME
    assert client.flushed == 1


def test_trace_swallows_client_errors():
    class ExplodingClient:
        def start_span(self, **kwargs):
            raise RuntimeError("langfuse down")

    obs = Observability(ExplodingClient())
    # Must not raise despite the client blowing up.
    obs.trace_extraction(RequestMetrics(input_length_chars=10), input_preview="p")


def test_invalid_output_status_recorded():
    client = FakeModernClient()
    obs = Observability(client)
    obs.trace_extraction(
        RequestMetrics(input_length_chars=10), output_preview="not json", status="invalid_output"
    )
    assert client.spans[0].updated_with["status"] == "invalid_output"


# ---------------------------------------------------------------------------
# End-to-end through the FastAPI app
# ---------------------------------------------------------------------------


def test_extract_records_trace_and_still_succeeds():
    import json

    from fastapi.testclient import TestClient

    from extractor.api import app, get_generator

    valid = json.dumps(
        {
            "document_name": "Acme Agreement",
            "parties": ["Acme Corp", "Beta LLC"],
            "agreement_date": None,
            "effective_date": None,
            "expiration_date": None,
            "governing_law": None,
            "renewal_term": None,
            "notice_period_to_terminate_renewal": None,
            "exclusivity": None,
            "non_compete": None,
            "cap_on_liability": None,
            "uncapped_liability": None,
        }
    )

    class FakeGenerator:
        tokenizer = None  # forces input_tokens=None (best-effort)

        def generate(self, messages, max_new_tokens=2048):
            return valid, 42

        def stream(self, messages, max_new_tokens=2048):
            yield valid

    # Inject a fake Langfuse client as the process-wide singleton.
    client = FakeV4Client()
    langfuse_setup._OBSERVABILITY = Observability(client)

    app.dependency_overrides[get_generator] = lambda: FakeGenerator()
    try:
        with TestClient(app) as c:
            resp = c.post(
                "/extract",
                json={"contract_text": "This Agreement is between Acme and Beta. " * 3},
            )
        assert resp.status_code == 200
        assert len(client.start_observation_calls) == 1
        meta = client.start_observation_calls[0]["metadata"]
        assert meta["output_tokens"] == 42
        assert "inference_time_ms" in meta
        assert "tokens_per_second" in meta
        assert meta.get("input_tokens") is None  # tokenizer was None → omitted
        assert client.spans[0].updated_with["status"] == "ok"
    finally:
        app.dependency_overrides.clear()
        app.state.generator = None


def test_extract_traces_invalid_output_then_502():
    from fastapi.testclient import TestClient

    from extractor.api import app, get_generator

    class BadGenerator:
        tokenizer = None

        def generate(self, messages, max_new_tokens=2048):
            return "Here is the extraction: not json", 7

        def stream(self, messages, max_new_tokens=2048):
            yield "nope"

    client = FakeModernClient()
    langfuse_setup._OBSERVABILITY = Observability(client)

    app.dependency_overrides[get_generator] = lambda: BadGenerator()
    try:
        with TestClient(app) as c:
            resp = c.post(
                "/extract",
                json={"contract_text": "This Agreement is between Acme and Beta. " * 3},
            )
        assert resp.status_code == 502
        # Even on a 502, the request was traced with the invalid-output status.
        assert client.spans[0].updated_with["status"] == "invalid_output"
    finally:
        app.dependency_overrides.clear()
        app.state.generator = None
