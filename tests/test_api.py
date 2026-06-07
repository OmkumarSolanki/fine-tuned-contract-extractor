"""Unit tests for ``extractor/api.py`` (FastAPI service).

CPU-only: no real model is loaded. We set ``EXTRACTOR_SKIP_MODEL_LOAD=1`` before
importing the app (so startup doesn't try to load a GPU model), and inject a
mock generator via FastAPI's dependency override. Covers the spec test cases:

- ``GET /health`` → 200.
- ``POST /extract`` valid → 200 with a valid ``ExtractResponse``.
- ``POST /extract`` missing ``contract_text`` → 422.
- ``POST /extract`` ``contract_text`` < 50 chars → 422.

Plus: 503 when no model, 502 on unparseable model output, SSE streaming, and
train/inference prompt parity.
"""

from __future__ import annotations

import json
import os

import pytest

# Must be set before importing the app so the startup lifespan skips model load.
os.environ["EXTRACTOR_SKIP_MODEL_LOAD"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from extractor.api import app, get_generator  # noqa: E402
from extractor.inference import prompt as prompt_mod  # noqa: E402

# A contract_text comfortably over the 50-char minimum.
VALID_CONTRACT = "This Supply Agreement is entered into by Acme Corp and Beta LLC " * 3


def _valid_extraction_json(document_name: str = "Acme Agreement") -> str:
    return json.dumps(
        {
            "document_name": document_name,
            "parties": ["Acme Corp", "Beta LLC"],
            "agreement_date": "2020-01-01",
            "effective_date": None,
            "expiration_date": None,
            "governing_law": "Delaware",
            "renewal_term": None,
            "notice_period_to_terminate_renewal": None,
            "exclusivity": None,
            "non_compete": None,
            "cap_on_liability": None,
            "uncapped_liability": None,
        }
    )


class FakeGenerator:
    """Stand-in for FineTunedGenerator: records calls, returns canned output."""

    def __init__(self, output: str | None = None, chunks: list[str] | None = None) -> None:
        self.output = output if output is not None else _valid_extraction_json()
        self.chunks = chunks if chunks is not None else ['{"document_name":', ' "X"}']
        self.calls: list[list[dict]] = []

    def generate(self, messages, max_new_tokens=2048):
        self.calls.append(messages)
        return self.output, 42

    def stream(self, messages, max_new_tokens=2048):
        self.calls.append(messages)
        for c in self.chunks:
            yield c


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    app.state.generator = None


def _use(generator) -> None:
    app.dependency_overrides[get_generator] = lambda: generator


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_ok_no_model(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_health_reports_model_loaded(client):
    app.state.generator = FakeGenerator()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


# ---------------------------------------------------------------------------
# /extract — happy path + validation
# ---------------------------------------------------------------------------


def test_extract_200_valid(client):
    _use(FakeGenerator(output=_valid_extraction_json("Strategic Alliance Agreement")))
    resp = client.post("/extract", json={"contract_text": VALID_CONTRACT})
    assert resp.status_code == 200
    body = resp.json()
    assert body["extraction"]["document_name"] == "Strategic Alliance Agreement"
    assert body["extraction"]["parties"] == ["Acme Corp", "Beta LLC"]
    # all 12 schema fields present
    assert len(body["extraction"]) == 12
    assert body["tokens_generated"] == 42
    assert isinstance(body["inference_time_ms"], (int, float))


def test_extract_422_missing_contract_text(client):
    _use(FakeGenerator())
    resp = client.post("/extract", json={})
    assert resp.status_code == 422


def test_extract_422_too_short(client):
    _use(FakeGenerator())
    resp = client.post("/extract", json={"contract_text": "too short"})
    assert resp.status_code == 422


def test_extract_503_when_no_model(client):
    # No override and no app.state.generator → dependency raises 503.
    resp = client.post("/extract", json={"contract_text": VALID_CONTRACT})
    assert resp.status_code == 503


def test_extract_502_on_unparseable_output(client):
    _use(FakeGenerator(output="Here is the extraction: not json"))
    resp = client.post("/extract", json={"contract_text": VALID_CONTRACT})
    assert resp.status_code == 502


def test_extract_validation_precedes_model_check(client):
    """A malformed body returns 422 even when no model is loaded (not 503)."""
    # No override and no app.state.generator → model is unavailable...
    resp = client.post("/extract", json={"contract_text": "short"})
    # ...but the too-short body must still be a 422 client error, not 503.
    assert resp.status_code == 422


def test_extract_uses_training_prompt(client):
    """The generator must be called with the exact training system/user turns."""
    fake = FakeGenerator()
    _use(fake)
    client.post("/extract", json={"contract_text": VALID_CONTRACT})
    messages = fake.calls[-1]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == prompt_mod.SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == prompt_mod.USER_PROMPT_TEMPLATE.format(contract_text=VALID_CONTRACT)


# ---------------------------------------------------------------------------
# /extract/stream
# ---------------------------------------------------------------------------


def test_extract_stream_sse(client):
    _use(FakeGenerator(chunks=['{"document_name":', ' "Acme"}']))
    resp = client.post("/extract/stream", json={"contract_text": VALID_CONTRACT})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    text = resp.text
    assert "data: " in text
    assert "[DONE]" in text
    # chunks are JSON-encoded inside the SSE frames
    assert json.dumps('{"document_name":') in text


def test_extract_stream_422_too_short(client):
    _use(FakeGenerator())
    resp = client.post("/extract/stream", json={"contract_text": "x"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Prompt parity (drift guard) + build_messages
# ---------------------------------------------------------------------------


def test_prompt_constants_are_the_training_ones():
    """Inference prompts MUST be the same objects as training — no drift."""
    from training.prepare_dataset import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

    assert prompt_mod.SYSTEM_PROMPT is SYSTEM_PROMPT
    assert prompt_mod.USER_PROMPT_TEMPLATE is USER_PROMPT_TEMPLATE


def test_build_messages_structure():
    msgs = prompt_mod.build_messages("CONTRACT BODY")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "CONTRACT BODY" in msgs[1]["content"]
