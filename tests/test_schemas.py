"""Tests for ``extractor/schemas.py``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from extractor.schemas import ContractExtraction, ExtractRequest, ExtractResponse


# ---------------------------------------------------------------------------
# ContractExtraction
# ---------------------------------------------------------------------------


EXPECTED_FIELDS = [
    "document_name",
    "parties",
    "agreement_date",
    "effective_date",
    "expiration_date",
    "governing_law",
    "renewal_term",
    "notice_period_to_terminate_renewal",
    "exclusivity",
    "non_compete",
    "cap_on_liability",
    "uncapped_liability",
]


def test_field_count_and_order() -> None:
    """The schema must have exactly 12 fields in the documented order.

    Several downstream pieces (``compact_json`` in ``prepare_dataset.py``,
    ``overall_f1`` in ``metrics.py``) iterate over ``model_fields`` and assume
    this order is stable.
    """
    assert list(ContractExtraction.model_fields) == EXPECTED_FIELDS
    assert len(ContractExtraction.model_fields) == 12


def test_contract_extraction_full_payload() -> None:
    """All 12 fields populated; validates and round-trips through model_dump."""
    payload = {
        "document_name": "License Agreement",
        "parties": ["Acme Corp", "Beta Inc"],
        "agreement_date": "2018-05-15",
        "effective_date": "2018-06-01",
        "expiration_date": "2023-06-01",
        "governing_law": "Delaware",
        "renewal_term": "Auto-renews for 1-year terms",
        "notice_period_to_terminate_renewal": "60 days written notice",
        "exclusivity": "Territorial — North America",
        "non_compete": "12 months post-termination",
        "cap_on_liability": "Limited to fees paid in the prior 12 months",
        "uncapped_liability": "IP infringement and breach of confidentiality are uncapped",
    }
    extraction = ContractExtraction.model_validate(payload)
    dumped = extraction.model_dump()
    assert dumped == payload
    # Re-validate the dumped output to confirm round-trip stability.
    assert ContractExtraction.model_validate(dumped).model_dump() == payload


def test_contract_extraction_all_null() -> None:
    """All-None payload (and empty parties list) validates."""
    payload = {field: None for field in EXPECTED_FIELDS}
    payload["parties"] = []
    extraction = ContractExtraction.model_validate(payload)
    assert extraction.parties == []
    assert extraction.document_name is None
    assert extraction.uncapped_liability is None


def test_contract_extraction_defaults_when_empty() -> None:
    """Constructing with no kwargs uses the documented defaults."""
    extraction = ContractExtraction()
    assert extraction.parties == []
    for field in EXPECTED_FIELDS:
        if field == "parties":
            continue
        assert getattr(extraction, field) is None


def test_contract_extraction_partial() -> None:
    """A mix of populated and None fields validates."""
    payload = {
        "document_name": "Distributor Agreement",
        "parties": ["Lime Energy Corp"],
        "governing_law": "Illinois",
    }
    extraction = ContractExtraction.model_validate(payload)
    assert extraction.document_name == "Distributor Agreement"
    assert extraction.parties == ["Lime Energy Corp"]
    assert extraction.governing_law == "Illinois"
    assert extraction.agreement_date is None
    assert extraction.cap_on_liability is None


def test_contract_extraction_rejects_wrong_types() -> None:
    """`parties` must be a list of strings, not a single string."""
    with pytest.raises(ValidationError):
        ContractExtraction.model_validate({"parties": "Acme"})


def test_contract_extraction_rejects_non_string_party() -> None:
    """List entries must be strings (or coercible — Pydantic v2 default behavior)."""
    with pytest.raises(ValidationError):
        ContractExtraction.model_validate({"parties": [{"name": "Acme"}]})


def test_contract_extraction_default_extra_fields_behavior() -> None:
    """Pydantic v2 defaults to ignoring extra fields silently.

    We assert this explicitly so we notice if model_config changes to
    ``extra='forbid'`` later. If that happens, downstream code that relies on
    forward-compatible payloads (e.g., when CUAD's category list grows) will
    need to be updated.
    """
    payload = {"document_name": "X", "parties": [], "future_field": "ignored"}
    extraction = ContractExtraction.model_validate(payload)
    assert extraction.document_name == "X"
    assert not hasattr(extraction, "future_field")


# ---------------------------------------------------------------------------
# ExtractRequest
# ---------------------------------------------------------------------------


def test_extract_request_min_length_rejects_short() -> None:
    """`contract_text` shorter than 50 characters must raise ValidationError."""
    with pytest.raises(ValidationError):
        ExtractRequest(contract_text="x" * 49)


def test_extract_request_accepts_at_minimum_length() -> None:
    """Exactly 50 characters is the boundary and must validate."""
    request = ExtractRequest(contract_text="x" * 50)
    assert len(request.contract_text) == 50


def test_extract_request_accepts_long_text() -> None:
    """Realistic contract-length text validates."""
    text = "AGREEMENT made as of January 1, 2024, between Acme Corp and Beta Inc. " * 50
    request = ExtractRequest(contract_text=text)
    assert request.contract_text.startswith("AGREEMENT")


# ---------------------------------------------------------------------------
# ExtractResponse
# ---------------------------------------------------------------------------


def test_extract_response_round_trip() -> None:
    """Construct ExtractResponse with a populated extraction and round-trip dump it."""
    extraction = ContractExtraction(
        document_name="License Agreement",
        parties=["Acme Corp", "Beta Inc"],
        governing_law="Delaware",
    )
    response = ExtractResponse(
        extraction=extraction,
        inference_time_ms=1234.5,
        tokens_generated=456,
    )
    dumped = response.model_dump()
    assert dumped["extraction"]["document_name"] == "License Agreement"
    assert dumped["inference_time_ms"] == 1234.5
    assert dumped["tokens_generated"] == 456
    # Re-validate to confirm round-trip.
    reloaded = ExtractResponse.model_validate(dumped)
    assert reloaded.extraction.parties == ["Acme Corp", "Beta Inc"]


def test_extract_response_requires_all_fields() -> None:
    """ExtractResponse has no defaults; missing fields must raise."""
    with pytest.raises(ValidationError):
        ExtractResponse.model_validate({"extraction": {}, "inference_time_ms": 1.0})
