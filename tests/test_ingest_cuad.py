"""Tests for ``training/ingest_cuad.py`` helpers.

These tests exercise pure-Python helpers and do not hit the network. The
heavy ``datasets``/``tqdm`` imports inside ``main()`` are deferred specifically
so this file remains importable in CI without those wheels.
"""

from __future__ import annotations

import pytest

from extractor.schemas import ContractExtraction
from training.ingest_cuad import (
    TARGET_CATEGORIES,
    aggregate_contract,
    assemble_contract_text,
    dedupe_preserve_case,
    extract_category_from_id,
    extract_chunk_index_from_id,
    parse_date_loose,
    pick_longest_span,
)


# ---------------------------------------------------------------------------
# extract_category_from_id / extract_chunk_index_from_id
# ---------------------------------------------------------------------------


def test_extract_category_from_id_basic() -> None:
    """The id format documented in the dataset card."""
    sample_id = (
        "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT__Document Name_0"
    )
    assert extract_category_from_id(sample_id) == "Document Name"
    assert extract_chunk_index_from_id(sample_id) == 0


def test_extract_category_from_id_with_underscores_in_category() -> None:
    """Categories with multiple words separated by spaces."""
    sample_id = "SOMECONTRACT__Notice Period To Terminate Renewal_2"
    assert extract_category_from_id(sample_id) == "Notice Period To Terminate Renewal"
    assert extract_chunk_index_from_id(sample_id) == 2


def test_extract_category_from_id_with_dash() -> None:
    """Non-Compete contains a literal dash."""
    sample_id = "FOO__Non-Compete_5"
    assert extract_category_from_id(sample_id) == "Non-Compete"
    assert extract_chunk_index_from_id(sample_id) == 5


def test_extract_category_from_id_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unparseable id"):
        extract_category_from_id("noseparator")


def test_extract_chunk_index_from_id_multidigit() -> None:
    sample_id = "X__Cap On Liability_27"
    assert extract_chunk_index_from_id(sample_id) == 27


def test_extract_category_from_id_no_chunk_suffix() -> None:
    """Some single-chunk contracts have ids without a trailing ``_<N>``."""
    sample_id = (
        "ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-"
        "JOINT VENTURE AGREEMENT__Document Name"
    )
    assert extract_category_from_id(sample_id) == "Document Name"
    # Missing chunk index defaults to 0.
    assert extract_chunk_index_from_id(sample_id) == 0


def test_extract_chunk_index_no_chunk_means_zero() -> None:
    assert extract_chunk_index_from_id("X__Parties") == 0


# ---------------------------------------------------------------------------
# parse_date_loose
# ---------------------------------------------------------------------------


def test_parse_date_loose_iso() -> None:
    assert parse_date_loose("2018-05-15") == "2018-05-15"


def test_parse_date_loose_natural_language() -> None:
    assert parse_date_loose("May 15, 2018") == "2018-05-15"


def test_parse_date_loose_with_surrounding_text() -> None:
    """``fuzzy=True`` lets us pull a date out of a clause-like span."""
    assert parse_date_loose("dated as of January 1, 2024") == "2024-01-01"


def test_parse_date_loose_year_only() -> None:
    """A bare year is a real date — should normalize to YYYY-01-01."""
    assert parse_date_loose("2018") == "2018-01-01"


def test_parse_date_loose_fallback_no_year() -> None:
    """No real date in the span → return the raw stripped string."""
    raw = "as of the Effective Date"
    assert parse_date_loose(raw) == raw


def test_parse_date_loose_none() -> None:
    assert parse_date_loose(None) is None


def test_parse_date_loose_empty_strings() -> None:
    assert parse_date_loose("") is None
    assert parse_date_loose("    ") is None


# ---------------------------------------------------------------------------
# dedupe_preserve_case
# ---------------------------------------------------------------------------


def test_dedupe_preserve_case_basic() -> None:
    """Case-insensitive dedup; first occurrence's casing wins."""
    assert dedupe_preserve_case(
        ["Acme Corp", "ACME CORP", "Beta", "  beta  "]
    ) == ["Acme Corp", "Beta"]


def test_dedupe_preserve_case_drops_empties() -> None:
    assert dedupe_preserve_case(["", "  ", "Acme", None]) == ["Acme"]  # type: ignore[list-item]


def test_dedupe_preserve_case_empty_input() -> None:
    assert dedupe_preserve_case([]) == []


# ---------------------------------------------------------------------------
# pick_longest_span
# ---------------------------------------------------------------------------


def test_pick_longest_span_basic() -> None:
    assert (
        pick_longest_span(["short", "much longer span", "mid"])
        == "much longer span"
    )


def test_pick_longest_span_strips_whitespace() -> None:
    assert pick_longest_span(["  short  ", "  longer  "]) == "longer"


def test_pick_longest_span_all_empty() -> None:
    assert pick_longest_span(["", "  ", ""]) is None
    assert pick_longest_span([]) is None


# ---------------------------------------------------------------------------
# aggregate_contract
# ---------------------------------------------------------------------------


def test_aggregate_contract_full() -> None:
    """A realistic mapping from CUAD categories to our 12 schema fields."""
    spans = {
        "Document Name": ["License Agreement"],
        "Parties": ["Acme Corp", "Beta Inc", "ACME CORP"],
        "Agreement Date": ["May 15, 2018"],
        "Effective Date": ["June 1, 2018"],
        "Expiration Date": ["the date of termination"],  # un-parseable → raw
        "Governing Law": ["Delaware"],
        "Renewal Term": ["Auto-renews for 1-year terms"],
        "Notice Period To Terminate Renewal": ["60 days written notice"],
        "Exclusivity": ["Territorial — North America"],
        "Non-Compete": ["12 months post-termination"],
        "Cap On Liability": ["Limited to fees paid in the prior 12 months"],
        "Uncapped Liability": ["IP infringement and gross negligence are uncapped"],
    }
    result = aggregate_contract(spans)

    # Validates as the schema.
    extraction = ContractExtraction.model_validate(result)

    assert extraction.document_name == "License Agreement"
    assert extraction.parties == ["Acme Corp", "Beta Inc"]  # deduped, casing preserved
    assert extraction.agreement_date == "2018-05-15"
    assert extraction.effective_date == "2018-06-01"
    assert extraction.expiration_date == "the date of termination"  # fallback
    assert extraction.governing_law == "Delaware"
    assert extraction.uncapped_liability == "IP infringement and gross negligence are uncapped"


def test_aggregate_contract_empty() -> None:
    """No spans for any category → all defaults; still validates."""
    result = aggregate_contract({})
    extraction = ContractExtraction.model_validate(result)
    assert extraction.parties == []
    for field in TARGET_CATEGORIES:
        if field == "parties":
            continue
        assert getattr(extraction, field) is None


def test_aggregate_contract_picks_longest_for_singular_fields() -> None:
    """When a category has multiple spans, the longest one wins."""
    spans = {
        "Uncapped Liability": [
            "short",
            "IP infringement, breach of confidentiality, and gross negligence are uncapped",
            "medium length",
        ],
    }
    result = aggregate_contract(spans)
    assert (
        result["uncapped_liability"]
        == "IP infringement, breach of confidentiality, and gross negligence are uncapped"
    )


# ---------------------------------------------------------------------------
# assemble_contract_text
# ---------------------------------------------------------------------------


def test_assemble_contract_text_orders_chunks_and_dedupes() -> None:
    """Chunks are ordered ascending; identical contexts within the same chunk
    index (which happens because each chunk repeats once per question) are
    deduplicated."""
    rows = [
        {"id": "X__Document Name_1", "context": "CHUNK ONE"},
        {"id": "X__Parties_0", "context": "CHUNK ZERO"},
        {"id": "X__Parties_1", "context": "CHUNK ONE"},  # duplicate of chunk 1
        {"id": "X__Document Name_0", "context": "CHUNK ZERO"},  # dup of chunk 0
        {"id": "X__Document Name_2", "context": "CHUNK TWO"},
    ]
    text = assemble_contract_text(rows)
    assert text == "CHUNK ZERO\n\nCHUNK ONE\n\nCHUNK TWO"


def test_assemble_contract_text_handles_unparseable_ids() -> None:
    """Bad ids are silently skipped; remaining rows still produce output."""
    rows = [
        {"id": "X__Parties_0", "context": "GOOD"},
        {"id": "garbage_id", "context": "IGNORED"},
    ]
    assert assemble_contract_text(rows) == "GOOD"


def test_assemble_contract_text_empty() -> None:
    assert assemble_contract_text([]) == ""
