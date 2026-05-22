"""Tests for ``evaluation/metrics.py``."""

from __future__ import annotations

import json

import pytest

from evaluation.metrics import field_accuracy, is_valid_json, overall_f1, parties_f1
from extractor.schemas import ContractExtraction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _full_extraction() -> ContractExtraction:
    return ContractExtraction(
        document_name="License Agreement",
        parties=["Acme Corp", "Beta Inc"],
        agreement_date="2018-05-15",
        effective_date="2018-06-01",
        expiration_date="2023-06-01",
        governing_law="Delaware",
        renewal_term="Auto-renews for 1-year terms",
        notice_period_to_terminate_renewal="60 days written notice",
        exclusivity="Territorial",
        non_compete="12 months",
        cap_on_liability="Limited to fees paid",
        uncapped_liability="IP infringement and gross negligence are uncapped",
    )


# ---------------------------------------------------------------------------
# is_valid_json
# ---------------------------------------------------------------------------


def test_is_valid_json_accepts_valid_extraction() -> None:
    payload = _full_extraction().model_dump()
    assert is_valid_json(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def test_is_valid_json_accepts_minimal_object() -> None:
    """All fields optional; an empty object is a valid extraction."""
    assert is_valid_json("{}") is True


def test_is_valid_json_accepts_partial_object() -> None:
    assert is_valid_json('{"document_name": "X", "parties": []}') is True


def test_is_valid_json_rejects_malformed_json() -> None:
    assert is_valid_json("{not json") is False


def test_is_valid_json_rejects_empty_string() -> None:
    assert is_valid_json("") is False


def test_is_valid_json_rejects_schema_violation() -> None:
    """`parties` must be a list; passing a string fails schema validation."""
    assert is_valid_json('{"parties": "not a list"}') is False


def test_is_valid_json_rejects_non_object_root() -> None:
    """Top-level JSON must be an object, not an array."""
    assert is_valid_json("[1, 2, 3]") is False


# ---------------------------------------------------------------------------
# field_accuracy
# ---------------------------------------------------------------------------


def test_field_accuracy_both_none() -> None:
    pred = ContractExtraction()
    gold = ContractExtraction()
    assert field_accuracy(pred, gold, "governing_law") == 1.0


def test_field_accuracy_one_none_pred() -> None:
    pred = ContractExtraction()
    gold = ContractExtraction(governing_law="Delaware")
    assert field_accuracy(pred, gold, "governing_law") == 0.0


def test_field_accuracy_one_none_gold() -> None:
    pred = ContractExtraction(governing_law="Delaware")
    gold = ContractExtraction()
    assert field_accuracy(pred, gold, "governing_law") == 0.0


def test_field_accuracy_string_match_case_insensitive() -> None:
    pred = ContractExtraction(governing_law="Delaware")
    gold = ContractExtraction(governing_law="delaware")
    assert field_accuracy(pred, gold, "governing_law") == 1.0


def test_field_accuracy_string_match_with_whitespace() -> None:
    pred = ContractExtraction(governing_law="  Delaware  ")
    gold = ContractExtraction(governing_law="Delaware")
    assert field_accuracy(pred, gold, "governing_law") == 1.0


def test_field_accuracy_string_mismatch() -> None:
    pred = ContractExtraction(governing_law="Delaware")
    gold = ContractExtraction(governing_law="California")
    assert field_accuracy(pred, gold, "governing_law") == 0.0


# ---------------------------------------------------------------------------
# parties_f1
# ---------------------------------------------------------------------------


def test_parties_f1_identical_lists() -> None:
    pred = ContractExtraction(parties=["Acme", "Beta"])
    gold = ContractExtraction(parties=["Acme", "Beta"])
    assert parties_f1(pred, gold) == 1.0


def test_parties_f1_disjoint_lists() -> None:
    pred = ContractExtraction(parties=["Acme"])
    gold = ContractExtraction(parties=["Gamma"])
    assert parties_f1(pred, gold) == 0.0


def test_parties_f1_half_overlap() -> None:
    """precision=0.5, recall=0.5 → F1=0.5."""
    pred = ContractExtraction(parties=["Acme", "Beta"])
    gold = ContractExtraction(parties=["Acme", "Gamma"])
    assert parties_f1(pred, gold) == pytest.approx(0.5)


def test_parties_f1_case_insensitive() -> None:
    pred = ContractExtraction(parties=["ACME"])
    gold = ContractExtraction(parties=["acme"])
    assert parties_f1(pred, gold) == 1.0


def test_parties_f1_whitespace_normalized() -> None:
    pred = ContractExtraction(parties=["  Acme  "])
    gold = ContractExtraction(parties=["Acme"])
    assert parties_f1(pred, gold) == 1.0


def test_parties_f1_both_empty() -> None:
    pred = ContractExtraction(parties=[])
    gold = ContractExtraction(parties=[])
    assert parties_f1(pred, gold) == 1.0


def test_parties_f1_pred_empty() -> None:
    pred = ContractExtraction(parties=[])
    gold = ContractExtraction(parties=["Acme"])
    assert parties_f1(pred, gold) == 0.0


def test_parties_f1_gold_empty() -> None:
    pred = ContractExtraction(parties=["Acme"])
    gold = ContractExtraction(parties=[])
    assert parties_f1(pred, gold) == 0.0


def test_parties_f1_precision_recall_imbalance() -> None:
    """pred has 4 parties, gold has 2 → P=0.5, R=1.0, F1=2*0.5*1/(0.5+1)=2/3."""
    pred = ContractExtraction(parties=["Acme", "Beta", "Extra1", "Extra2"])
    gold = ContractExtraction(parties=["Acme", "Beta"])
    assert parties_f1(pred, gold) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# overall_f1
# ---------------------------------------------------------------------------


def test_overall_f1_perfect_match() -> None:
    """If pred==gold for every example, every per-field score and overall_f1 = 1.0."""
    examples = [_full_extraction() for _ in range(3)]
    result = overall_f1(examples, [e.model_copy() for e in examples])
    assert result["overall_f1"] == pytest.approx(1.0)
    for field in ContractExtraction.model_fields:
        assert result[field] == pytest.approx(1.0), f"field {field} expected 1.0"


def test_overall_f1_keys_match_field_set() -> None:
    """Returned dict has one key per schema field plus the aggregate."""
    examples = [_full_extraction()]
    result = overall_f1(examples, [_full_extraction()])
    expected_keys = set(ContractExtraction.model_fields) | {"overall_f1"}
    assert set(result.keys()) == expected_keys


def test_overall_f1_full_mismatch() -> None:
    """Disjoint pred vs gold on every field → all zeros (except both-empty edge cases)."""
    pred = ContractExtraction(
        document_name="A",
        parties=["P1"],
        governing_law="Delaware",
    )
    gold = ContractExtraction(
        document_name="B",
        parties=["P2"],
        governing_law="California",
    )
    result = overall_f1([pred], [gold])
    assert result["document_name"] == 0.0
    assert result["parties"] == 0.0
    assert result["governing_law"] == 0.0
    # Untouched fields are None on both sides → both-None match → 1.0.
    assert result["agreement_date"] == 1.0


def test_overall_f1_mixed_examples() -> None:
    """Two examples: one perfect, one with parties wrong. parties_avg = (1.0 + 0.0)/2."""
    e1 = _full_extraction()
    e2_pred = _full_extraction()
    e2_pred.parties = ["Wrong"]
    result = overall_f1([e1, e2_pred], [e1, _full_extraction()])
    assert result["parties"] == pytest.approx(0.5)
    # All other fields match perfectly across both examples.
    assert result["document_name"] == 1.0
    assert result["governing_law"] == 1.0


def test_overall_f1_empty_inputs() -> None:
    """Empty input lists return zeros without crashing."""
    result = overall_f1([], [])
    for field in ContractExtraction.model_fields:
        assert result[field] == 0.0
    assert result["overall_f1"] == 0.0
