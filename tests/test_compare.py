"""Unit tests for ``evaluation/compare.py`` — the three-way comparison.

Pure-Python and CPU-only: no model, no network. Coverage:

- ``record_to_extraction`` (valid → parsed; missing/invalid → empty extraction).
- ``load_golds`` / ``load_predictions`` round-trips.
- ``score_model`` (validity rate + per-field/overall_f1, alignment by id).
- ``build_comparison`` (skips models whose prediction file is absent).
- ``format_table`` / ``write_summary`` / CLI surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import compare
from evaluation.compare import (
    build_comparison,
    format_table,
    load_golds,
    load_predictions,
    record_to_extraction,
    score_model,
    write_summary,
)
from extractor.schemas import ContractExtraction


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _extraction_dict(document_name="Acme", parties=("A", "B")) -> dict:
    return {
        "document_name": document_name,
        "parties": list(parties),
        "agreement_date": None,
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


def _write_test_jsonl(path: Path, golds: dict[str, dict]) -> None:
    """Write a gold test.jsonl: system + user + assistant(gold compact JSON)."""
    with path.open("w", encoding="utf-8") as fh:
        for cid, extraction in golds.items():
            row = {
                "contract_id": cid,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": f"user {cid}"},
                    {"role": "assistant", "content": json.dumps(extraction)},
                ],
            }
            fh.write(json.dumps(row) + "\n")


def _write_predictions(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh)


def _valid_record(cid: str, extraction: dict) -> dict:
    return {
        "contract_id": cid,
        "raw_output": json.dumps(extraction),
        "parsed": extraction,
        "is_valid_json": True,
    }


def _invalid_record(cid: str) -> dict:
    return {
        "contract_id": cid,
        "raw_output": "not json",
        "parsed": None,
        "is_valid_json": False,
    }


# ---------------------------------------------------------------------------
# record_to_extraction
# ---------------------------------------------------------------------------


def test_record_to_extraction_valid():
    ext, is_valid = record_to_extraction(_valid_record("c1", _extraction_dict("X")))
    assert is_valid is True
    assert ext.document_name == "X"
    assert ext.governing_law == "Delaware"


def test_record_to_extraction_none_is_empty():
    ext, is_valid = record_to_extraction(None)
    assert is_valid is False
    assert ext.document_name is None
    assert ext.parties == []


def test_record_to_extraction_invalid_is_empty():
    ext, is_valid = record_to_extraction(_invalid_record("c1"))
    assert is_valid is False
    assert ext == ContractExtraction()


def test_record_to_extraction_valid_flag_but_unparseable_parsed():
    """Defensive: is_valid_json True but parsed fails schema → empty, not a crash."""
    bad = {"contract_id": "c1", "raw_output": "x", "parsed": {"parties": "nope"},
           "is_valid_json": True}
    ext, is_valid = record_to_extraction(bad)
    assert is_valid is False
    assert ext == ContractExtraction()


# ---------------------------------------------------------------------------
# load_golds / load_predictions
# ---------------------------------------------------------------------------


def test_load_golds_reads_assistant_turn(tmp_path: Path):
    p = tmp_path / "test.jsonl"
    _write_test_jsonl(p, {"c1": _extraction_dict("One"), "c2": _extraction_dict("Two")})
    golds = load_golds(p)
    assert [cid for cid, _ in golds] == ["c1", "c2"]
    assert golds[0][1].document_name == "One"


def test_load_predictions_keys_by_id(tmp_path: Path):
    p = tmp_path / "preds.json"
    _write_predictions(p, [_valid_record("c1", _extraction_dict()), _invalid_record("c2")])
    by_id = load_predictions(p)
    assert set(by_id) == {"c1", "c2"}
    assert by_id["c1"]["is_valid_json"] is True


def test_load_predictions_missing_file_returns_empty(tmp_path: Path):
    assert load_predictions(tmp_path / "nope.json") == {}


# ---------------------------------------------------------------------------
# score_model
# ---------------------------------------------------------------------------


def test_score_model_perfect_predictions():
    golds = [("c1", ContractExtraction.model_validate(_extraction_dict("X")))]
    preds = {"c1": _valid_record("c1", _extraction_dict("X"))}
    result = score_model(preds, golds)
    assert result["n_contracts"] == 1
    assert result["json_valid"] == 1
    assert result["json_validity_rate"] == 1.0
    assert result["per_field_match_rate_CAVEATED"]["overall_f1"] == pytest.approx(1.0)


def test_score_model_all_invalid_zero_validity():
    golds = [("c1", ContractExtraction.model_validate(_extraction_dict("X")))]
    preds = {"c1": _invalid_record("c1")}
    result = score_model(preds, golds)
    assert result["json_valid"] == 0
    assert result["json_validity_rate"] == 0.0
    # document_name gold is "X", empty pred is None → that field is wrong
    assert result["per_field_match_rate_CAVEATED"]["document_name"] == 0.0


def test_score_model_missing_prediction_counts_as_invalid():
    """A gold contract with no matching prediction is scored, not skipped."""
    golds = [("c1", ContractExtraction.model_validate(_extraction_dict("X")))]
    result = score_model({}, golds)  # no predictions at all
    assert result["n_contracts"] == 1
    assert result["json_valid"] == 0
    assert result["json_validity_rate"] == 0.0


def test_score_model_aligns_by_contract_id():
    """Prediction order must not matter; alignment is by contract_id."""
    golds = [
        ("c1", ContractExtraction.model_validate(_extraction_dict("One"))),
        ("c2", ContractExtraction.model_validate(_extraction_dict("Two"))),
    ]
    # Provide predictions in reversed insertion order; both correct.
    preds = {
        "c2": _valid_record("c2", _extraction_dict("Two")),
        "c1": _valid_record("c1", _extraction_dict("One")),
    }
    result = score_model(preds, golds)
    assert result["json_validity_rate"] == 1.0
    assert result["per_field_match_rate_CAVEATED"]["document_name"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_comparison
# ---------------------------------------------------------------------------


def test_build_comparison_all_models(tmp_path: Path):
    test_path = tmp_path / "test.jsonl"
    _write_test_jsonl(test_path, {"c1": _extraction_dict("X")})

    base = tmp_path / "base.json"
    prompt = tmp_path / "prompt.json"
    ft = tmp_path / "ft.json"
    _write_predictions(base, [_invalid_record("c1")])
    _write_predictions(prompt, [_invalid_record("c1")])
    _write_predictions(ft, [_valid_record("c1", _extraction_dict("X"))])

    comparison = build_comparison(
        {"naive": str(base), "strong_prompt": str(prompt), "finetuned": str(ft)},
        test_path,
    )
    assert comparison["n_contracts"] == 1
    assert set(comparison["models"]) == {"naive", "strong_prompt", "finetuned"}
    assert comparison["models"]["finetuned"]["json_validity_rate"] == 1.0
    assert comparison["models"]["naive"]["json_validity_rate"] == 0.0


def test_build_comparison_skips_missing_files(tmp_path: Path):
    test_path = tmp_path / "test.jsonl"
    _write_test_jsonl(test_path, {"c1": _extraction_dict("X")})

    ft = tmp_path / "ft.json"
    _write_predictions(ft, [_valid_record("c1", _extraction_dict("X"))])

    comparison = build_comparison(
        {
            "naive": str(tmp_path / "absent_base.json"),
            "strong_prompt": str(tmp_path / "absent_prompt.json"),
            "finetuned": str(ft),
        },
        test_path,
    )
    assert set(comparison["models"]) == {"finetuned"}


def test_build_comparison_raises_on_empty_gold(tmp_path: Path):
    empty = tmp_path / "test.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="No gold examples"):
        build_comparison({"finetuned": str(tmp_path / "x.json")}, empty)


# ---------------------------------------------------------------------------
# format_table / write_summary
# ---------------------------------------------------------------------------


def test_format_table_lists_present_models():
    comparison = {
        "n_contracts": 1,
        "models": {
            "naive": {"n_contracts": 1, "json_valid": 0, "json_validity_rate": 0.0,
                      "per_field_match_rate_CAVEATED": {"overall_f1": 0.40}},
            "finetuned": {"n_contracts": 1, "json_valid": 1, "json_validity_rate": 1.0,
                          "per_field_match_rate_CAVEATED": {"overall_f1": 0.95}},
        },
    }
    table = format_table(comparison)
    assert "Naive baseline" in table
    assert "Fine-tuned (QLoRA)" in table
    # strong_prompt absent from this comparison → must not appear
    assert "Strong-prompt baseline" not in table
    assert "100%" in table  # finetuned validity


def test_write_summary_roundtrip(tmp_path: Path):
    out = tmp_path / "nested" / "comparison_summary.json"
    comparison = {"n_contracts": 0, "models": {}}
    write_summary(comparison, out)
    assert out.exists()
    assert json.loads(out.read_text()) == comparison


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        compare.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--base" in out
    assert "--finetuned" in out
    assert "--output" in out


def test_main_end_to_end_writes_summary(tmp_path: Path):
    test_path = tmp_path / "test.jsonl"
    _write_test_jsonl(test_path, {"c1": _extraction_dict("X")})
    ft = tmp_path / "ft.json"
    _write_predictions(ft, [_valid_record("c1", _extraction_dict("X"))])
    out = tmp_path / "comparison_summary.json"

    rc = compare.main(
        [
            "--base", str(tmp_path / "absent.json"),
            "--prompt", str(tmp_path / "absent2.json"),
            "--finetuned", str(ft),
            "--test", str(test_path),
            "--output", str(out),
        ]
    )
    assert rc == 0
    summary = json.loads(out.read_text())
    assert summary["models"]["finetuned"]["json_validity_rate"] == 1.0


def test_main_returns_1_when_no_predictions(tmp_path: Path):
    test_path = tmp_path / "test.jsonl"
    _write_test_jsonl(test_path, {"c1": _extraction_dict("X")})
    out = tmp_path / "summary.json"
    rc = compare.main(
        [
            "--base", str(tmp_path / "a.json"),
            "--prompt", str(tmp_path / "b.json"),
            "--finetuned", str(tmp_path / "c.json"),
            "--test", str(test_path),
            "--output", str(out),
        ]
    )
    assert rc == 1
    assert not out.exists()
