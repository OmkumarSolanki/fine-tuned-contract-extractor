"""CPU-only tests for evaluation/analysis.py (stats primitives + analysis blocks)."""

from __future__ import annotations

import json

import pytest

from extractor.schemas import ContractExtraction
from evaluation.analysis import (
    always_null_floor,
    bootstrap_ci,
    check_leakage,
    classify_failure,
    failure_mode_breakdown,
    mcnemar,
    null_confusion,
    per_example_overall,
    wilson_interval,
)
from evaluation.metrics import overall_f1


# --------------------------------------------------------------------------- wilson


def test_wilson_interval_bounds_and_order():
    lo, hi = wilson_interval(49, 51)
    assert 0.0 <= lo < hi <= 1.0
    # 49/51 ≈ 0.96 — the interval should sit high and exclude the strong baseline.
    assert lo > 0.8 and hi > 0.95


def test_wilson_interval_extremes_stay_in_range():
    assert wilson_interval(0, 51)[0] == 0.0  # zero successes → lower bound 0
    assert wilson_interval(51, 51)[1] == pytest.approx(1.0)  # all successes → upper bound ≈1


def test_wilson_interval_zero_n():
    assert wilson_interval(0, 0) == (0.0, 0.0)


# --------------------------------------------------------------------------- bootstrap


def test_bootstrap_ci_deterministic_for_seed():
    vals = [0.0, 0.5, 1.0, 0.25, 0.75]
    assert bootstrap_ci(vals, n_boot=2000, seed=42) == bootstrap_ci(vals, n_boot=2000, seed=42)


def test_bootstrap_ci_brackets_the_mean():
    vals = [0.6, 0.7, 0.8, 0.9, 0.7, 0.75]
    mean = sum(vals) / len(vals)
    lo, hi = bootstrap_ci(vals, n_boot=3000, seed=1)
    assert lo <= mean <= hi


def test_bootstrap_ci_empty():
    assert bootstrap_ci([], n_boot=100) == (0.0, 0.0)


# --------------------------------------------------------------------------- mcnemar


def test_mcnemar_all_improvement_is_significant():
    # B fixes 10 items A got wrong, never regresses → strongly significant.
    a = [False] * 10 + [True] * 5
    b = [True] * 10 + [True] * 5
    res = mcnemar(b, a)  # treat b as "model A-arg", a as "model B-arg"
    assert res["b"] == 10 and res["c"] == 0
    assert res["p_value"] < 0.01


def test_mcnemar_symmetric_disagreement_not_significant():
    a = [True, False, True, False]
    b = [False, True, False, True]
    res = mcnemar(a, b)
    assert res["b"] == res["c"]
    assert res["p_value"] > 0.05


def test_mcnemar_no_discordant_pairs():
    a = [True, True, False]
    res = mcnemar(a, a)
    assert res == {"b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0}


def test_mcnemar_length_mismatch_raises():
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])


# --------------------------------------------------------------------------- per-example


def test_per_example_mean_equals_overall_f1():
    preds = [
        ContractExtraction(document_name="A", parties=["x", "y"]),
        ContractExtraction(document_name="B"),
    ]
    golds = [
        ContractExtraction(document_name="A", parties=["x", "y"]),
        ContractExtraction(document_name="C"),
    ]
    per_ex = per_example_overall(preds, golds)
    assert len(per_ex) == 2
    assert sum(per_ex) / len(per_ex) == pytest.approx(overall_f1(preds, golds)["overall_f1"])


# --------------------------------------------------------------------------- floor


def test_always_null_floor_perfect_on_all_null_golds():
    golds = [ContractExtraction(), ContractExtraction()]
    floor = always_null_floor(golds)
    assert floor["overall_f1"] == pytest.approx(1.0)


def test_always_null_floor_returns_overall_key():
    golds = [ContractExtraction(document_name="A", parties=["x"])]
    assert "overall_f1" in always_null_floor(golds)


# --------------------------------------------------------------------------- null confusion


def test_null_confusion_counts_hallucination_and_miss():
    # gold: document_name present, governing_law null.
    gold = ContractExtraction(document_name="A")
    # pred: invents governing_law (hallucination), drops document_name (miss).
    pred = ContractExtraction(governing_law="invented")
    conf = null_confusion([pred], [gold])
    assert conf["governing_law"]["fp_hallucination"] == 1
    assert conf["document_name"]["fn_miss"] == 1
    agg = conf["_aggregate"]
    assert agg["fp_hallucination"] == 1 and agg["fn_miss"] == 1


def test_null_confusion_parties_present_logic():
    # Non-empty parties = present; empty = null.
    gold = ContractExtraction(parties=["x"])
    pred_present = ContractExtraction(parties=["x"])
    pred_empty = ContractExtraction(parties=[])
    assert null_confusion([pred_present], [gold])["parties"]["tp"] == 1
    assert null_confusion([pred_empty], [gold])["parties"]["fn_miss"] == 1


def test_null_confusion_rates_when_only_true_negatives():
    # All gold+pred null for a field → one true-negative: no positives (precision
    # undefined → None), and zero hallucination among the truly-empty (rate 0.0).
    conf = null_confusion([ContractExtraction()], [ContractExtraction()])
    assert conf["governing_law"]["precision"] is None
    assert conf["governing_law"]["recall"] is None
    assert conf["governing_law"]["hallucination_rate"] == 0.0


# --------------------------------------------------------------------------- leakage


# --------------------------------------------------------------------------- failure modes


def test_classify_failure_empty():
    assert classify_failure("") == "empty"
    assert classify_failure("   \n ") == "empty"


def test_classify_failure_markdown_fence():
    assert classify_failure('```json\n{"document_name": "A"}\n```') == "markdown_fence"


def test_classify_failure_no_json_object():
    assert classify_failure("I cannot extract this contract.") == "no_json_object"


def test_classify_failure_truncated():
    assert classify_failure('{"document_name": "A", "parties": ["X"') == "truncated"


def test_classify_failure_malformed_json():
    assert classify_failure('{"document_name": "A" "parties": []}') == "malformed_json"


def test_classify_failure_prose_around_json():
    valid = json.dumps({f: None for f in ContractExtraction.model_fields} | {"parties": []})
    assert classify_failure(f"Here is the answer: {valid} Hope this helps!") == "prose_around_json"


def test_classify_failure_schema_mismatch():
    # Clean JSON, but `parties` must be a list — wrong type fails schema validation.
    assert classify_failure('{"parties": 123}') == "schema_mismatch"


def test_failure_mode_breakdown_counts_only_invalids():
    records = [
        {"is_valid_json": True, "raw_output": "{}"},
        {"is_valid_json": False, "raw_output": ""},
        {"is_valid_json": False, "raw_output": "```\n{}\n```"},
        {"is_valid_json": False, "raw_output": "```\n{}\n```"},
    ]
    out = failure_mode_breakdown(records)
    assert out["n_invalid"] == 3
    assert out["by_reason"]["markdown_fence"] == 2
    assert out["by_reason"]["empty"] == 1
    # most-common-first ordering
    assert list(out["by_reason"])[0] == "markdown_fence"


def _write_jsonl(path, ids):
    with open(path, "w", encoding="utf-8") as fh:
        for cid in ids:
            fh.write(json.dumps({"contract_id": cid, "messages": []}) + "\n")


def test_check_leakage_clean(tmp_path):
    _write_jsonl(tmp_path / "train.jsonl", ["a", "b", "c"])
    _write_jsonl(tmp_path / "val.jsonl", ["d"])
    _write_jsonl(tmp_path / "test.jsonl", ["e", "f"])
    res = check_leakage(tmp_path / "train.jsonl", tmp_path / "val.jsonl", tmp_path / "test.jsonl")
    assert res["clean"] is True
    assert res["n_train"] == 3 and res["n_test"] == 2


def test_check_leakage_detects_overlap(tmp_path):
    _write_jsonl(tmp_path / "train.jsonl", ["a", "b", "shared"])
    _write_jsonl(tmp_path / "val.jsonl", ["d"])
    _write_jsonl(tmp_path / "test.jsonl", ["shared", "f"])
    res = check_leakage(tmp_path / "train.jsonl", tmp_path / "val.jsonl", tmp_path / "test.jsonl")
    assert res["clean"] is False
    assert res["overlap_train_test"] == ["shared"]
