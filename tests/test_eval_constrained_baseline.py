"""CPU-only tests for evaluation/eval_constrained_baseline.py.

The constrained generation itself is GPU-only (lazy lm-format-enforcer import);
these tests cover the pure schema builder and the driver with a mocked generator.
"""

from __future__ import annotations

import json

import pytest

from extractor.schemas import ContractExtraction
from evaluation import eval_constrained_baseline as ecb
from evaluation._runner import USER_PROMPT_PREFIX


# --------------------------------------------------------------------------- schema


def test_build_json_schema_shape():
    s = ecb.build_json_schema()
    assert s["type"] == "object"
    assert s["additionalProperties"] is False
    assert len(s["required"]) == 12
    assert set(s["required"]) == set(ContractExtraction.model_fields)
    assert s["properties"]["parties"] == {"type": "array", "items": {"type": "string"}}
    assert s["properties"]["governing_law"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}


def test_build_json_schema_order_matches_model():
    s = ecb.build_json_schema()
    assert list(s["properties"]) == list(ContractExtraction.model_fields)


def test_module_imports_without_loading_a_model():
    # The driver + schema are importable with no torch / transformers / lmformatenforcer.
    assert hasattr(ecb, "generate_constrained")
    assert hasattr(ecb, "run_constrained_baseline")


# --------------------------------------------------------------------------- driver


def _write_test_jsonl(path, ids):
    with open(path, "w", encoding="utf-8") as fh:
        for cid in ids:
            row = {
                "contract_id": cid,
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": USER_PROMPT_PREFIX + f"Contract body for {cid}"},
                    {"role": "assistant", "content": "{}"},
                ],
            }
            fh.write(json.dumps(row) + "\n")


def _valid_extraction_json():
    return json.dumps(
        {f: ([] if f == "parties" else None) for f in ContractExtraction.model_fields}
    )


def test_run_constrained_baseline_end_to_end(tmp_path, monkeypatch):
    test_file = tmp_path / "test.jsonl"
    _write_test_jsonl(test_file, ["A", "B"])
    monkeypatch.setattr(ecb, "generate_constrained", lambda *a, **k: _valid_extraction_json())
    out = tmp_path / "preds.json"
    rc = ecb.run_constrained_baseline(
        lambda t: "PROMPT", ecb.build_json_schema(), test_file, out,
        tokenizer=object(), model=object(),
    )
    assert rc == 0
    recs = json.load(open(out))
    assert [r["contract_id"] for r in recs] == ["A", "B"]
    assert all(r["is_valid_json"] for r in recs)


def test_run_constrained_baseline_persists_invalid(tmp_path, monkeypatch):
    test_file = tmp_path / "test.jsonl"
    _write_test_jsonl(test_file, ["A"])
    # A truncated (still-invalid) generation must be recorded, not dropped.
    monkeypatch.setattr(ecb, "generate_constrained", lambda *a, **k: '{"document_name":')
    out = tmp_path / "preds.json"
    rc = ecb.run_constrained_baseline(
        lambda t: "P", ecb.build_json_schema(), test_file, out,
        tokenizer=object(), model=object(),
    )
    assert rc == 0
    recs = json.load(open(out))
    assert recs[0]["is_valid_json"] is False
    assert recs[0]["parsed"] is None


def test_run_constrained_baseline_empty_test_returns_1(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    rc = ecb.run_constrained_baseline(
        lambda t: "P", ecb.build_json_schema(), empty, tmp_path / "o.json",
        tokenizer=object(), model=object(),
    )
    assert rc == 1


def test_run_constrained_baseline_respects_limit(tmp_path, monkeypatch):
    test_file = tmp_path / "test.jsonl"
    _write_test_jsonl(test_file, ["A", "B", "C"])
    monkeypatch.setattr(ecb, "generate_constrained", lambda *a, **k: "{}")
    out = tmp_path / "p.json"
    ecb.run_constrained_baseline(
        lambda t: "P", ecb.build_json_schema(), test_file, out,
        tokenizer=object(), model=object(), limit=2,
    )
    assert len(json.load(open(out))) == 2


def test_main_help_exits_zero():
    with pytest.raises(SystemExit) as e:
        ecb.main(["--help"])
    assert e.value.code == 0


def test_main_returns_1_on_empty_train(tmp_path):
    empty_train = tmp_path / "train.jsonl"
    empty_train.write_text("")
    test_file = tmp_path / "test.jsonl"
    _write_test_jsonl(test_file, ["A"])
    rc = ecb.main(
        ["--train", str(empty_train), "--input", str(test_file), "--output", str(tmp_path / "o.json")]
    )
    assert rc == 1
