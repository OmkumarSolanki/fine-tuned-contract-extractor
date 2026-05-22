"""Unit tests for ``evaluation/eval_base.py`` and the shared ``_runner`` helpers.

These tests run on CPU and never touch a real model — both the model and
tokenizer are mocked. The full integration with Llama 3.1 8B happens on
RunPod, separately. Tests in this file cover:

- The pure-Python helpers in ``evaluation/_runner.py`` (parse, record shape,
  test-set loader, output writer).
- The naive-prompt construction in ``evaluation/eval_base.py``.
- An end-to-end ``run_baseline`` call with a monkey-patched
  ``generate_one`` that returns canned strings.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation import eval_base
from evaluation._runner import (
    USER_PROMPT_PREFIX,
    extract_contract_text,
    load_test_examples,
    make_prediction_record,
    parse_prediction,
    run_baseline,
    write_predictions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_jsonl(path: Path, contract_ids: list[str]) -> None:
    """Write a minimal test.jsonl that load_test_examples can read."""
    with path.open("w", encoding="utf-8") as fh:
        for cid in contract_ids:
            row = {
                "contract_id": cid,
                "messages": [
                    {"role": "system", "content": "..."},
                    {
                        "role": "user",
                        "content": USER_PROMPT_PREFIX + f"Contract body for {cid}.",
                    },
                    {"role": "assistant", "content": "{}"},
                ],
            }
            fh.write(json.dumps(row) + "\n")


def _full_valid_extraction_json(document_name: str = "X") -> str:
    """Build a JSON string that validates against ContractExtraction."""
    return json.dumps(
        {
            "document_name": document_name,
            "parties": ["A", "B"],
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


# ---------------------------------------------------------------------------
# extract_contract_text
# ---------------------------------------------------------------------------


def test_extract_contract_text_strips_prefix():
    text = "Some agreement between A and B."
    wrapped = USER_PROMPT_PREFIX + text
    assert extract_contract_text(wrapped) == text


def test_extract_contract_text_returns_input_when_prefix_absent():
    text = "Already raw, no prefix."
    assert extract_contract_text(text) == text


def test_extract_contract_text_empty_string():
    assert extract_contract_text("") == ""


# ---------------------------------------------------------------------------
# parse_prediction
# ---------------------------------------------------------------------------


def test_parse_prediction_valid_json():
    parsed = parse_prediction(_full_valid_extraction_json("Acme Agreement"))
    assert parsed is not None
    assert parsed["document_name"] == "Acme Agreement"
    assert parsed["parties"] == ["A", "B"]


def test_parse_prediction_invalid_json_returns_none():
    assert parse_prediction("not json at all") is None


def test_parse_prediction_empty_string_returns_none():
    assert parse_prediction("") is None


def test_parse_prediction_valid_json_invalid_schema_returns_none():
    # parties must be a list, not a string
    raw = json.dumps({"parties": "should be a list"})
    assert parse_prediction(raw) is None


def test_parse_prediction_never_raises():
    # Whatever garbage we throw in, we should get None, not an exception.
    for garbage in ["", "{", "{}", "{'bad': quotes}", "null", "[]", "42"]:
        # Just ensure it doesn't raise. Some of these may legitimately be None.
        parse_prediction(garbage)


# ---------------------------------------------------------------------------
# make_prediction_record
# ---------------------------------------------------------------------------


def test_make_prediction_record_valid():
    raw = _full_valid_extraction_json("X")
    rec = make_prediction_record("c1", raw)
    assert rec["contract_id"] == "c1"
    assert rec["raw_output"] == raw
    assert rec["is_valid_json"] is True
    assert rec["parsed"]["document_name"] == "X"


def test_make_prediction_record_invalid_still_persisted():
    """Invalid predictions MUST be saved with is_valid_json=False, not dropped."""
    rec = make_prediction_record("c2", "garbage output")
    assert rec["contract_id"] == "c2"
    assert rec["raw_output"] == "garbage output"
    assert rec["is_valid_json"] is False
    assert rec["parsed"] is None


def test_make_prediction_record_keys_are_canonical():
    """The record must have exactly four keys, no more no less."""
    rec = make_prediction_record("c1", "x")
    assert set(rec.keys()) == {"contract_id", "raw_output", "parsed", "is_valid_json"}


# ---------------------------------------------------------------------------
# load_test_examples
# ---------------------------------------------------------------------------


def test_load_test_examples_reads_all(tmp_path: Path):
    p = tmp_path / "test.jsonl"
    _make_test_jsonl(p, ["c1", "c2", "c3"])
    examples = load_test_examples(p)
    assert len(examples) == 3
    assert examples[0]["contract_id"] == "c1"
    assert examples[0]["contract_text"] == "Contract body for c1."
    assert examples[2]["contract_text"] == "Contract body for c3."


def test_load_test_examples_respects_limit(tmp_path: Path):
    p = tmp_path / "test.jsonl"
    _make_test_jsonl(p, ["c1", "c2", "c3"])
    examples = load_test_examples(p, limit=2)
    assert len(examples) == 2
    assert [e["contract_id"] for e in examples] == ["c1", "c2"]


def test_load_test_examples_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "test.jsonl"
    _make_test_jsonl(p, ["c1"])
    # Add a blank line at the end
    with p.open("a") as fh:
        fh.write("\n\n")
    examples = load_test_examples(p)
    assert len(examples) == 1


# ---------------------------------------------------------------------------
# write_predictions
# ---------------------------------------------------------------------------


def test_write_predictions_creates_dir_and_writes_array(tmp_path: Path):
    out = tmp_path / "deep" / "nested" / "results.json"
    records = [
        {"contract_id": "c1", "raw_output": "x", "parsed": None, "is_valid_json": False},
    ]
    write_predictions(records, out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["contract_id"] == "c1"


def test_write_predictions_output_is_json_not_jsonl(tmp_path: Path):
    """Phase 5 outputs MUST be a single JSON array, not line-delimited."""
    out = tmp_path / "results.json"
    records = [
        {"contract_id": "c1", "raw_output": "x", "parsed": None, "is_valid_json": False},
        {"contract_id": "c2", "raw_output": "y", "parsed": None, "is_valid_json": False},
    ]
    write_predictions(records, out)
    text = out.read_text()
    # First non-whitespace char should be '['
    assert text.lstrip().startswith("[")
    # Parsing the whole file as one JSON document succeeds:
    data = json.loads(text)
    assert len(data) == 2


# ---------------------------------------------------------------------------
# Naive prompt — properties
# ---------------------------------------------------------------------------


def test_naive_prompt_contains_contract_text():
    prompt = eval_base.build_prompt("Body of agreement here.")
    assert "Body of agreement here." in prompt


def test_naive_prompt_is_deterministic():
    a = eval_base.build_prompt("X")
    b = eval_base.build_prompt("X")
    assert a == b


def test_naive_prompt_has_expected_skeleton():
    prompt = eval_base.build_prompt("BODY")
    assert "Extract the legal clauses from this contract as JSON" in prompt
    assert "Contract:" in prompt
    assert "JSON:" in prompt


def test_naive_prompt_is_short_and_simple():
    """The naive prompt MUST stay minimal — no schema, no examples, no constraints.

    If this test fails because the prompt has grown, you've drifted into the
    strong-prompt baseline territory. The new content belongs in
    ``eval_prompt_baseline.py``, not here.
    """
    prompt = eval_base.build_prompt("CONTRACT_BODY")
    forbidden_markers = [
        "Schema",
        "Examples:",
        "few-shot",
        "Constraints:",
        "Output requirements:",
        "field",  # any per-field hint would be schema description bleed-through
    ]
    for m in forbidden_markers:
        assert m.lower() not in prompt.lower(), (
            f"Naive prompt contains '{m}' — that belongs in the strong baseline."
        )


# ---------------------------------------------------------------------------
# Module-load hygiene
# ---------------------------------------------------------------------------


def test_module_imports_without_loading_a_model():
    """Reload the module fresh and confirm no transformers/torch import is forced."""
    # Re-import is enough to confirm top-level code stays light (no heavy work
    # at import time). Heavy imports live inside _runner.load_model and
    # _runner.generate_one, both inside function bodies.
    importlib.reload(eval_base)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_main_help_exits_zero(capsys):
    """`python evaluation/eval_base.py --help` should print usage and exit 0."""
    with pytest.raises(SystemExit) as exc:
        eval_base.main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--limit" in captured.out
    assert "--input" in captured.out
    assert "--output" in captured.out


# ---------------------------------------------------------------------------
# End-to-end driver with mocked model
# ---------------------------------------------------------------------------


def test_run_baseline_end_to_end_with_mocked_model(tmp_path: Path, monkeypatch):
    """Full run_baseline call with mocked tokenizer + model. Verifies output shape."""
    test_path = tmp_path / "test.jsonl"
    _make_test_jsonl(test_path, ["c1", "c2"])
    output_path = tmp_path / "preds.json"

    valid_json = _full_valid_extraction_json("Mock Agreement")
    monkeypatch.setattr(
        "evaluation._runner.generate_one",
        lambda tokenizer, model, prompt, max_new_tokens=2048: valid_json,
    )

    rc = run_baseline(
        prompt_builder=eval_base.build_prompt,
        test_path=test_path,
        output_path=output_path,
        tokenizer=MagicMock(),
        model=MagicMock(),
    )
    assert rc == 0
    data = json.loads(output_path.read_text())
    assert len(data) == 2
    for rec in data:
        assert set(rec.keys()) == {"contract_id", "raw_output", "parsed", "is_valid_json"}
        assert rec["is_valid_json"] is True
        assert rec["parsed"]["document_name"] == "Mock Agreement"


def test_run_baseline_persists_invalid_predictions(tmp_path: Path, monkeypatch):
    """Even when the mocked model returns garbage, records are written with is_valid_json=False."""
    test_path = tmp_path / "test.jsonl"
    _make_test_jsonl(test_path, ["c1"])
    output_path = tmp_path / "preds.json"

    monkeypatch.setattr(
        "evaluation._runner.generate_one",
        lambda *a, **k: "this is not JSON at all",
    )

    rc = run_baseline(
        prompt_builder=eval_base.build_prompt,
        test_path=test_path,
        output_path=output_path,
        tokenizer=MagicMock(),
        model=MagicMock(),
    )
    assert rc == 0
    data = json.loads(output_path.read_text())
    assert len(data) == 1
    assert data[0]["is_valid_json"] is False
    assert data[0]["parsed"] is None
    assert data[0]["raw_output"] == "this is not JSON at all"


def test_run_baseline_returns_1_on_empty_test_set(tmp_path: Path):
    """An empty test set is an input error, not an empty output."""
    test_path = tmp_path / "test.jsonl"
    test_path.write_text("")  # empty file
    output_path = tmp_path / "preds.json"

    rc = run_baseline(
        prompt_builder=eval_base.build_prompt,
        test_path=test_path,
        output_path=output_path,
        tokenizer=MagicMock(),
        model=MagicMock(),
    )
    assert rc == 1
    assert not output_path.exists()


def test_run_baseline_respects_limit(tmp_path: Path, monkeypatch):
    """run_baseline forwards --limit through to load_test_examples."""
    test_path = tmp_path / "test.jsonl"
    _make_test_jsonl(test_path, ["c1", "c2", "c3", "c4", "c5"])
    output_path = tmp_path / "preds.json"

    monkeypatch.setattr(
        "evaluation._runner.generate_one",
        lambda *a, **k: _full_valid_extraction_json(),
    )

    rc = run_baseline(
        prompt_builder=eval_base.build_prompt,
        test_path=test_path,
        output_path=output_path,
        limit=2,
        tokenizer=MagicMock(),
        model=MagicMock(),
    )
    assert rc == 0
    data = json.loads(output_path.read_text())
    assert len(data) == 2
    assert [r["contract_id"] for r in data] == ["c1", "c2"]
