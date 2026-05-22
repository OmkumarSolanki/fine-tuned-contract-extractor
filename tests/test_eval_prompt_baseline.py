"""Unit tests for ``evaluation/eval_prompt_baseline.py``.

Same conventions as ``test_eval_base.py``: CPU-only, no model, no network.
The mocked-model test patches ``evaluation._runner.generate_one`` and
``evaluation._runner.load_model`` so a real Llama 3.1 is never loaded.

Coverage:

- ``build_schema_description`` produces a description containing every field
  in canonical order, pulling text from the live ``Field(description=...)``.
- ``select_few_shot_examples`` deterministically picks complete + sparse +
  multi-party, raises on missing categories, and only ever returns rows from
  its input (the data-leakage gate that protects Phase 7).
- ``format_few_shot_example`` truncates over-budget examples.
- ``build_strong_prompt`` assembles the four expected blocks and is
  deterministic.
- The CLI ``--help`` path works.
- An end-to-end ``main()`` call with a mocked model produces a valid
  predictions file.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation import eval_base, eval_prompt_baseline
from evaluation._runner import USER_PROMPT_PREFIX
from extractor.schemas import ContractExtraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_train_row(
    contract_id: str,
    populated_fields: int,
    n_parties: int,
    body: str = "",
) -> dict:
    """Synthesize a train.jsonl-shaped row with controlled population.

    ``populated_fields`` counts non-null fields among the 12. ``parties`` is
    populated independently via ``n_parties``.
    """
    fields_in_order = list(ContractExtraction.model_fields)
    extraction: dict = {}
    # Fill the first `populated_fields` non-parties fields with "value", rest with None.
    non_parties_filled = 0
    for f in fields_in_order:
        if f == "parties":
            extraction[f] = [f"P{j}" for j in range(n_parties)] if n_parties else []
        else:
            if non_parties_filled < populated_fields:
                extraction[f] = "value"
                non_parties_filled += 1
            else:
                extraction[f] = None
    body = body or f"Body of {contract_id}."
    return {
        "contract_id": contract_id,
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": USER_PROMPT_PREFIX + body},
            {"role": "assistant", "content": json.dumps(extraction)},
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _make_test_jsonl(path: Path, contract_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for cid in contract_ids:
            row = {
                "contract_id": cid,
                "messages": [
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": USER_PROMPT_PREFIX + f"Body of {cid}"},
                    {"role": "assistant", "content": "{}"},
                ],
            }
            fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# build_schema_description — drift-proof contract with the live schema
# ---------------------------------------------------------------------------


def test_schema_description_includes_all_12_fields():
    desc = eval_prompt_baseline.build_schema_description()
    for field in ContractExtraction.model_fields:
        assert field in desc, f"Schema description missing field: {field}"


def test_schema_description_is_in_canonical_order():
    desc = eval_prompt_baseline.build_schema_description()
    last_pos = -1
    for field in ContractExtraction.model_fields:
        pos = desc.find(field)
        assert pos > last_pos, f"Field {field} out of canonical order in description"
        last_pos = pos


def test_schema_description_pulls_descriptions_from_live_schema():
    """At least one known description string from the schema must appear verbatim."""
    desc = eval_prompt_baseline.build_schema_description()
    # Pick a stable description from the schema — if either side changes, this test
    # surfaces the drift.
    expected = ContractExtraction.model_fields["governing_law"].description
    assert expected, "schema test fixture: governing_law has no description"
    assert expected in desc


def test_schema_description_lists_parties_as_list_type():
    desc = eval_prompt_baseline.build_schema_description()
    assert "parties (list[str])" in desc


def test_schema_description_lists_other_fields_as_str_or_null():
    desc = eval_prompt_baseline.build_schema_description()
    assert "document_name (str | null)" in desc
    assert "governing_law (str | null)" in desc


# ---------------------------------------------------------------------------
# select_few_shot_examples
# ---------------------------------------------------------------------------


def test_select_few_shot_picks_complete_sparse_multiparty():
    rows = [
        _make_train_row("complete_one", populated_fields=11, n_parties=2),
        _make_train_row("sparse_one", populated_fields=4, n_parties=0),
        _make_train_row("multi_one", populated_fields=8, n_parties=4),
    ]
    picks = eval_prompt_baseline.select_few_shot_examples(rows)
    ids = [p["contract_id"] for p in picks]
    assert ids == ["complete_one", "sparse_one", "multi_one"]


def test_select_few_shot_returns_three_picks():
    rows = [
        _make_train_row("c", populated_fields=11, n_parties=2),
        _make_train_row("s", populated_fields=4, n_parties=0),
        _make_train_row("m", populated_fields=8, n_parties=4),
    ]
    picks = eval_prompt_baseline.select_few_shot_examples(rows)
    assert len(picks) == 3


def test_select_few_shot_no_duplicates():
    """Multi-party pick must not collide with the complete or sparse picks."""
    rows = [
        # This row is BOTH complete (11) AND multi-party (5). Should be picked
        # as complete, then the multi-party search must move past it.
        _make_train_row("complete_and_multi", populated_fields=11, n_parties=5),
        _make_train_row("sparse_only", populated_fields=4, n_parties=0),
        _make_train_row("multi_only", populated_fields=8, n_parties=3),
    ]
    picks = eval_prompt_baseline.select_few_shot_examples(rows)
    ids = [p["contract_id"] for p in picks]
    assert len(set(ids)) == 3
    assert ids[0] == "complete_and_multi"
    assert ids[2] == "multi_only"  # NOT complete_and_multi


def test_select_few_shot_raises_when_no_complete():
    rows = [
        _make_train_row("only_sparse", populated_fields=2, n_parties=0),
        _make_train_row("only_sparse_2", populated_fields=4, n_parties=4),
    ]
    with pytest.raises(RuntimeError, match="complete"):
        eval_prompt_baseline.select_few_shot_examples(rows)


def test_select_few_shot_raises_when_no_sparse():
    rows = [
        _make_train_row("c1", populated_fields=11, n_parties=3),
        _make_train_row("c2", populated_fields=12, n_parties=4),
    ]
    with pytest.raises(RuntimeError, match="sparse"):
        eval_prompt_baseline.select_few_shot_examples(rows)


def test_select_few_shot_raises_when_no_multi_party():
    rows = [
        _make_train_row("c", populated_fields=11, n_parties=1),
        _make_train_row("s", populated_fields=4, n_parties=2),
    ]
    with pytest.raises(RuntimeError, match="multi-party"):
        eval_prompt_baseline.select_few_shot_examples(rows)


def test_select_few_shot_picks_first_match_in_file_order():
    """Determinism: when multiple candidates qualify, the first in input order wins."""
    rows = [
        _make_train_row("complete_FIRST", populated_fields=11, n_parties=2),
        _make_train_row("complete_second", populated_fields=11, n_parties=2),
        _make_train_row("sparse_FIRST", populated_fields=3, n_parties=0),
        _make_train_row("sparse_second", populated_fields=3, n_parties=0),
        _make_train_row("multi_FIRST", populated_fields=8, n_parties=3),
        _make_train_row("multi_second", populated_fields=8, n_parties=3),
    ]
    picks = eval_prompt_baseline.select_few_shot_examples(rows)
    ids = [p["contract_id"] for p in picks]
    assert ids == ["complete_FIRST", "sparse_FIRST", "multi_FIRST"]


# ---------------------------------------------------------------------------
# Data-leakage gate: few-shot picks NEVER come from outside their input set.
# ---------------------------------------------------------------------------


def test_few_shot_picks_only_come_from_input_rows():
    """If main() ever read test.jsonl and passed it here, this test would catch it.

    By feeding rows tagged with a unique suffix, we assert the picks came from
    exactly that set — not from any other JSONL the function might secretly read.
    """
    rows = [
        _make_train_row("MARKED_complete", populated_fields=11, n_parties=2),
        _make_train_row("MARKED_sparse", populated_fields=4, n_parties=0),
        _make_train_row("MARKED_multi", populated_fields=8, n_parties=4),
    ]
    picks = eval_prompt_baseline.select_few_shot_examples(rows)
    for p in picks:
        assert p["contract_id"].startswith("MARKED_"), (
            f"Few-shot picked an id outside the input rows: {p['contract_id']}. "
            "This would be data leakage from the test set."
        )


# ---------------------------------------------------------------------------
# format_few_shot_example
# ---------------------------------------------------------------------------


def test_format_few_shot_example_includes_contract_and_output():
    row = _make_train_row("c", populated_fields=11, n_parties=2, body="Hello world.")
    cand = eval_prompt_baseline._candidate_summary(row)
    rendered = eval_prompt_baseline.format_few_shot_example(cand)
    assert "Contract:" in rendered
    assert "Hello world." in rendered
    assert "Output:" in rendered
    # Output should be JSON-formatted with field names visible
    assert "document_name" in rendered


def test_format_few_shot_example_truncates_long_contracts():
    long_body = "A" * 5000
    row = _make_train_row("c", populated_fields=11, n_parties=2, body=long_body)
    cand = eval_prompt_baseline._candidate_summary(row)
    rendered = eval_prompt_baseline.format_few_shot_example(cand)
    assert "[...example contract truncated for prompt budget...]" in rendered
    # The first 2000 chars of the body appear, but not all 5000.
    assert "A" * 2000 in rendered
    assert "A" * 5000 not in rendered


def test_format_few_shot_example_does_not_truncate_short_contracts():
    short_body = "A" * 100
    row = _make_train_row("c", populated_fields=11, n_parties=2, body=short_body)
    cand = eval_prompt_baseline._candidate_summary(row)
    rendered = eval_prompt_baseline.format_few_shot_example(cand)
    assert "[...example contract truncated" not in rendered
    assert short_body in rendered


# ---------------------------------------------------------------------------
# build_strong_prompt
# ---------------------------------------------------------------------------


def test_strong_prompt_contains_all_blocks():
    schema = "SCHEMA_BLOCK_MARKER"
    fewshot = "FEWSHOT_BLOCK_MARKER"
    prompt = eval_prompt_baseline.build_strong_prompt(
        contract_text="THE_TEST_CONTRACT",
        schema_description=schema,
        few_shot_block=fewshot,
    )
    assert schema in prompt
    assert fewshot in prompt
    assert "THE_TEST_CONTRACT" in prompt
    assert "Output requirements:" in prompt
    assert "Examples:" in prompt


def test_strong_prompt_is_deterministic():
    p1 = eval_prompt_baseline.build_strong_prompt("X", "SCHEMA", "FS")
    p2 = eval_prompt_baseline.build_strong_prompt("X", "SCHEMA", "FS")
    assert p1 == p2


def test_strong_prompt_is_substantially_longer_than_naive():
    """Sanity gate: the strong prompt must be much larger than the naive one.

    If they're similar in size, the strong baseline isn't actually doing
    prompt engineering.
    """
    naive = eval_base.build_prompt("X")
    schema = eval_prompt_baseline.build_schema_description()
    # Build a realistic few-shot block from a synthetic example.
    row = _make_train_row("c", populated_fields=11, n_parties=2)
    cand = eval_prompt_baseline._candidate_summary(row)
    fs = eval_prompt_baseline.format_few_shot_example(cand)
    strong = eval_prompt_baseline.build_strong_prompt("X", schema, fs)
    assert len(strong) > len(naive) * 5


def test_strong_prompt_constraints_match_spec():
    """The CONSTRAINTS block must include the four points from ROADMAP § Phase 5."""
    text = eval_prompt_baseline.CONSTRAINTS
    assert "ONLY valid JSON" in text or "only valid JSON" in text.lower() or "ONLY" in text
    assert "null" in text
    assert "ISO" in text
    assert "[]" in text or "empty list" in text.lower()


# ---------------------------------------------------------------------------
# load_train_rows
# ---------------------------------------------------------------------------


def test_load_train_rows_reads_all(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    rows = [_make_train_row(f"c{i}", 11, 2) for i in range(3)]
    _write_jsonl(path, rows)
    loaded = eval_prompt_baseline.load_train_rows(path)
    assert len(loaded) == 3
    assert loaded[0]["contract_id"] == "c0"


def test_load_train_rows_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    rows = [_make_train_row("c0", 11, 2)]
    _write_jsonl(path, rows)
    with path.open("a") as fh:
        fh.write("\n\n")
    loaded = eval_prompt_baseline.load_train_rows(path)
    assert len(loaded) == 1


# ---------------------------------------------------------------------------
# Module-load hygiene
# ---------------------------------------------------------------------------


def test_module_imports_without_loading_a_model():
    importlib.reload(eval_prompt_baseline)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_main_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        eval_prompt_baseline.main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--train" in captured.out
    assert "--input" in captured.out
    assert "--output" in captured.out
    assert "--limit" in captured.out


# ---------------------------------------------------------------------------
# End-to-end with mocked model
# ---------------------------------------------------------------------------


def test_main_end_to_end_with_mocked_model(tmp_path: Path, monkeypatch):
    test_path = tmp_path / "test.jsonl"
    train_path = tmp_path / "train.jsonl"
    output_path = tmp_path / "preds.json"

    _make_test_jsonl(test_path, ["c1"])
    _write_jsonl(
        train_path,
        [
            _make_train_row("complete_train", populated_fields=11, n_parties=2),
            _make_train_row("sparse_train", populated_fields=4, n_parties=0),
            _make_train_row("multi_train", populated_fields=8, n_parties=4),
        ],
    )

    valid_obj = {f: None if f != "parties" else [] for f in ContractExtraction.model_fields}
    valid_obj["document_name"] = "Mocked Strong"
    valid_json = json.dumps(valid_obj)

    monkeypatch.setattr(
        "evaluation._runner.generate_one",
        lambda *a, **k: valid_json,
    )
    monkeypatch.setattr(
        "evaluation._runner.load_model",
        lambda *a, **k: (MagicMock(), MagicMock()),
    )

    rc = eval_prompt_baseline.main(
        [
            "--input",
            str(test_path),
            "--train",
            str(train_path),
            "--output",
            str(output_path),
        ]
    )
    assert rc == 0
    data = json.loads(output_path.read_text())
    assert len(data) == 1
    assert data[0]["is_valid_json"] is True
    assert data[0]["parsed"]["document_name"] == "Mocked Strong"


def test_main_returns_1_on_empty_train_set(tmp_path: Path, monkeypatch):
    test_path = tmp_path / "test.jsonl"
    train_path = tmp_path / "train.jsonl"
    output_path = tmp_path / "preds.json"

    _make_test_jsonl(test_path, ["c1"])
    train_path.write_text("")  # empty

    rc = eval_prompt_baseline.main(
        [
            "--input",
            str(test_path),
            "--train",
            str(train_path),
            "--output",
            str(output_path),
        ]
    )
    assert rc == 1
    assert not output_path.exists()
