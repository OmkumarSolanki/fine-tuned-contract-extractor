"""Unit tests for ``evaluation/eval_finetuned.py`` and the fine-tuned ``_runner`` helpers.

CPU-only: the model + tokenizer are mocked and ``generate_chat`` is
monkey-patched, so no GPU / unsloth / torch is needed. The real adapter run
happens on RunPod, separately. Coverage:

- ``load_test_messages`` (reads system+user turns, drops the gold assistant turn).
- ``run_finetuned`` end-to-end with a patched ``generate_chat`` (valid, invalid,
  empty-set, and limit behaviour).
- The ``eval_finetuned`` CLI surface.
- Module-load hygiene (no heavy imports at import time).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation import eval_finetuned
from evaluation._runner import (
    USER_PROMPT_PREFIX,
    load_test_messages,
    run_finetuned,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_jsonl(path: Path, contract_ids: list[str]) -> None:
    """Write a minimal training-format test.jsonl (system + user + assistant)."""
    with path.open("w", encoding="utf-8") as fh:
        for cid in contract_ids:
            row = {
                "contract_id": cid,
                "messages": [
                    {"role": "system", "content": "You are a legal contract analyst."},
                    {
                        "role": "user",
                        "content": USER_PROMPT_PREFIX + f"Contract body for {cid}.",
                    },
                    {"role": "assistant", "content": "{}"},
                ],
            }
            fh.write(json.dumps(row) + "\n")


def _full_valid_extraction_json(document_name: str = "X") -> str:
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
# load_test_messages
# ---------------------------------------------------------------------------


def test_load_test_messages_reads_prompt_turns(tmp_path: Path):
    p = tmp_path / "test.jsonl"
    _make_test_jsonl(p, ["c1", "c2"])
    examples = load_test_messages(p)
    assert len(examples) == 2
    assert examples[0]["contract_id"] == "c1"
    roles = [m["role"] for m in examples[0]["prompt_messages"]]
    assert roles == ["system", "user"]


def test_load_test_messages_excludes_assistant_gold(tmp_path: Path):
    """The gold assistant turn must NOT be fed back to the model as a prompt."""
    p = tmp_path / "test.jsonl"
    _make_test_jsonl(p, ["c1"])
    examples = load_test_messages(p)
    roles = [m["role"] for m in examples[0]["prompt_messages"]]
    assert "assistant" not in roles


def test_load_test_messages_preserves_training_prompt_verbatim(tmp_path: Path):
    """The user turn must be byte-identical to what training produced."""
    p = tmp_path / "test.jsonl"
    _make_test_jsonl(p, ["c1"])
    examples = load_test_messages(p)
    user_turn = examples[0]["prompt_messages"][1]
    assert user_turn["content"] == USER_PROMPT_PREFIX + "Contract body for c1."


def test_load_test_messages_respects_limit(tmp_path: Path):
    p = tmp_path / "test.jsonl"
    _make_test_jsonl(p, ["c1", "c2", "c3"])
    examples = load_test_messages(p, limit=2)
    assert [e["contract_id"] for e in examples] == ["c1", "c2"]


def test_load_test_messages_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "test.jsonl"
    _make_test_jsonl(p, ["c1"])
    with p.open("a") as fh:
        fh.write("\n\n")
    assert len(load_test_messages(p)) == 1


# ---------------------------------------------------------------------------
# run_finetuned (mocked generate_chat)
# ---------------------------------------------------------------------------


def test_run_finetuned_end_to_end_valid(tmp_path: Path, monkeypatch):
    test_path = tmp_path / "test.jsonl"
    _make_test_jsonl(test_path, ["c1", "c2"])
    output_path = tmp_path / "finetuned_predictions.json"

    valid_json = _full_valid_extraction_json("FT Agreement")
    monkeypatch.setattr(
        "evaluation._runner.generate_chat",
        lambda tokenizer, model, messages, max_new_tokens=2048: valid_json,
    )

    rc = run_finetuned(
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
        assert rec["parsed"]["document_name"] == "FT Agreement"


def test_run_finetuned_persists_invalid(tmp_path: Path, monkeypatch):
    test_path = tmp_path / "test.jsonl"
    _make_test_jsonl(test_path, ["c1"])
    output_path = tmp_path / "preds.json"

    monkeypatch.setattr(
        "evaluation._runner.generate_chat",
        lambda *a, **k: "Here is the extraction: {not json}",
    )

    rc = run_finetuned(
        test_path=test_path,
        output_path=output_path,
        tokenizer=MagicMock(),
        model=MagicMock(),
    )
    assert rc == 0
    data = json.loads(output_path.read_text())
    assert data[0]["is_valid_json"] is False
    assert data[0]["parsed"] is None


def test_run_finetuned_passes_prompt_messages_to_generate(tmp_path: Path, monkeypatch):
    """The fine-tuned path must hand generate_chat the system+user messages, not a string."""
    test_path = tmp_path / "test.jsonl"
    _make_test_jsonl(test_path, ["c1"])
    output_path = tmp_path / "preds.json"

    captured = {}

    def fake_generate(tokenizer, model, messages, max_new_tokens=2048):
        captured["messages"] = messages
        return _full_valid_extraction_json()

    monkeypatch.setattr("evaluation._runner.generate_chat", fake_generate)

    run_finetuned(
        test_path=test_path,
        output_path=output_path,
        tokenizer=MagicMock(),
        model=MagicMock(),
    )
    assert [m["role"] for m in captured["messages"]] == ["system", "user"]


def test_run_finetuned_empty_test_set_returns_1(tmp_path: Path):
    test_path = tmp_path / "test.jsonl"
    test_path.write_text("")
    output_path = tmp_path / "preds.json"
    rc = run_finetuned(
        test_path=test_path,
        output_path=output_path,
        tokenizer=MagicMock(),
        model=MagicMock(),
    )
    assert rc == 1
    assert not output_path.exists()


def test_run_finetuned_respects_limit(tmp_path: Path, monkeypatch):
    test_path = tmp_path / "test.jsonl"
    _make_test_jsonl(test_path, ["c1", "c2", "c3", "c4"])
    output_path = tmp_path / "preds.json"
    monkeypatch.setattr(
        "evaluation._runner.generate_chat",
        lambda *a, **k: _full_valid_extraction_json(),
    )
    rc = run_finetuned(
        test_path=test_path,
        output_path=output_path,
        limit=2,
        tokenizer=MagicMock(),
        model=MagicMock(),
    )
    assert rc == 0
    data = json.loads(output_path.read_text())
    assert [r["contract_id"] for r in data] == ["c1", "c2"]


# ---------------------------------------------------------------------------
# Module-load hygiene + CLI
# ---------------------------------------------------------------------------


def test_module_imports_without_loading_a_model():
    importlib.reload(eval_finetuned)


def test_main_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        eval_finetuned.main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "--adapter" in captured.out
    assert "--limit" in captured.out
    assert "--output" in captured.out
