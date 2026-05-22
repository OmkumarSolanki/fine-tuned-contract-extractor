"""Tests for ``training/prepare_dataset.py``.

A simple whitespace-tokenizer fake is used so these tests never touch the
network or require transformers. The real Llama tokenizer is exercised at
runtime via ``load_tokenizer()`` and a sanity-preview log line, but it is
deliberately not unit-tested here (network/license-gated).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from extractor.schemas import ContractExtraction
from training.prepare_dataset import (
    MAX_TOTAL_TOKENS,
    SYSTEM_PROMPT,
    TRUNC_MARKER,
    build_messages,
    compact_json,
    split_indices,
    split_rows,
    truncate_text,
)


# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------


class WhitespaceTokenizer:
    """Whitespace-tokenized stand-in. Sufficient for testing truncate_text and
    build_messages structure. Not a faithful tokenizer — just deterministic."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Token ids are positions; we only need len() and slice semantics here.
        return list(range(len(text.split())))

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        # We use the encode/decode pair only for the head/tail join in
        # truncate_text. Reconstruct via positional indexing on the cached text.
        return "WORD " * len(ids)  # not used for real reconstruction in tests


class IdentityTokenizer:
    """Tokenizer that round-trips: encode/decode use the original text words."""

    def __init__(self) -> None:
        self._cache: dict[int, list[str]] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        words = text.split()
        # Token "ids" are pointers into a per-call words list. To keep
        # decode() lossless across head/tail slicing, we encode the position.
        token_ids = list(range(len(words)))
        self._cache[id(token_ids)] = words
        return token_ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        # Find the cached words list whose length covers all requested ids.
        # In tests we always operate on a single text per truncate_text call,
        # so we can grab the most recently cached list.
        if not self._cache:
            return ""
        words = list(self._cache.values())[-1]
        return " ".join(words[i] for i in ids if 0 <= i < len(words))


def _valid_annotations() -> dict:
    return {
        "document_name": "License Agreement",
        "parties": ["Acme Corp", "Beta Inc"],
        "agreement_date": "2018-05-15",
        "effective_date": "2018-06-01",
        "expiration_date": None,
        "governing_law": "Delaware",
        "renewal_term": None,
        "notice_period_to_terminate_renewal": None,
        "exclusivity": None,
        "non_compete": None,
        "cap_on_liability": "Limited to fees paid",
        "uncapped_liability": "IP infringement and confidentiality are uncapped",
    }


# ---------------------------------------------------------------------------
# compact_json
# ---------------------------------------------------------------------------


def test_compact_json_field_order_canonical() -> None:
    """Output keys must follow ContractExtraction.model_fields declaration order."""
    annotations = _valid_annotations()
    # Shuffle the input dict to prove we don't rely on input ordering.
    shuffled = {k: annotations[k] for k in reversed(list(annotations))}
    out = compact_json(shuffled)
    parsed = json.loads(out)
    assert list(parsed.keys()) == list(ContractExtraction.model_fields)


def test_compact_json_no_whitespace() -> None:
    """No spaces after commas or colons (compact JSON)."""
    out = compact_json(_valid_annotations())
    assert ", " not in out
    assert ": " not in out


def test_compact_json_preserves_unicode() -> None:
    """Non-ASCII characters must not be escaped."""
    annotations = _valid_annotations()
    annotations["governing_law"] = "España"
    out = compact_json(annotations)
    assert "España" in out
    assert "\\u" not in out


def test_compact_json_handles_missing_keys() -> None:
    """Missing keys default to None (or [] for parties)."""
    out = compact_json({"document_name": "X"})
    parsed = json.loads(out)
    assert parsed["document_name"] == "X"
    assert parsed["parties"] == []
    assert parsed["governing_law"] is None


def test_compact_json_handles_explicit_null_parties() -> None:
    """If parties is explicitly None, normalize to []."""
    annotations = _valid_annotations()
    annotations["parties"] = None
    out = compact_json(annotations)
    assert json.loads(out)["parties"] == []


# ---------------------------------------------------------------------------
# truncate_text
# ---------------------------------------------------------------------------


def test_truncate_text_short_unchanged() -> None:
    """Text under budget passes through verbatim."""
    text = "word " * 100  # 100 tokens
    tok = IdentityTokenizer()
    out = truncate_text(text, tok, max_total=8000, head=5000, tail=3000)
    assert out == text


def test_truncate_text_long_uses_head_tail_marker() -> None:
    """Text over budget gets head + marker + tail."""
    # 10000 distinct words so we can verify head/tail boundaries.
    words = [f"w{i}" for i in range(10000)]
    text = " ".join(words)
    tok = IdentityTokenizer()
    out = truncate_text(text, tok, max_total=8000, head=5000, tail=3000)
    assert TRUNC_MARKER in out
    head_part, tail_part = out.split(TRUNC_MARKER, 1)
    head_words = head_part.split()
    tail_words = tail_part.split()
    assert head_words[0] == "w0"
    assert head_words[-1] == "w4999"
    assert tail_words[0] == "w7000"
    assert tail_words[-1] == "w9999"


def test_truncate_text_at_budget_unchanged() -> None:
    """Exactly at the budget should NOT trigger truncation."""
    words = [f"w{i}" for i in range(8000)]
    text = " ".join(words)
    tok = IdentityTokenizer()
    out = truncate_text(text, tok, max_total=8000, head=5000, tail=3000)
    assert out == text
    assert TRUNC_MARKER not in out


def test_truncate_text_uses_module_constants_by_default() -> None:
    """The MAX_TOTAL_TOKENS default should be 8000 per project plan."""
    assert MAX_TOTAL_TOKENS == 8000


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


def test_build_messages_valid_structure() -> None:
    tok = IdentityTokenizer()
    out = build_messages(
        contract_id="CONTRACT_001",
        contract_text="A short contract text " * 5,
        annotations=_valid_annotations(),
        tokenizer=tok,
    )
    assert out["contract_id"] == "CONTRACT_001"
    assert isinstance(out["messages"], list)
    assert [m["role"] for m in out["messages"]] == ["system", "user", "assistant"]
    assert out["messages"][0]["content"] == SYSTEM_PROMPT
    assert "Extract structured clauses" in out["messages"][1]["content"]
    # Assistant content is compact JSON.
    parsed = json.loads(out["messages"][2]["content"])
    assert parsed["document_name"] == "License Agreement"
    assert parsed["parties"] == ["Acme Corp", "Beta Inc"]


def test_build_messages_drops_invalid_annotations() -> None:
    """Invalid annotations (e.g., parties=str) raise ValueError so the caller drops."""
    tok = IdentityTokenizer()
    bad = _valid_annotations()
    bad["parties"] = "not a list"
    with pytest.raises(ValueError):
        build_messages(
            contract_id="X",
            contract_text="some text " * 10,
            annotations=bad,
            tokenizer=tok,
        )


def test_build_messages_assistant_json_canonical_order() -> None:
    tok = IdentityTokenizer()
    out = build_messages(
        contract_id="X",
        contract_text="text " * 10,
        annotations=_valid_annotations(),
        tokenizer=tok,
    )
    assistant = out["messages"][2]["content"]
    parsed = json.loads(assistant)
    assert list(parsed.keys()) == list(ContractExtraction.model_fields)


# ---------------------------------------------------------------------------
# split_indices / split_rows
# ---------------------------------------------------------------------------


def test_split_indices_80_10_10() -> None:
    train, val, test = split_indices(100, seed=42)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10
    # All indices accounted for, no overlap.
    assert set(train) | set(val) | set(test) == set(range(100))
    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)


def test_split_indices_deterministic_with_seed() -> None:
    a = split_indices(100, seed=42)
    b = split_indices(100, seed=42)
    assert a == b


def test_split_indices_different_seeds_differ() -> None:
    a = split_indices(100, seed=42)
    b = split_indices(100, seed=7)
    # Highly unlikely to be equal across two seeds.
    assert a != b


def test_split_rows_preserves_total() -> None:
    rows = [{"i": i} for i in range(50)]
    train, val, test = split_rows(rows, seed=42)
    assert len(train) + len(val) + len(test) == 50


def test_split_indices_handles_uneven_counts() -> None:
    """For 510 (the realistic CUAD count): 408/51/51."""
    train, val, test = split_indices(510, seed=42)
    assert len(train) == 408
    assert len(val) == 51
    assert len(test) == 51
