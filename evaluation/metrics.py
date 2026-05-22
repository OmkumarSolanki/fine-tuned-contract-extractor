"""Evaluation metrics for the contract extractor.

This module is intentionally minimal:

1. The functions are pure-Python and have no upstream dependencies.
2. They lock down what "correct output" means before we generate the training
   data, so any drift between what we train the model to produce and what we
   later score is caught at the schema layer rather than at eval time.

Public functions:

- :func:`is_valid_json`  — does a model prediction parse as JSON and validate
  against :class:`ContractExtraction`?
- :func:`field_accuracy` — binary case-insensitive equality on a single field,
  with both-None treated as a correct match.
- :func:`parties_f1`     — F1 over the ``parties`` list field (case-insensitive
  set comparison).
- :func:`overall_f1`     — aggregate per-field scores across a held-out test
  set and add an ``overall_f1`` mean.
"""

from __future__ import annotations

import json
from typing import Optional  # noqa: F401  (kept for parity with spec signature comments)

from extractor.schemas import ContractExtraction


def is_valid_json(prediction_str: str) -> bool:
    """Did the model return parseable JSON matching our schema?"""
    try:
        data = json.loads(prediction_str)
        ContractExtraction.model_validate(data)
        return True
    except Exception:
        return False


def field_accuracy(pred: ContractExtraction, gold: ContractExtraction, field: str) -> float:
    """Binary accuracy: did the model get this field right?"""
    pred_val = getattr(pred, field)
    gold_val = getattr(gold, field)

    # Both empty/null is a correct match.
    if pred_val is None and gold_val is None:
        return 1.0
    if pred_val is None or gold_val is None:
        return 0.0

    # String comparison: case-insensitive, whitespace-trimmed equality.
    return float(str(pred_val).strip().lower() == str(gold_val).strip().lower())


def parties_f1(pred: ContractExtraction, gold: ContractExtraction) -> float:
    """F1 for parties (list field) — handles set comparison."""
    pred_set = {p.strip().lower() for p in pred.parties}
    gold_set = {p.strip().lower() for p in gold.parties}

    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0

    tp = len(pred_set & gold_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def overall_f1(
    predictions: list[ContractExtraction],
    golds: list[ContractExtraction],
) -> dict:
    """Compute aggregate metrics across all test examples.

    Returns a dict mapping each field name to its mean per-example score, plus
    an ``overall_f1`` key holding the mean across all per-field scores.
    """
    field_scores: dict[str, float] = {}

    for field in ContractExtraction.model_fields:
        if field == "parties":
            scores = [parties_f1(p, g) for p, g in zip(predictions, golds)]
        else:
            scores = [field_accuracy(p, g, field) for p, g in zip(predictions, golds)]
        field_scores[field] = sum(scores) / len(scores) if scores else 0.0

    field_scores["overall_f1"] = (
        sum(field_scores.values()) / len(field_scores) if field_scores else 0.0
    )
    return field_scores
