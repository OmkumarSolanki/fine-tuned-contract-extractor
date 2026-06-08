"""Three-way comparison: naive baseline vs. strong-prompt baseline vs. fine-tuned.

Consumes the three prediction files written by the evaluators and the gold
answers in ``test.jsonl``, and reports — per model — two metrics:

1. **JSON-validity rate** — the fraction of outputs that parse as JSON *and*
   validate against the 12-field :class:`ContractExtraction` schema. This is
   the headline, defensible metric (``is_valid_json`` in ``metrics.py``).

2. **Per-field match + overall_f1** — computed with ``metrics.overall_f1``.
   A schema-invalid prediction is scored as an **empty extraction** (all
   ``null`` / ``parties=[]``), exactly as documented in
   ``data/results/baseline_summary.json``. NOTE: because many CUAD fields are
   null in the gold, an empty prediction scores "correct" on those sparse
   fields (both-null match), which inflates per-field/overall numbers for the
   weak baselines. These per-field scores are only meaningful *across* models
   that mostly emit valid JSON — read them alongside the validity rate, never
   alone.

Inputs are aligned by ``contract_id`` against the gold set, so a missing or
extra prediction can't silently shift the score.

The raw prediction files embed CUAD-derived text and are gitignored; the
output of this script (``data/results/comparison_summary.json``) is aggregate
numbers only and is safe to commit.

CLI::

    python evaluation/compare.py \
        [--base data/results/base_predictions.json] \
        [--prompt data/results/prompt_baseline_predictions.json] \
        [--finetuned data/results/finetuned_predictions.json] \
        [--test data/processed/test.jsonl] \
        [--output data/results/comparison_summary.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from extractor.schemas import ContractExtraction
from evaluation.metrics import overall_f1

logger = logging.getLogger(__name__)


# Canonical display order / labels for the three models in the comparison.
MODEL_LABELS = {
    "naive": "Naive baseline",
    "strong_prompt": "Strong-prompt baseline",
    "constrained_prompt": "Constrained-decoding baseline",
    "finetuned": "Fine-tuned (QLoRA)",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_predictions(path: Path | str) -> dict[str, dict]:
    """Load a predictions JSON array into ``{contract_id: record}``.

    Returns an empty dict if the file is missing (a model not yet run), so the
    comparison can still report on whichever models are available.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Predictions file not found: %s (skipping that model)", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    return {rec["contract_id"]: rec for rec in records}


def load_golds(test_path: Path | str) -> list[tuple[str, ContractExtraction]]:
    """Load gold extractions from ``test.jsonl`` (the assistant turn).

    Returns ``[(contract_id, ContractExtraction), ...]`` in file order. Raises
    if a gold row fails to parse — the gold set is authored by our own
    pipeline, so a parse failure there is a real error, not something to skip.
    """
    test_path = Path(test_path)
    golds: list[tuple[str, ContractExtraction]] = []
    with test_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assistant = row["messages"][-1]
            if assistant.get("role") != "assistant":
                raise ValueError(
                    f"Last message for {row.get('contract_id')!r} is not the assistant turn."
                )
            gold = ContractExtraction.model_validate(json.loads(assistant["content"]))
            golds.append((row["contract_id"], gold))
    return golds


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def record_to_extraction(record: Optional[dict]) -> tuple[ContractExtraction, bool]:
    """Map a prediction record to ``(ContractExtraction, is_valid)``.

    A valid record yields the parsed extraction; a missing/invalid record
    yields an **empty** extraction (all ``null`` / ``parties=[]``) and
    ``is_valid=False``. Never raises.
    """
    if record is not None and record.get("is_valid_json") and record.get("parsed") is not None:
        try:
            return ContractExtraction.model_validate(record["parsed"]), True
        except Exception:  # noqa: BLE001
            pass
    return ContractExtraction(), False


def score_model(
    predictions_by_id: dict[str, dict],
    golds: list[tuple[str, ContractExtraction]],
) -> dict:
    """Score one model's predictions against the gold set.

    Aligns by ``contract_id`` over the gold set (the source of truth for which
    contracts must be scored). Returns JSON-validity counts plus the per-field
    + ``overall_f1`` block from :func:`metrics.overall_f1`.
    """
    pred_exts: list[ContractExtraction] = []
    gold_exts: list[ContractExtraction] = []
    n_valid = 0
    for contract_id, gold in golds:
        ext, is_valid = record_to_extraction(predictions_by_id.get(contract_id))
        pred_exts.append(ext)
        gold_exts.append(gold)
        n_valid += int(is_valid)

    n = len(golds)
    return {
        "n_contracts": n,
        "json_valid": n_valid,
        "json_validity_rate": (n_valid / n) if n else 0.0,
        "per_field_match_rate_CAVEATED": overall_f1(pred_exts, gold_exts),
    }


def build_comparison(
    prediction_paths: dict[str, str],
    test_path: Path | str,
) -> dict:
    """Score every available model and assemble the comparison structure.

    ``prediction_paths`` maps a model key (``naive`` / ``strong_prompt`` /
    ``finetuned``) to its predictions JSON path. Models whose file is missing
    are skipped (with a warning) rather than failing the whole comparison.
    """
    golds = load_golds(test_path)
    if not golds:
        raise ValueError(f"No gold examples found in {test_path}.")

    results: dict[str, dict] = {}
    for key, path in prediction_paths.items():
        preds = load_predictions(path)
        if not preds:
            continue
        results[key] = score_model(preds, golds)

    return {
        "_headline": (
            "Three-way comparison on the held-out test set. The reportable metric is "
            "json_validity_rate; per-field scores are CAVEATED (schema-invalid predictions "
            "scored as empty extractions inflate sparse-null fields) and are only meaningful "
            "across models that mostly emit valid JSON. Aggregate numbers only - no "
            "CUAD-derived contract text is included."
        ),
        "n_contracts": len(golds),
        "models": results,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_table(comparison: dict) -> str:
    """Render the comparison as a plain-text table (validity rate + overall_f1)."""
    rows = []
    header = f"{'Model':<26} {'JSON-valid':>12} {'overall_f1':>12}"
    rows.append(header)
    rows.append("-" * len(header))
    for key, label in MODEL_LABELS.items():
        if key not in comparison["models"]:
            continue
        m = comparison["models"][key]
        validity = f"{m['json_valid']}/{m['n_contracts']} ({100 * m['json_validity_rate']:.0f}%)"
        of1 = m["per_field_match_rate_CAVEATED"]["overall_f1"]
        rows.append(f"{label:<26} {validity:>12} {of1:>12.4f}")
    return "\n".join(rows)


def write_summary(comparison: dict, path: Path | str) -> None:
    """Write the text-free comparison summary as indented JSON (dir auto-created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Three-way model comparison.")
    parser.add_argument(
        "--base",
        default="data/results/base_predictions.json",
        help="Naive baseline predictions. Default: data/results/base_predictions.json",
    )
    parser.add_argument(
        "--prompt",
        default="data/results/prompt_baseline_predictions.json",
        help="Strong-prompt predictions. Default: data/results/prompt_baseline_predictions.json",
    )
    parser.add_argument(
        "--constrained",
        default="data/results/constrained_baseline_predictions.json",
        help=(
            "Constrained-decoding baseline predictions (skipped if absent). "
            "Default: data/results/constrained_baseline_predictions.json"
        ),
    )
    parser.add_argument(
        "--finetuned",
        default="data/results/finetuned_predictions.json",
        help="Fine-tuned predictions. Default: data/results/finetuned_predictions.json",
    )
    parser.add_argument(
        "--test",
        default="data/processed/test.jsonl",
        help="Gold test set JSONL. Default: data/processed/test.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/results/comparison_summary.json",
        help="Where to write the text-free summary. Default: data/results/comparison_summary.json",
    )
    args = parser.parse_args(argv)

    prediction_paths = {
        "naive": args.base,
        "strong_prompt": args.prompt,
        "constrained_prompt": args.constrained,
        "finetuned": args.finetuned,
    }

    comparison = build_comparison(prediction_paths, args.test)
    if not comparison["models"]:
        logger.error(
            "No prediction files found. Run the evaluators first "
            "(eval_base.py / eval_prompt_baseline.py / eval_finetuned.py)."
        )
        return 1

    write_summary(comparison, args.output)
    print(format_table(comparison))
    logger.info("Wrote comparison summary to %s", args.output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    sys.exit(main())
