"""Deeper, honest analysis of the evaluation results — no GPU, no network.

This module turns the raw prediction files (already on disk) into the numbers a
careful reviewer actually asks for, beyond the headline validity rate:

- :func:`wilson_interval`   — a confidence interval for a proportion (validity
  rate), so "96% on 51 contracts" is reported as a *range*, not a point.
- :func:`bootstrap_ci`      — a bootstrap confidence interval for a mean (used
  for ``overall_f1``).
- :func:`mcnemar`           — a paired significance test on the *difference*
  between two models' per-contract correctness.
- :func:`always_null_floor` — the score of a trivial model that answers ``null``
  for every field of every contract. This is the floor our real model must
  clear; without it the per-field/``overall_f1`` numbers are unreadable (many
  CUAD fields are genuinely null, so "always null" scores deceptively high).
- :func:`null_confusion`    — per-field present/absent confusion counts, which
  expose **hallucination** (model invents a value where the gold is null) and
  **misses** (model says null where the gold has a value). The hallucination
  number is the most safety-relevant metric for a *legal* extractor.
- :func:`check_leakage`     — confirms no test/val contract also appears in
  train (a clean split is what makes the score trustworthy).

All functions are pure and dependency-free (stdlib + our own ``metrics`` /
``schemas``). The CLI loads predictions via the same loaders as ``compare.py``
and writes a text-free ``analysis_summary.json`` that is safe to commit.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Optional

from extractor.schemas import ContractExtraction
from evaluation.metrics import field_accuracy, overall_f1, parties_f1

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistics primitives (pure, no third-party deps)
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the naive normal interval for small ``n`` and proportions
    near 0/1 (exactly our case: 49/51). Returns ``(low, high)`` in ``[0, 1]``.
    ``z=1.96`` is the 95% level. ``n == 0`` yields ``(0.0, 0.0)``.
    """
    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(
    values: list[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the *mean* of ``values``.

    Resamples ``values`` with replacement ``n_boot`` times and returns the
    ``(alpha/2, 1-alpha/2)`` percentiles of the resampled means. Deterministic
    for a fixed ``seed``. Empty input yields ``(0.0, 0.0)``.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return (means[lo_idx], means[hi_idx])


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """Paired McNemar test on two models' per-item correctness.

    ``a_correct[i]`` / ``b_correct[i]`` are whether model A / model B got item
    ``i`` right. Tests whether the *difference* in their correctness is real or
    chance. Returns the discordant counts, the (continuity-corrected) chi-square
    statistic, and a two-sided p-value from the chi-square(df=1) survival
    function (``erfc`` — no scipy needed).

    ``b`` = A right, B wrong; ``c`` = A wrong, B right. ``b + c == 0`` (models
    never disagree) yields ``p_value = 1.0``.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("a_correct and b_correct must have the same length.")
    b = sum(1 for a, bb in zip(a_correct, b_correct) if a and not bb)
    c = sum(1 for a, bb in zip(a_correct, b_correct) if not a and bb)
    n_discordant = b + c
    if n_discordant == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1.0) ** 2 / n_discordant  # continuity-corrected
    # Chi-square df=1 survival: P(X > stat) = erfc(sqrt(stat / 2)).
    p_value = math.erfc(math.sqrt(stat / 2.0))
    return {"b": b, "c": c, "statistic": stat, "p_value": p_value}


# ---------------------------------------------------------------------------
# Per-example scoring (enables a bootstrap CI on overall_f1)
# ---------------------------------------------------------------------------


def per_example_overall(
    predictions: list[ContractExtraction],
    golds: list[ContractExtraction],
) -> list[float]:
    """Per-contract overall score = mean across the 12 fields for that contract.

    The mean of this list equals ``metrics.overall_f1(...)['overall_f1']``
    (the field count is constant, so averaging fields-then-examples equals
    examples-then-fields). Returning the per-example list is what lets us
    bootstrap a CI around ``overall_f1``.
    """
    fields = list(ContractExtraction.model_fields)
    out: list[float] = []
    for pred, gold in zip(predictions, golds):
        scores = [
            parties_f1(pred, gold) if f == "parties" else field_accuracy(pred, gold, f)
            for f in fields
        ]
        out.append(sum(scores) / len(scores) if scores else 0.0)
    return out


# ---------------------------------------------------------------------------
# Always-null floor
# ---------------------------------------------------------------------------


def always_null_floor(golds: list[ContractExtraction]) -> dict:
    """Score a trivial model that returns an empty extraction for every contract.

    Returns the per-field + ``overall_f1`` block (via ``metrics.overall_f1``).
    Any field this floor scores highly is one where "always null" is usually
    correct — so the real model's lift on that field is what matters, not its
    absolute score.
    """
    empties = [ContractExtraction() for _ in golds]
    return overall_f1(empties, golds)


# ---------------------------------------------------------------------------
# Hallucination / null-confusion (the safety-relevant view)
# ---------------------------------------------------------------------------


def _is_present(value) -> bool:
    """A field is 'present' if the model committed to a value (non-null / non-empty)."""
    if value is None:
        return False
    if isinstance(value, (list, str)):
        return len(value) > 0
    return True


def null_confusion(
    predictions: list[ContractExtraction],
    golds: list[ContractExtraction],
) -> dict:
    """Per-field present/absent confusion, exposing hallucination and misses.

    For each field, treat "present" (model gave a value) as positive:

    - ``tp`` gold present & pred present     (model spoke up correctly)
    - ``fp`` gold **null** & pred present    → **hallucination** (invented a value)
    - ``fn`` gold present & pred **null**    → **miss** (stayed silent wrongly)
    - ``tn`` gold null & pred null           (correctly silent)

    Plus derived ``precision`` (when the model speaks, how often the gold agrees
    it should), ``recall``, and ``hallucination_rate`` = ``fp / (fp + tn)`` (of
    the truly-empty fields, how often the model invented something). Note this
    measures *commitment*, not value-correctness — a present/present pair counts
    as ``tp`` even if the extracted text differs (value accuracy is
    ``field_accuracy`` / the LLM-judge metric, separately).

    Returns ``{field: {...counts + rates...}, "_aggregate": {...}}``.
    """
    fields = list(ContractExtraction.model_fields)
    per_field: dict[str, dict] = {}
    agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for f in fields:
        tp = fp = fn = tn = 0
        for pred, gold in zip(predictions, golds):
            p_present = _is_present(getattr(pred, f))
            g_present = _is_present(getattr(gold, f))
            if g_present and p_present:
                tp += 1
            elif not g_present and p_present:
                fp += 1
            elif g_present and not p_present:
                fn += 1
            else:
                tn += 1
        per_field[f] = _confusion_rates(tp, fp, fn, tn)
        agg["tp"] += tp
        agg["fp"] += fp
        agg["fn"] += fn
        agg["tn"] += tn

    per_field["_aggregate"] = _confusion_rates(agg["tp"], agg["fp"], agg["fn"], agg["tn"])
    return per_field


def _confusion_rates(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    hallucination_rate = fp / (fp + tn) if (fp + tn) else None
    return {
        "tp": tp,
        "fp_hallucination": fp,
        "fn_miss": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "hallucination_rate": hallucination_rate,
    }


# ---------------------------------------------------------------------------
# Failure-mode breakdown — *why* an invalid output was invalid
# ---------------------------------------------------------------------------


def classify_failure(raw_output: str) -> str:
    """Best-effort single reason a raw model output failed schema-valid JSON.

    Checked in priority order so each invalid output gets one primary bucket:

    - ``empty``             nothing (or only whitespace) produced.
    - ``markdown_fence``    wrapped in a ```` ``` ```` code fence (parse fails as-is).
    - ``no_json_object``    prose with no ``{`` at all.
    - ``truncated``         an opening ``{`` but no closing ``}`` (ran out of budget).
    - ``malformed_json``    has both braces but doesn't parse.
    - ``prose_around_json`` valid JSON object, but with extra text before/after it.
    - ``schema_mismatch``   parses as JSON but fails the 12-field schema.
    - ``valid_but_flagged`` parses + validates cleanly (classifier/loader disagree).
    """
    s = (raw_output or "").strip()
    if not s:
        return "empty"
    if "```" in s:
        return "markdown_fence"
    first, last = s.find("{"), s.rfind("}")
    if first == -1:
        return "no_json_object"
    if last == -1:
        return "truncated"
    candidate = s[first : last + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return "malformed_json"
    if s[:first].strip() or s[last + 1 :].strip():
        return "prose_around_json"
    try:
        ContractExtraction.model_validate(data)
    except Exception:  # noqa: BLE001
        return "schema_mismatch"
    return "valid_but_flagged"


def failure_mode_breakdown(records: list[dict]) -> dict:
    """Tally :func:`classify_failure` over the *invalid* records of one model.

    ``records`` are raw prediction dicts (``raw_output`` + ``is_valid_json``).
    Returns the invalid count and a reason→count map (most common first).
    """
    counts: dict[str, int] = {}
    n_invalid = 0
    for rec in records:
        if rec.get("is_valid_json"):
            continue
        n_invalid += 1
        reason = classify_failure(rec.get("raw_output", ""))
        counts[reason] = counts.get(reason, 0) + 1
    return {
        "n_invalid": n_invalid,
        "by_reason": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }


# ---------------------------------------------------------------------------
# Truncation ceiling — can the gold span even survive head+tail truncation?
# ---------------------------------------------------------------------------

# The marker `prepare_dataset.truncate_text` inserts between the kept head and tail.
DEFAULT_TRUNC_MARKER = "\n[...TRUNCATED...]\n"

# Long, verbatim-span fields (dates/parties are normalized, so a substring check
# isn't meaningful for them; these clause fields are copied spans).
DEFAULT_LONG_FIELDS = (
    "expiration_date",
    "governing_law",
    "renewal_term",
    "notice_period_to_terminate_renewal",
    "exclusivity",
    "non_compete",
    "cap_on_liability",
    "uncapped_liability",
)


def truncation_survival(
    test_rows: list[dict],
    marker: str = DEFAULT_TRUNC_MARKER,
    long_fields: tuple[str, ...] = DEFAULT_LONG_FIELDS,
    min_len: int = 25,
) -> dict:
    """Measure how often a gold long-text span is *missing* from the truncated input.

    ``test_rows`` are raw test JSONL rows (``messages`` = system/user/assistant).
    A contract is "truncated" if the user message contains ``marker``. For each
    truncated contract, each gold long-text field (≥ ``min_len`` chars) is checked
    for a **verbatim** presence in the (truncated) user input; absence means the
    model could not have extracted it — a hard ceiling, not a model error.

    ``lost`` is an approximate *upper bound* (a verbatim miss can also be a
    whitespace/normalization difference even when the span is present).
    """
    n_total = len(test_rows)
    n_trunc = 0
    present = lost = 0
    per_field: dict[str, dict] = {}
    for row in test_rows:
        user = row["messages"][1]["content"]
        gold = json.loads(row["messages"][-1]["content"])
        if marker not in user:
            continue
        n_trunc += 1
        for f in long_fields:
            v = gold.get(f)
            if isinstance(v, str) and len(v) >= min_len:
                present += 1
                pf = per_field.setdefault(f, {"present": 0, "lost": 0})
                pf["present"] += 1
                if v not in user:
                    lost += 1
                    pf["lost"] += 1
    return {
        "n_contracts": n_total,
        "n_truncated": n_trunc,
        "truncated_rate": (n_trunc / n_total) if n_total else 0.0,
        "long_spans_in_truncated": present,
        "long_spans_lost": lost,
        "lost_rate": (lost / present) if present else None,
        "by_field": {
            f: {**c, "lost_rate": (c["lost"] / c["present"]) if c["present"] else None}
            for f, c in per_field.items()
        },
        "_note": (
            "lost = gold long-text span not found verbatim in the truncated input; an "
            "approximate upper bound on truncation-driven loss for these fields."
        ),
    }


# ---------------------------------------------------------------------------
# Leakage / split-integrity
# ---------------------------------------------------------------------------


def _contract_ids(jsonl_path: Path | str) -> set[str]:
    ids: set[str] = set()
    with Path(jsonl_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["contract_id"])
    return ids


def check_leakage(
    train_path: Path | str,
    val_path: Path | str,
    test_path: Path | str,
) -> dict:
    """Confirm the three splits share no ``contract_id`` (no train→test leakage)."""
    tr, va, te = (_contract_ids(train_path), _contract_ids(val_path), _contract_ids(test_path))
    train_test = sorted(tr & te)
    train_val = sorted(tr & va)
    val_test = sorted(va & te)
    clean = not (train_test or train_val or val_test)
    return {
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
        "overlap_train_test": train_test,
        "overlap_train_val": train_val,
        "overlap_val_test": val_test,
        "clean": clean,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def analyze_model(
    predictions: list[ContractExtraction],
    golds: list[ContractExtraction],
    valid_flags: list[bool],
    seed: int = 42,
) -> dict:
    """Full analysis block for one model's aligned (prediction, gold) lists.

    ``valid_flags[i]`` is whether prediction ``i`` was schema-valid JSON.
    Returns validity rate + Wilson CI, ``overall_f1`` + bootstrap CI, the
    per-field block, and the null/hallucination confusion.
    """
    n = len(golds)
    n_valid = sum(valid_flags)
    per_ex = per_example_overall(predictions, golds)
    of1_block = overall_f1(predictions, golds)
    return {
        "n_contracts": n,
        "json_valid": n_valid,
        "json_validity_rate": (n_valid / n) if n else 0.0,
        "json_validity_ci95": wilson_interval(n_valid, n),
        "overall_f1": of1_block["overall_f1"],
        "overall_f1_ci95": bootstrap_ci(per_ex, seed=seed),
        "per_field": of1_block,
        "null_confusion": null_confusion(predictions, golds),
        "_per_example_overall": per_ex,
    }


def _round(obj, ndigits: int = 4):
    """Recursively round floats for a tidy committed summary."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, ndigits) for v in obj]
    return obj


def main(argv: Optional[list[str]] = None) -> int:
    # Imported here so the module stays import-light for unit tests.
    from evaluation.compare import load_golds, load_predictions, record_to_extraction

    parser = argparse.ArgumentParser(description="Deep analysis of evaluation results.")
    parser.add_argument("--base", default="data/results/base_predictions.json")
    parser.add_argument("--prompt", default="data/results/prompt_baseline_predictions.json")
    parser.add_argument(
        "--constrained", default="data/results/constrained_baseline_predictions.json"
    )
    parser.add_argument("--finetuned", default="data/results/finetuned_predictions.json")
    parser.add_argument("--train", default="data/processed/train.jsonl")
    parser.add_argument("--val", default="data/processed/val.jsonl")
    parser.add_argument("--test", default="data/processed/test.jsonl")
    parser.add_argument("--output", default="data/results/analysis_summary.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    golds_pairs = load_golds(args.test)
    if not golds_pairs:
        logger.error("No gold examples in %s", args.test)
        return 1
    golds = [g for _, g in golds_pairs]

    # Raw rows (with the truncated user message) for the truncation-ceiling check.
    with Path(args.test).open("r", encoding="utf-8") as fh:
        test_rows = [json.loads(line) for line in fh if line.strip()]

    model_paths = {
        "naive": args.base,
        "strong_prompt": args.prompt,
        "constrained_prompt": args.constrained,
        "finetuned": args.finetuned,
    }

    models: dict[str, dict] = {}
    correctness: dict[str, list[bool]] = {}  # per-contract validity, for McNemar
    for key, path in model_paths.items():
        preds_by_id = load_predictions(path)
        if not preds_by_id:
            continue
        pred_exts, valid_flags = [], []
        for contract_id, _ in golds_pairs:
            ext, is_valid = record_to_extraction(preds_by_id.get(contract_id))
            pred_exts.append(ext)
            valid_flags.append(is_valid)
        models[key] = analyze_model(pred_exts, golds, valid_flags, seed=args.seed)
        models[key]["failure_modes"] = failure_mode_breakdown(list(preds_by_id.values()))
        correctness[key] = valid_flags

    # Always-null floor + significance of fine-tuned vs each baseline (on validity).
    floor = always_null_floor(golds)
    significance = {}
    if "finetuned" in correctness:
        for base_key in ("naive", "strong_prompt", "constrained_prompt"):
            if base_key in correctness:
                significance[f"finetuned_vs_{base_key}_validity"] = mcnemar(
                    correctness["finetuned"], correctness[base_key]
                )

    summary = {
        "_headline": (
            "Deep analysis: validity with Wilson CI, overall_f1 with bootstrap CI, per-field "
            "match, null/hallucination confusion, the always-null floor, and McNemar significance "
            "of the fine-tuned vs. baseline difference. Aggregate numbers only — no CUAD text."
        ),
        "n_contracts": len(golds),
        "leakage": check_leakage(args.train, args.val, args.test),
        "truncation_ceiling": truncation_survival(test_rows),
        "always_null_floor": floor,
        "models": models,
        "significance": significance,
    }
    # Drop the bulky per-example arrays from the committed file.
    for m in summary["models"].values():
        m.pop("_per_example_overall", None)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(_round(summary), fh, ensure_ascii=False, indent=2)
    logger.info("Wrote analysis summary to %s", out)
    print(json.dumps(_round(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(main())
