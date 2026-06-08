"""5-fold cross-validation harness — kills the n=51 weakness.

The headline metrics (96% valid, 0.73 ``overall_f1``) come from a single
408/51/51 split — only **51** test contracts, so the confidence intervals are
wide (validity 96% spans 87–99%). Cross-validation instead tests on **all 510**
contracts: partition them into ``k`` folds, and for each fold train an adapter on
the other folds and evaluate on the held-out one. Every contract is then scored
exactly once by a model that never trained on it, and we report **mean ± std**
across folds — a far more credible number than one lucky split.

This module is split into two halves:

- **Pure, CPU-tested helpers** — deterministic fold partitioning
  (:func:`kfold_partition`, :func:`fold_splits`), dataset materialization
  (:func:`write_fold_datasets`), and result aggregation
  (:func:`aggregate_metrics`, :func:`aggregate_from_summaries`).
- **GPU orchestration** (``main``) — writes the ``k`` fold datasets and emits the
  per-fold train + eval + analyze command plan; with ``--aggregate`` it combines
  the per-fold ``analysis_summary.json`` files into a single
  ``cross_val_summary.json`` (mean ± std). The heavy per-fold training runs on
  the GPU host; this driver prepares the data and crunches the results.

CLI::

    # Prepare the k fold datasets (deterministic) + print the per-fold plan:
    python training/cross_validate.py --k 5 --out-root data/cv

    # After running each fold's train+eval+analysis on the GPU, aggregate:
    python training/cross_validate.py --aggregate data/cv --output data/results/cross_val_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic fold partitioning (pure)
# ---------------------------------------------------------------------------


def kfold_partition(ids: list[str], k: int = 5, seed: int = 42) -> list[list[str]]:
    """Partition ``ids`` into ``k`` balanced, deterministic folds.

    Shuffles with ``seed`` then stride-assigns (``ids[i::k]``) so fold sizes
    differ by at most one. Every id lands in exactly one fold. Deterministic for
    a fixed ``(ids, k, seed)``.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if len(ids) < k:
        raise ValueError(f"need at least k={k} ids, got {len(ids)}")
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i::k] for i in range(k)]


def fold_splits(
    folds: list[list[str]],
    fold_idx: int,
) -> tuple[list[str], list[str], list[str]]:
    """For one fold, return ``(train_ids, val_ids, test_ids)``.

    ``test`` = fold ``fold_idx``; ``val`` = the next fold (wrapping); ``train`` =
    all remaining folds. For ``k=5`` over 510 contracts this is ≈ 306/102/102 —
    the same train/val/test *roles* as the original split, but with a ~2× larger
    test set per fold (tighter per-fold CIs), and 5 folds covering all 510.
    """
    k = len(folds)
    if not 0 <= fold_idx < k:
        raise ValueError(f"fold_idx {fold_idx} out of range for k={k}")
    test_ids = list(folds[fold_idx])
    val_ids = list(folds[(fold_idx + 1) % k])
    train_ids: list[str] = []
    for j in range(k):
        if j != fold_idx and j != (fold_idx + 1) % k:
            train_ids.extend(folds[j])
    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Dataset materialization (pure-ish: filesystem only)
# ---------------------------------------------------------------------------


def write_fold_datasets(
    rows_by_id: dict[str, dict],
    folds: list[list[str]],
    out_root: Path | str,
) -> list[dict]:
    """Write ``out_root/fold_<i>/{train,val,test}.jsonl`` for every fold.

    ``rows_by_id`` maps ``contract_id`` → the full ChatML row (as produced by
    ``prepare_dataset.py``). Returns a manifest: one dict per fold with the split
    sizes and paths, so the orchestrator can build the per-fold commands.
    """
    out_root = Path(out_root)
    manifest: list[dict] = []
    for i in range(len(folds)):
        train_ids, val_ids, test_ids = fold_splits(folds, i)
        fold_dir = out_root / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
            p = fold_dir / f"{name}.jsonl"
            with p.open("w", encoding="utf-8") as fh:
                for cid in ids:
                    fh.write(json.dumps(rows_by_id[cid], ensure_ascii=False) + "\n")
            paths[name] = str(p)
        manifest.append(
            {
                "fold": i,
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "n_test": len(test_ids),
                "paths": paths,
            }
        )
    return manifest


# ---------------------------------------------------------------------------
# Aggregation (pure)
# ---------------------------------------------------------------------------


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def aggregate_metrics(fold_results: list[dict]) -> dict:
    """Combine per-fold ``{json_validity_rate, overall_f1}`` into mean ± std.

    Also returns ``pooled_validity`` = total valid / total contracts across all
    folds (a single number over the whole dataset, complementing the fold mean).
    """
    validity = [r["json_validity_rate"] for r in fold_results]
    f1 = [r["overall_f1"] for r in fold_results]
    total_valid = sum(r.get("json_valid", 0) for r in fold_results)
    total_n = sum(r.get("n_contracts", 0) for r in fold_results)
    return {
        "n_folds": len(fold_results),
        "json_validity_rate": _mean_std(validity),
        "overall_f1": _mean_std(f1),
        "pooled_validity": (total_valid / total_n) if total_n else None,
        "total_contracts_scored": total_n,
        "per_fold": fold_results,
    }


def aggregate_from_summaries(
    summary_paths: list[Path | str],
    model_key: str = "finetuned",
) -> dict:
    """Aggregate per-fold ``analysis_summary.json`` files for one model.

    Reads each fold's summary (written by ``evaluation/analysis.py``), pulls the
    ``model_key`` block's validity + ``overall_f1``, and runs
    :func:`aggregate_metrics`. Folds missing the model are skipped with a warning.
    """
    fold_results: list[dict] = []
    for path in summary_paths:
        with Path(path).open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
        model = summary.get("models", {}).get(model_key)
        if model is None:
            logger.warning("Summary %s has no model %r; skipping.", path, model_key)
            continue
        fold_results.append(
            {
                "json_validity_rate": model["json_validity_rate"],
                "json_valid": model.get("json_valid", 0),
                "n_contracts": model.get("n_contracts", 0),
                "overall_f1": model["overall_f1"],
            }
        )
    return aggregate_metrics(fold_results)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_rows_by_id(data_paths: list[Path | str]) -> dict[str, dict]:
    """Concatenate ChatML jsonl files into ``{contract_id: row}`` (dedup by id)."""
    rows_by_id: dict[str, dict] = {}
    for path in data_paths:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows_by_id[row["contract_id"]] = row
    return rows_by_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print_plan(manifest: list[dict], adapter_root: str) -> None:
    """Emit the per-fold train + eval + analyze command plan (GPU host)."""
    print("\n# Per-fold plan (run on the GPU host):")
    for m in manifest:
        i = m["fold"]
        ad = f"{adapter_root}/fold_{i}"
        print(f"\n## fold {i}  (train {m['n_train']} / val {m['n_val']} / test {m['n_test']})")
        print(f"#   train  → {ad}")
        print(f"#   eval   → data/cv/fold_{i}/finetuned_predictions.json")
        print(f"#   analyze→ data/cv/fold_{i}/analysis_summary.json")
    print("\n# Then aggregate:")
    print(
        "#   python training/cross_validate.py --aggregate data/cv "
        "--output data/results/cross_val_summary.json"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="5-fold cross-validation harness.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data",
        nargs="+",
        default=[
            "data/processed/train.jsonl",
            "data/processed/val.jsonl",
            "data/processed/test.jsonl",
        ],
        help="ChatML jsonl files to pool into the full dataset (default: all 3 splits).",
    )
    parser.add_argument("--out-root", default="data/cv", help="Where to write fold datasets.")
    parser.add_argument(
        "--adapter-root",
        default="checkpoints/cv",
        help="Where per-fold adapters will be saved (used only to print the plan).",
    )
    parser.add_argument(
        "--aggregate",
        metavar="CV_DIR",
        default=None,
        help="Aggregate per-fold analysis_summary.json under CV_DIR/fold_*/ and exit.",
    )
    parser.add_argument("--model-key", default="finetuned")
    parser.add_argument("--output", default="data/results/cross_val_summary.json")
    args = parser.parse_args(argv)

    # --- aggregate mode -----------------------------------------------------
    if args.aggregate:
        cv_dir = Path(args.aggregate)
        summaries = sorted(cv_dir.glob("fold_*/analysis_summary.json"))
        if not summaries:
            logger.error("No fold_*/analysis_summary.json found under %s", cv_dir)
            return 1
        agg = aggregate_from_summaries(summaries, model_key=args.model_key)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(agg, fh, ensure_ascii=False, indent=2)
        v, f = agg["json_validity_rate"], agg["overall_f1"]
        print(
            f"{agg['n_folds']}-fold ({agg['total_contracts_scored']} contracts): "
            f"validity {v['mean']:.3f} ± {v['std']:.3f}  |  "
            f"overall_f1 {f['mean']:.4f} ± {f['std']:.4f}"
        )
        logger.info("Wrote %s", out)
        return 0

    # --- prepare mode -------------------------------------------------------
    rows_by_id = load_rows_by_id(args.data)
    if not rows_by_id:
        logger.error("No rows loaded from %s", args.data)
        return 1
    folds = kfold_partition(sorted(rows_by_id), k=args.k, seed=args.seed)
    manifest = write_fold_datasets(rows_by_id, folds, args.out_root)
    logger.info(
        "Wrote %d folds (%d contracts) to %s", len(folds), len(rows_by_id), args.out_root
    )
    _print_plan(manifest, args.adapter_root)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(main())
