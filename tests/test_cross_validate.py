"""CPU-only tests for training/cross_validate.py (fold partitioning + aggregation)."""

from __future__ import annotations

import json

import pytest

from training.cross_validate import (
    aggregate_from_summaries,
    aggregate_metrics,
    fold_splits,
    kfold_partition,
    load_rows_by_id,
    write_fold_datasets,
)


# --------------------------------------------------------------------------- partition


def test_kfold_partition_covers_all_ids_once():
    ids = [f"c{i}" for i in range(23)]
    folds = kfold_partition(ids, k=5, seed=42)
    assert len(folds) == 5
    flat = [x for f in folds for x in f]
    assert sorted(flat) == sorted(ids)  # every id exactly once
    assert len(flat) == len(set(flat))


def test_kfold_partition_balanced():
    folds = kfold_partition([f"c{i}" for i in range(50)], k=5, seed=1)
    sizes = sorted(len(f) for f in folds)
    assert sizes[-1] - sizes[0] <= 1  # differ by at most one


def test_kfold_partition_deterministic():
    ids = [f"c{i}" for i in range(30)]
    assert kfold_partition(ids, k=5, seed=42) == kfold_partition(ids, k=5, seed=42)


def test_kfold_partition_seed_changes_assignment():
    ids = [f"c{i}" for i in range(30)]
    assert kfold_partition(ids, k=5, seed=1) != kfold_partition(ids, k=5, seed=2)


def test_kfold_partition_rejects_small_k_or_n():
    with pytest.raises(ValueError):
        kfold_partition(["a", "b"], k=1)
    with pytest.raises(ValueError):
        kfold_partition(["a", "b"], k=5)


# --------------------------------------------------------------------------- fold_splits


def test_fold_splits_disjoint_and_complete():
    folds = kfold_partition([f"c{i}" for i in range(25)], k=5, seed=42)
    for i in range(5):
        train, val, test = fold_splits(folds, i)
        allids = set(train) | set(val) | set(test)
        # disjoint
        assert len(train) + len(val) + len(test) == len(allids)
        # test is exactly fold i; val is the next fold
        assert set(test) == set(folds[i])
        assert set(val) == set(folds[(i + 1) % 5])
        # covers everything
        assert allids == {x for f in folds for x in f}


def test_fold_splits_out_of_range():
    folds = kfold_partition([f"c{i}" for i in range(10)], k=5, seed=0)
    with pytest.raises(ValueError):
        fold_splits(folds, 5)


# --------------------------------------------------------------------------- write datasets


def test_write_fold_datasets_writes_all_splits(tmp_path):
    rows_by_id = {f"c{i}": {"contract_id": f"c{i}", "messages": [{"role": "user", "content": str(i)}]} for i in range(20)}
    folds = kfold_partition(list(rows_by_id), k=5, seed=42)
    manifest = write_fold_datasets(rows_by_id, folds, tmp_path)
    assert len(manifest) == 5
    for m in manifest:
        for name in ("train", "val", "test"):
            p = tmp_path / f"fold_{m['fold']}" / f"{name}.jsonl"
            assert p.exists()
            lines = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            assert len(lines) == m[f"n_{name}"]
        # train+val+test for a fold sums to all 20 contracts
        assert m["n_train"] + m["n_val"] + m["n_test"] == 20


# --------------------------------------------------------------------------- aggregate


def test_aggregate_metrics_mean_std_and_pooled():
    fold_results = [
        {"json_validity_rate": 0.90, "json_valid": 90, "n_contracts": 100, "overall_f1": 0.70},
        {"json_validity_rate": 1.00, "json_valid": 100, "n_contracts": 100, "overall_f1": 0.80},
    ]
    agg = aggregate_metrics(fold_results)
    assert agg["n_folds"] == 2
    assert agg["json_validity_rate"]["mean"] == pytest.approx(0.95)
    assert agg["json_validity_rate"]["std"] == pytest.approx(0.0707, abs=1e-3)
    assert agg["overall_f1"]["mean"] == pytest.approx(0.75)
    assert agg["pooled_validity"] == pytest.approx(190 / 200)
    assert agg["total_contracts_scored"] == 200


def test_aggregate_metrics_single_fold_zero_std():
    agg = aggregate_metrics([{"json_validity_rate": 0.9, "json_valid": 9, "n_contracts": 10, "overall_f1": 0.7}])
    assert agg["json_validity_rate"]["std"] == 0.0


def test_aggregate_from_summaries_reads_model_block(tmp_path):
    for i, (vr, f1, nv) in enumerate([(0.92, 0.71, 94), (0.96, 0.74, 98)]):
        d = tmp_path / f"fold_{i}"
        d.mkdir()
        summary = {
            "models": {
                "finetuned": {
                    "json_validity_rate": vr,
                    "json_valid": nv,
                    "n_contracts": 102,
                    "overall_f1": f1,
                }
            }
        }
        (d / "analysis_summary.json").write_text(json.dumps(summary))
    paths = sorted(tmp_path.glob("fold_*/analysis_summary.json"))
    agg = aggregate_from_summaries(paths, model_key="finetuned")
    assert agg["n_folds"] == 2
    assert agg["json_validity_rate"]["mean"] == pytest.approx(0.94)
    assert agg["overall_f1"]["mean"] == pytest.approx(0.725)


def test_aggregate_from_summaries_skips_missing_model(tmp_path):
    d = tmp_path / "fold_0"
    d.mkdir()
    (d / "analysis_summary.json").write_text(json.dumps({"models": {}}))
    agg = aggregate_from_summaries([d / "analysis_summary.json"], model_key="finetuned")
    assert agg["n_folds"] == 0


# --------------------------------------------------------------------------- loader


def test_load_rows_by_id_dedupes(tmp_path):
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text(json.dumps({"contract_id": "x", "v": 1}) + "\n" + json.dumps({"contract_id": "y", "v": 2}) + "\n")
    p2.write_text(json.dumps({"contract_id": "z", "v": 3}) + "\n")
    rows = load_rows_by_id([p1, p2])
    assert set(rows) == {"x", "y", "z"}


def test_main_help_exits_zero():
    from training.cross_validate import main

    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
