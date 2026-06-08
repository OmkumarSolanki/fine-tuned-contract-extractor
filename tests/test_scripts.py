"""Unit tests for the Phase 11 operational scripts' pure helpers.

CPU-only, no network, no GPU, no model load: we only exercise the module-level
pure functions (percentile math, report aggregation, adapter-path precedence,
JSON formatting). The heavy/HTTP paths in ``main`` are intentionally not
imported here.
"""

from __future__ import annotations

import json

import pytest

from scripts.benchmark_latency import percentile, summarize
from scripts.run_local_inference import format_extraction, resolve_adapter_path

# ---------------------------------------------------------------------------
# percentile
# ---------------------------------------------------------------------------


def test_percentile_single_element():
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 99) == 42.0


def test_percentile_median_even_and_odd():
    assert percentile([1, 2, 3], 50) == 2.0
    assert percentile([1, 2, 3, 4], 50) == 2.5


def test_percentile_min_max_bounds():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 0) == 10.0
    assert percentile(values, 100) == 50.0


def test_percentile_interpolates():
    # p90 of 0..100 (11 points) lands at rank 9.0 → exactly 90.
    assert percentile(list(range(0, 101, 10)), 90) == 90.0


def test_percentile_unsorted_input():
    assert percentile([30, 10, 20], 50) == 20.0


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


@pytest.mark.parametrize("p", [-1, 101, 150])
def test_percentile_out_of_range_raises(p):
    with pytest.raises(ValueError):
        percentile([1, 2, 3], p)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def test_summarize_latency_only():
    report = summarize([100.0, 200.0, 300.0])
    assert report["n"] == 3
    assert report["latency_ms"]["p50"] == 200.0
    assert report["latency_ms"]["mean"] == 200.0
    assert "ttft_ms" not in report
    assert "tokens_per_second" not in report


def test_summarize_includes_ttft_when_given():
    report = summarize([100.0, 200.0], ttfts_ms=[10.0, 30.0])
    assert report["ttft_ms"]["p50"] == 20.0
    assert report["ttft_ms"]["mean"] == 20.0


def test_summarize_tokens_per_second():
    # 100 tokens in 1000 ms → 100 tok/s; 200 tokens in 1000 ms → 200 tok/s.
    report = summarize([1000.0, 1000.0], token_counts=[100, 200])
    tps = report["tokens_per_second"]
    assert tps["mean"] == pytest.approx(150.0)
    assert tps["max"] == pytest.approx(200.0)
    assert tps["min"] == pytest.approx(100.0)


def test_summarize_skips_zero_latency_for_tps():
    report = summarize([0.0, 1000.0], token_counts=[100, 100])
    # the zero-latency sample is skipped to avoid div-by-zero
    assert report["tokens_per_second"]["mean"] == pytest.approx(100.0)


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([])


# ---------------------------------------------------------------------------
# resolve_adapter_path
# ---------------------------------------------------------------------------


def test_resolve_adapter_path_cli_wins(monkeypatch):
    monkeypatch.setenv("EXTRACTOR_ADAPTER_PATH", "from-env")
    assert resolve_adapter_path("from-cli") == "from-cli"


def test_resolve_adapter_path_env_fallback(monkeypatch):
    monkeypatch.setenv("EXTRACTOR_ADAPTER_PATH", "from-env")
    assert resolve_adapter_path(None) == "from-env"


def test_resolve_adapter_path_default(monkeypatch):
    monkeypatch.delenv("EXTRACTOR_ADAPTER_PATH", raising=False)
    from extractor.inference.model_loader import DEFAULT_ADAPTER_PATH

    assert resolve_adapter_path(None) == DEFAULT_ADAPTER_PATH


# ---------------------------------------------------------------------------
# format_extraction
# ---------------------------------------------------------------------------


def test_format_extraction_pretty_prints_json():
    out = format_extraction('{"document_name":"X","parties":["A"]}')
    assert "\n" in out  # indented
    assert json.loads(out) == {"document_name": "X", "parties": ["A"]}


def test_format_extraction_passes_through_invalid_json():
    raw = "Here is the extraction: not json"
    assert format_extraction(raw) == raw
