"""Benchmark the running serving API: latency, TTFT, and throughput.

Sends N requests to a running ``extractor.api`` instance and reports latency
and throughput percentiles. Time-to-first-token (TTFT) is measured against the
SSE ``/extract/stream`` endpoint (the blocking ``/extract`` path can't observe
it); total latency and tokens/sec are measured against ``/extract``.

The statistics helpers (:func:`percentile`, :func:`summarize`) are pure and
unit-tested on CPU; ``httpx`` is imported inside :func:`main` so this module
imports without it.

Usage::

    # start the API first, e.g. uvicorn extractor.api:app --port 8000
    python scripts/benchmark_latency.py --n 20 --base-url http://localhost:8000
    python scripts/benchmark_latency.py --api-key "$EXTRACTOR_API_KEY"
"""

from __future__ import annotations

import argparse
import json
import time

SAMPLE_CONTRACT = (
    "SUPPLY AGREEMENT. This Supply Agreement is entered into as of March 3, 2021 "
    "by and between Globex Corporation (\"Supplier\") and Initech Inc. "
    "(\"Buyer\"). This Agreement is governed by the laws of the State of "
    "Delaware and remains in effect for three (3) years."
)


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile of ``values`` for ``p`` in [0, 100].

    Pure helper. Raises ``ValueError`` on an empty list or an out-of-range
    ``p``. Uses the same interpolation convention as ``numpy.percentile`` so a
    single-element list returns that element for any ``p``.
    """
    if not values:
        raise ValueError("percentile() requires a non-empty list")
    if not 0.0 <= p <= 100.0:
        raise ValueError("p must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (p / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def summarize(
    latencies_ms: list[float],
    ttfts_ms: list[float] | None = None,
    token_counts: list[int] | None = None,
) -> dict:
    """Aggregate raw per-request measurements into a report dict.

    Pure helper: takes already-collected measurements and returns p50/p90/p99
    and means for latency, TTFT (if provided), and derived tokens-per-second.
    """
    if not latencies_ms:
        raise ValueError("summarize() requires at least one latency sample")

    report: dict = {
        "n": len(latencies_ms),
        "latency_ms": {
            "p50": percentile(latencies_ms, 50),
            "p90": percentile(latencies_ms, 90),
            "p99": percentile(latencies_ms, 99),
            "mean": sum(latencies_ms) / len(latencies_ms),
        },
    }

    if ttfts_ms:
        report["ttft_ms"] = {
            "p50": percentile(ttfts_ms, 50),
            "p90": percentile(ttfts_ms, 90),
            "p99": percentile(ttfts_ms, 99),
            "mean": sum(ttfts_ms) / len(ttfts_ms),
        }

    if token_counts:
        # tokens/sec per request = tokens / (latency_s); report the mean.
        per_request = [
            (toks / (lat_ms / 1000.0))
            for toks, lat_ms in zip(token_counts, latencies_ms)
            if lat_ms > 0
        ]
        if per_request:
            report["tokens_per_second"] = {
                "mean": sum(per_request) / len(per_request),
                "max": max(per_request),
                "min": min(per_request),
            }

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the serving API.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL.")
    parser.add_argument("--n", type=int, default=10, help="Number of requests (default: 10).")
    parser.add_argument("--api-key", default=None, help="X-API-Key value if the API requires auth.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout (s).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import httpx  # noqa: PLC0415

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    payload = {"contract_text": SAMPLE_CONTRACT}

    latencies_ms: list[float] = []
    ttfts_ms: list[float] = []
    token_counts: list[int] = []

    with httpx.Client(base_url=args.base_url, timeout=args.timeout, headers=headers) as client:
        for i in range(args.n):
            # Total latency + token count via the blocking endpoint.
            start = time.perf_counter()
            resp = client.post("/extract", json=payload)
            resp.raise_for_status()
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            token_counts.append(int(resp.json().get("tokens_generated", 0)))

            # TTFT via the streaming endpoint.
            start = time.perf_counter()
            with client.stream("POST", "/extract/stream", json=payload) as stream:
                for line in stream.iter_lines():
                    if line and line.startswith("data: "):
                        ttfts_ms.append((time.perf_counter() - start) * 1000.0)
                        break
            print(f"  request {i + 1}/{args.n} done", flush=True)

    report = summarize(latencies_ms, ttfts_ms, token_counts)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
