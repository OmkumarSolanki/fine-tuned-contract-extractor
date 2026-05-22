"""Download CUAD-QA from Hugging Face, parse into one record per contract, and
write ``data/raw/cuad_parsed.jsonl``.

Mapping rules (per project plan):

- ``parties`` (list field): collect all non-empty answer spans across all
  chunks for the "Parties" category, deduplicate case-insensitively while
  preserving the original casing of the first occurrence.
- Date singular fields (``agreement_date``, ``effective_date``,
  ``expiration_date``): take the LONGEST non-empty answer span, attempt
  ``dateutil.parser.parse(..., fuzzy=True)`` to ISO ``YYYY-MM-DD``, fall back
  to the raw text on parse failure.
- All other singular string fields: take the LONGEST non-empty answer span,
  or ``None`` if all spans are empty.

CLI:

    python training/ingest_cuad.py [--limit N] [--force] [--output PATH]

Idempotency: by default, exits early if the output file already exists. Pass
``--force`` to overwrite.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from dateutil.parser import parse as dateutil_parse

from extractor.schemas import ContractExtraction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Map our 12 schema field names → CUAD category strings (as they appear in the
# CUAD-QA dataset's `id` field, e.g. ``...__Document Name_0``).
TARGET_CATEGORIES: dict[str, str] = {
    "document_name": "Document Name",
    "parties": "Parties",
    "agreement_date": "Agreement Date",
    "effective_date": "Effective Date",
    "expiration_date": "Expiration Date",
    "governing_law": "Governing Law",
    "renewal_term": "Renewal Term",
    "notice_period_to_terminate_renewal": "Notice Period To Terminate Renewal",
    "exclusivity": "Exclusivity",
    "non_compete": "Non-Compete",
    "cap_on_liability": "Cap On Liability",
    "uncapped_liability": "Uncapped Liability",
}
# Reverse lookup: CUAD category string → schema field name.
CUAD_TO_FIELD: dict[str, str] = {v: k for k, v in TARGET_CATEGORIES.items()}

DATE_FIELDS: set[str] = {"agreement_date", "effective_date", "expiration_date"}
LIST_FIELDS: set[str] = {"parties"}

# Regex for extracting (category, chunk_index) from a CUAD-QA `id`.
# Two observed id forms:
#   ``LIMEENERGYCO_..._DISTRIBUTOR AGREEMENT__Document Name_0``  (chunked)
#   ``ACCELERATED..._JOINT VENTURE AGREEMENT__Document Name``    (single-chunk)
# The category may contain spaces, dashes, and other characters but no digits
# in the CUAD taxonomy. The chunk index, when present, is a trailing
# ``_<digits>`` anchored at end of string. When absent we treat the chunk as 0.
_ID_REGEX = re.compile(r"__(?P<cat>.+?)(?:_(?P<chunk>\d+))?$")

# Sentinel year used as the dateutil ``default``. If parsing produces this
# year, dateutil filled it in itself (i.e., the input had no year), and we
# treat the parse as a failure.
_DATE_SENTINEL_YEAR = 1900
_DATE_DEFAULT = datetime(_DATE_SENTINEL_YEAR, 1, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_category_from_id(id_str: str) -> str:
    """Pull the CUAD category name out of a CUAD-QA ``id`` string.

    Raises ``ValueError`` if the id does not match the expected
    ``__<cat>_<chunk>`` suffix.
    """
    match = _ID_REGEX.search(id_str)
    if match is None:
        raise ValueError(f"Unparseable id: {id_str!r}")
    return match.group("cat")


def extract_chunk_index_from_id(id_str: str) -> int:
    """Pull the trailing chunk index out of a CUAD-QA ``id`` string.

    Returns ``0`` when the id has no explicit ``_<digits>`` suffix (this
    happens for short contracts that fit in a single chunk and therefore
    don't get a trailing chunk index).
    """
    match = _ID_REGEX.search(id_str)
    if match is None:
        raise ValueError(f"Unparseable id: {id_str!r}")
    chunk = match.group("chunk")
    return int(chunk) if chunk is not None else 0


def parse_date_loose(value: Optional[str]) -> Optional[str]:
    """Best-effort date normalization to ISO ``YYYY-MM-DD``.

    Returns ``None`` for empty/whitespace input. On parse failure, returns the
    stripped raw string so downstream callers still see the original
    annotation.
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = dateutil_parse(stripped, fuzzy=True, default=_DATE_DEFAULT)
    except Exception:
        return stripped
    if parsed.year == _DATE_SENTINEL_YEAR:
        # dateutil only used our default — the input had no real year.
        return stripped
    return parsed.date().isoformat()


def dedupe_preserve_case(spans: list[str]) -> list[str]:
    """Deduplicate strings case-insensitively while preserving first-seen casing.

    Empty/whitespace-only entries are dropped.
    """
    seen: set[str] = set()
    out: list[str] = []
    for span in spans:
        if span is None:
            continue
        stripped = span.strip()
        if not stripped:
            continue
        key = stripped.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(stripped)
    return out


def pick_longest_span(spans: list[str]) -> Optional[str]:
    """Return the longest non-empty span, or ``None`` if none."""
    candidates = [s.strip() for s in spans if s and s.strip()]
    if not candidates:
        return None
    return max(candidates, key=len)


def aggregate_contract(category_to_spans: dict[str, list[str]]) -> dict:
    """Apply the 12-field mapping rules to raw category→spans data."""
    result: dict = {}
    for field, cuad_cat in TARGET_CATEGORIES.items():
        spans = category_to_spans.get(cuad_cat, [])
        if field in LIST_FIELDS:
            result[field] = dedupe_preserve_case(spans)
        elif field in DATE_FIELDS:
            longest = pick_longest_span(spans)
            result[field] = parse_date_loose(longest)
        else:
            result[field] = pick_longest_span(spans)
    return result


def assemble_contract_text(rows_for_title: list[dict]) -> str:
    """Recover full contract text by deduplicating contexts per chunk index.

    CUAD-QA chunks long contracts; for each chunk we have one row per question
    (so the same context repeats ~41 times). We keep one context per chunk
    index, ordered ascending, joined by blank lines.
    """
    chunk_to_context: dict[int, str] = {}
    for row in rows_for_title:
        try:
            chunk_idx = extract_chunk_index_from_id(row["id"])
        except ValueError:
            continue
        if chunk_idx not in chunk_to_context:
            chunk_to_context[chunk_idx] = row.get("context", "") or ""
    return "\n\n".join(chunk_to_context[k] for k in sorted(chunk_to_context))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_category_to_spans(rows_for_title: list[dict]) -> dict[str, list[str]]:
    """Group answer spans by CUAD category for one contract."""
    out: dict[str, list[str]] = defaultdict(list)
    for row in rows_for_title:
        try:
            category = extract_category_from_id(row["id"])
        except ValueError:
            continue
        if category not in CUAD_TO_FIELD:
            # Not one of our 12 target categories; skip.
            continue
        answer_texts = (row.get("answers") or {}).get("text") or []
        for txt in answer_texts:
            if txt is not None:
                out[category].append(txt)
    return dict(out)


def _short_validation_reason(exc: Exception) -> str:
    """Render a Pydantic ValidationError as a single short line."""
    msg = str(exc).replace("\n", " | ")
    return msg[:200] + ("…" if len(msg) > 200 else "")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output",
        default="data/raw/cuad_parsed.jsonl",
        help="Path to write parsed contracts (JSONL). Default: data/raw/cuad_parsed.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only process the first N contracts (after deterministic title sort).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists() and not args.force:
        logger.info(
            "Output exists at %s. Skipping. Use --force to overwrite.", output
        )
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)

    # Deferred import: heavy and only needed at runtime, not at import time
    # (so unit tests of the helpers above don't pull in `datasets`).
    from datasets import concatenate_datasets, load_dataset
    from tqdm import tqdm

    logger.info("Loading theatticusproject/cuad-qa from Hugging Face …")
    ds = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)
    logger.info(
        "Loaded splits: train=%d, test=%d", len(ds["train"]), len(ds["test"])
    )
    pooled = concatenate_datasets([ds["train"], ds["test"]])
    logger.info("Pooled rows: %d", len(pooled))

    # Group rows by contract title.
    by_title: dict[str, list[dict]] = defaultdict(list)
    for row in pooled:
        by_title[row["title"]].append(row)
    titles = sorted(by_title.keys())
    logger.info("Unique contract titles: %d", len(titles))

    if args.limit is not None:
        titles = titles[: args.limit]
        logger.info("Limiting to first %d titles", len(titles))

    n_written = 0
    n_skipped = 0
    with output.open("w", encoding="utf-8") as fh:
        for title in tqdm(titles, desc="Parsing contracts"):
            rows = by_title[title]
            category_to_spans = _build_category_to_spans(rows)
            annotations = aggregate_contract(category_to_spans)
            try:
                ContractExtraction.model_validate(annotations)
            except Exception as exc:  # noqa: BLE001 — we log a short reason
                logger.warning(
                    "Skipped %s: validation failed (%s)",
                    title,
                    _short_validation_reason(exc),
                )
                n_skipped += 1
                continue

            contract_text = assemble_contract_text(rows)
            if not contract_text.strip():
                logger.warning("Skipped %s: empty contract text", title)
                n_skipped += 1
                continue

            record = {
                "contract_id": title,
                "contract_text": contract_text,
                "annotations": annotations,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    logger.info(
        "Wrote %d contracts to %s (skipped %d)", n_written, output, n_skipped
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    sys.exit(main())
