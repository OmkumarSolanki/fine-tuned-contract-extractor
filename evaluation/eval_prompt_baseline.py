"""Strong-prompt baseline for the three-way comparison.

Same base model and generation parameters as ``eval_base.py``, but with a
carefully-engineered prompt:

1. **Full schema description**, generated at runtime from
   :class:`~extractor.schemas.ContractExtraction.model_fields` so it can never
   drift from the live schema. Each field's description is pulled directly
   from its Pydantic ``Field(description=...)`` declaration.

2. **Three worked few-shot examples** drawn from the *train* set
   (``data/processed/train.jsonl``):

   - one "complete" example (≥11 of 12 fields populated),
   - one "sparse" example (≤6 of 12 fields populated, demonstrating ``null``),
   - one "multi-party" example (parties list with ≥3 entries).

   Examples are picked deterministically by scanning the train set in file
   order and taking the first row matching each criterion. None come from the
   test set — that would be data leakage and would invalidate the comparison.

3. **Explicit output constraints** — JSON only, ``null`` for missing fields,
   ISO dates, empty list for parties.

This measures how far prompt engineering alone can take the unmodified base
model. It's the realistic baseline a working engineer would ship if
fine-tuning were not an option.

CLI::

    python evaluation/eval_prompt_baseline.py [--limit N] [--input PATH] [--output PATH] [--train PATH]

Outputs ``data/results/prompt_baseline_predictions.json`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from extractor.schemas import ContractExtraction
from evaluation._runner import extract_contract_text, run_baseline

logger = logging.getLogger(__name__)


# Soft cap on how much of each few-shot contract body to include in the prompt.
# Few-shot examples don't need full context; the goal is to demonstrate the
# input/output shape, not to retrain the model in-context.
FEW_SHOT_CONTRACT_CHAR_BUDGET = 2000


CONSTRAINTS = """\
Output requirements:
- Output ONLY valid JSON. No prose, no markdown fences, no commentary before or after.
- Use null (not the string "null") for any field the contract does not address.
- Use ISO date format (YYYY-MM-DD) for date fields where the contract makes the date unambiguous.
- For parties, return a list of strings. Use [] if no parties are clearly identified.
- All 12 fields must appear, in the order shown in the schema."""


# ---------------------------------------------------------------------------
# Schema description (drift-proof: pulled from the live schema)
# ---------------------------------------------------------------------------


def _type_label(field_name: str) -> str:
    """Display label for the field type in the prompt."""
    if field_name == "parties":
        return "list[str]"
    return "str | null"


def build_schema_description() -> str:
    """Render the schema as a prompt-friendly bulleted block.

    Pulls field names and descriptions directly from
    :attr:`ContractExtraction.model_fields`. Any new field added to the schema
    automatically appears here on the next run, with no separate dictionary
    to maintain. A field whose description is empty is logged as a warning;
    such a field would still be listed but with an empty hint, so this
    function never blocks a run on it.
    """
    lines = ["Schema (12 fields, in this exact order):"]
    for field_name, field_info in ContractExtraction.model_fields.items():
        desc = (field_info.description or "").strip()
        if not desc:
            logger.warning(
                "Schema field %s has no description; prompt will list it without a hint.",
                field_name,
            )
        lines.append(f"  - {field_name} ({_type_label(field_name)}): {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Few-shot example selection (deterministic, train-only)
# ---------------------------------------------------------------------------


def _count_populated(extraction: dict) -> int:
    """Count fields with non-null, non-empty values."""
    n = 0
    for k, v in extraction.items():
        if k == "parties":
            n += int(bool(v))
        else:
            n += int(v is not None)
    return n


def _candidate_summary(row: dict) -> Optional[dict]:
    """Augment a raw train row with selection-relevant stats.

    Returns ``None`` if the row's assistant message can't be parsed (defensive;
    shouldn't happen for files produced by ``training/prepare_dataset.py``).
    """
    try:
        assistant_content = row["messages"][2]["content"]
        extraction = json.loads(assistant_content)
    except Exception:  # noqa: BLE001
        return None
    return {
        "contract_id": row["contract_id"],
        "user_msg": row["messages"][1]["content"],
        "extraction": extraction,
        "n_pop": _count_populated(extraction),
        "n_parties": len(extraction.get("parties") or []),
    }


def select_few_shot_examples(train_rows: list[dict]) -> list[dict]:
    """Pick three deterministic, contrasting few-shot examples from train.

    Selection criteria, applied in file order over ``train_rows``:

    1. ``complete`` — first row whose extraction populates ≥11 of 12 fields.
    2. ``sparse``   — first row whose extraction populates ≤6 of 12 fields.
    3. ``multi``    — first row with ≥3 parties, not already picked above.

    Returns the three picks in canonical order ``[complete, sparse, multi]``.

    Raises
    ------
    RuntimeError
        If any of the three categories has no candidate. That would indicate
        the train set has changed shape and the baseline needs re-tuning.
    """
    parsed: list[dict] = []
    for row in train_rows:
        cand = _candidate_summary(row)
        if cand is not None:
            parsed.append(cand)

    complete = next((p for p in parsed if p["n_pop"] >= 11), None)
    sparse = next((p for p in parsed if p["n_pop"] <= 6), None)
    if not complete or not sparse:
        raise RuntimeError(
            "Could not find suitable 'complete' (≥11 populated) AND 'sparse' "
            "(≤6 populated) few-shot examples in the train set. The data may "
            "have changed shape; review the selection criteria."
        )

    used_ids = {complete["contract_id"], sparse["contract_id"]}
    multi = next(
        (p for p in parsed if p["n_parties"] >= 3 and p["contract_id"] not in used_ids),
        None,
    )
    if not multi:
        raise RuntimeError(
            "Could not find a multi-party (≥3 parties) few-shot example "
            "outside the already-picked complete/sparse set."
        )

    picks = [complete, sparse, multi]
    logger.info(
        "Few-shot picks (from train, deterministic): %s",
        ", ".join(p["contract_id"][:60] for p in picks),
    )
    return picks


# ---------------------------------------------------------------------------
# Strong-prompt assembly
# ---------------------------------------------------------------------------


def format_few_shot_example(picked: dict) -> str:
    """Render one few-shot example as ``Contract:\\n... \\n\\nOutput:\\n...``."""
    contract_text = extract_contract_text(picked["user_msg"])
    snippet = contract_text[:FEW_SHOT_CONTRACT_CHAR_BUDGET]
    if len(contract_text) > FEW_SHOT_CONTRACT_CHAR_BUDGET:
        snippet += "\n[...example contract truncated for prompt budget...]"
    target = json.dumps(picked["extraction"], ensure_ascii=False, indent=2)
    return f"Contract:\n{snippet}\n\nOutput:\n{target}"


def build_strong_prompt(
    contract_text: str,
    schema_description: str,
    few_shot_block: str,
) -> str:
    """Assemble the final strong-prompt string for one test contract."""
    return (
        f"{schema_description}\n\n"
        f"{CONSTRAINTS}\n\n"
        f"Examples:\n\n{few_shot_block}\n\n"
        f"Now extract from this contract:\n\n"
        f"Contract:\n{contract_text}\n\n"
        f"Output:\n"
    )


# ---------------------------------------------------------------------------
# Train-set loader
# ---------------------------------------------------------------------------


def load_train_rows(train_path: Path | str) -> list[dict]:
    """Read ``train.jsonl`` into raw row dicts (preserves file order)."""
    train_path = Path(train_path)
    rows: list[dict] = []
    with train_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Strong-prompt baseline evaluator.")
    parser.add_argument(
        "--input",
        default="data/processed/test.jsonl",
        help="Path to test set JSONL. Default: data/processed/test.jsonl",
    )
    parser.add_argument(
        "--train",
        default="data/processed/train.jsonl",
        help=(
            "Path to train set JSONL (source of few-shot examples). "
            "Default: data/processed/train.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default="data/results/prompt_baseline_predictions.json",
        help=(
            "Where to save predictions. "
            "Default: data/results/prompt_baseline_predictions.json"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N examples. Default: process the full test set.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Generation budget per example. Default: 2048.",
    )
    args = parser.parse_args(argv)

    schema_description = build_schema_description()

    train_rows = load_train_rows(args.train)
    if not train_rows:
        logger.error("No train rows found at %s — required for few-shot.", args.train)
        return 1

    picks = select_few_shot_examples(train_rows)
    few_shot_block = "\n\n".join(format_few_shot_example(p) for p in picks)

    def builder(contract_text: str) -> str:
        return build_strong_prompt(contract_text, schema_description, few_shot_block)

    return run_baseline(
        prompt_builder=builder,
        test_path=args.input,
        output_path=args.output,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    sys.exit(main())
