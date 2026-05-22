"""Naive-prompt baseline for the three-way comparison.

Loads ``meta-llama/Llama-3.1-8B-Instruct`` (gated, with HF_TOKEN) or the
``unsloth/Meta-Llama-3.1-8B-Instruct`` mirror (fallback) in bf16 and prompts
it with the simplest possible instruction::

    Extract the legal clauses from this contract as JSON.

    Contract:
    <contract>

    JSON:

This measures how well the *unmodified* base model performs at structured
extraction with no schema description, no examples, and no constraints. It
sets the floor for the three-way comparison reported in Phase 7.

The companion strong-prompt baseline lives in
``evaluation/eval_prompt_baseline.py``.

CLI::

    python evaluation/eval_base.py [--limit N] [--input PATH] [--output PATH]

Outputs ``data/results/base_predictions.json`` (gitignored).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from evaluation._runner import run_baseline

logger = logging.getLogger(__name__)


NAIVE_PROMPT_TEMPLATE = """Extract the legal clauses from this contract as JSON.

Contract:
{contract_text}

JSON:
"""


def build_prompt(contract_text: str) -> str:
    """Construct the naive prompt for one contract.

    Deliberately minimal: no schema, no examples, no output constraints. Any
    addition here would erode the meaning of the "naive baseline" floor.
    """
    return NAIVE_PROMPT_TEMPLATE.format(contract_text=contract_text)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Naive-prompt baseline evaluator.")
    parser.add_argument(
        "--input",
        default="data/processed/test.jsonl",
        help="Path to test set JSONL. Default: data/processed/test.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/results/base_predictions.json",
        help="Where to save predictions. Default: data/results/base_predictions.json",
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

    return run_baseline(
        prompt_builder=build_prompt,
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
