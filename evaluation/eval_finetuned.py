"""Fine-tuned model evaluator for the three-way comparison.

Loads the QLoRA-fine-tuned LoRA adapter (produced by ``training/train.py``)
on top of its 4-bit base via Unsloth, and runs it over the held-out test set.

Unlike the two baselines (``eval_base.py`` / ``eval_prompt_baseline.py``),
which wrap the raw contract body in their own engineered prompts, the
fine-tuned model is evaluated with the **exact ChatML prompt it was trained
on** — the fixed system turn plus the ``"Extract structured clauses from this
contract:\\n\\n<contract>"`` user turn, read straight out of ``test.jsonl`` and
rendered through the chat template with a generation prompt appended. That
keeps train/inference parity, which is the whole point of fine-tuning for a
strict output format.

Greedy decoding (``do_sample=False``) keeps the run deterministic, matching the
baselines so the three-way comparison is apples-to-apples.

WHERE TO RUN
------------
Linux + CUDA (the 4-bit adapter load needs bitsandbytes + a GPU + unsloth),
i.e. the same RunPod box used for training. The pure-Python helpers are
unit-tested on CPU with a mocked model; the real run happens on the pod.

CLI::

    python evaluation/eval_finetuned.py \
        [--adapter checkpoints/contract-extractor/final-adapter] \
        [--input data/processed/test.jsonl] \
        [--output data/results/finetuned_predictions.json] \
        [--limit N] [--max-new-tokens N]

Outputs ``data/results/finetuned_predictions.json`` (gitignored — it embeds
CUAD-derived text). The text-free three-way summary is produced by
``evaluation/compare.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from evaluation._runner import DEFAULT_ADAPTER_PATH, run_finetuned

logger = logging.getLogger(__name__)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tuned (QLoRA) model evaluator.")
    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER_PATH,
        help=f"Path to the trained LoRA adapter directory. Default: {DEFAULT_ADAPTER_PATH}",
    )
    parser.add_argument(
        "--input",
        default="data/processed/test.jsonl",
        help="Path to test set JSONL. Default: data/processed/test.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/results/finetuned_predictions.json",
        help="Where to save predictions. Default: data/results/finetuned_predictions.json",
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

    return run_finetuned(
        test_path=args.input,
        output_path=args.output,
        adapter_path=args.adapter,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    sys.exit(main())
