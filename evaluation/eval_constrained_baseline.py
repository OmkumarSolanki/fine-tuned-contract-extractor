"""Constrained-decoding baseline — the *fair* baseline for the comparison.

The naive and strong-prompt baselines fail mostly on **formatting** (markdown
fences, prose, truncation) — see ``docs/EVALUATION.md`` §"failure modes", where
the naive base model is 51/51 markdown-fenced. That formatting failure is fixable
**without fine-tuning** by constraining the decoder to emit only schema-valid
JSON. This baseline does exactly that: the *same* base Llama 3.1 8B and the *same*
strong prompt as ``eval_prompt_baseline.py``, but generation is constrained to the
12-field ``ContractExtraction`` JSON schema via a grammar/logits constraint.

It exists to answer the central fine-tuning question honestly: once both the base
model and the fine-tuned model are *guaranteed* to emit valid JSON, **how much of
the field-accuracy lift survives?** That surviving ``overall_f1`` delta — not the
0%→96% validity jump — is the fine-tune's real contribution.

Constraint backend: ``lm-format-enforcer`` (a ``prefix_allowed_tokens_fn`` that
plugs straight into HF ``model.generate``). It's lazy-imported so this module
stays CPU-importable for the test suite; install it on the GPU host alongside the
model stack (it is intentionally **not** in ``pyproject.toml``).

CLI::

    python evaluation/eval_constrained_baseline.py [--limit N] [--input PATH] \
        [--train PATH] [--output PATH]

Outputs ``data/results/constrained_baseline_predictions.json`` (gitignored).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from extractor.schemas import ContractExtraction
from evaluation._runner import (
    load_model,
    load_test_examples,
    make_prediction_record,
    write_predictions,
)
from evaluation.eval_prompt_baseline import (
    build_schema_description,
    build_strong_prompt,
    format_few_shot_example,
    load_train_rows,
    select_few_shot_examples,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON schema for the decoder constraint (drift-proof: derived from the model)
# ---------------------------------------------------------------------------


def build_json_schema() -> dict:
    """Build the JSON schema the decoder is constrained to.

    Derived from :attr:`ContractExtraction.model_fields` so it can never drift
    from the live schema: ``parties`` is an array of strings, every other field
    is ``string | null`` (as ``anyOf`` — the form the constraint parser supports
    most reliably), all 12 are required, and no extra keys are allowed.
    """
    properties: dict[str, dict] = {}
    for name in ContractExtraction.model_fields:
        if name == "parties":
            properties[name] = {"type": "array", "items": {"type": "string"}}
        else:
            properties[name] = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    return {
        "type": "object",
        "properties": properties,
        "required": list(ContractExtraction.model_fields),
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Constrained generation (heavy; lazy-imported)
# ---------------------------------------------------------------------------


def generate_constrained(
    tokenizer: Any,
    model: Any,
    prompt: str,
    schema: dict,
    max_new_tokens: int = 2048,
) -> str:
    """Greedy decode constrained to ``schema`` via lm-format-enforcer.

    Builds a ``prefix_allowed_tokens_fn`` from a ``JsonSchemaParser`` and passes
    it to the standard HF ``model.generate`` — so the only difference from
    :func:`evaluation._runner.generate_one` is that every step is restricted to
    tokens that keep the output a valid prefix of the schema. Deterministic
    (``do_sample=False``). Returns the generated continuation only.

    Lazy-imports ``torch`` and ``lmformatenforcer`` so the module stays
    CPU-importable; install ``lm-format-enforcer`` on the GPU host.
    """
    import torch  # noqa: PLC0415
    from lmformatenforcer import JsonSchemaParser  # noqa: PLC0415
    from lmformatenforcer.integrations.transformers import (  # noqa: PLC0415
        build_transformers_prefix_allowed_tokens_fn,
    )

    parser = JsonSchemaParser(schema)
    prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=prefix_fn,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_constrained_baseline(
    prompt_builder,
    schema: dict,
    test_path: Path | str,
    output_path: Path | str,
    *,
    limit: Optional[int] = None,
    max_new_tokens: int = 2048,
    tokenizer: Optional[Any] = None,
    model: Optional[Any] = None,
) -> int:
    """Drive the constrained baseline end-to-end (mirrors ``run_baseline``).

    Same shape as the other evaluators — loads the test set, builds the strong
    prompt per contract, generates schema-constrained, and writes prediction
    records (including any that still fail validation, e.g. on truncation) so
    ``compare.py`` / ``analysis.py`` score it identically.

    If ``tokenizer`` and ``model`` are both supplied (the unit-test path), no
    model load happens. Returns ``0`` on success, ``1`` on input error.
    """
    examples = load_test_examples(test_path, limit=limit)
    if not examples:
        logger.error("No test examples found at %s", test_path)
        return 1

    if tokenizer is None or model is None:
        tokenizer, model = load_model()

    records: list[dict] = []
    for i, ex in enumerate(examples, start=1):
        prompt = prompt_builder(ex["contract_text"])
        raw_output = generate_constrained(tokenizer, model, prompt, schema, max_new_tokens)
        rec = make_prediction_record(ex["contract_id"], raw_output)
        records.append(rec)
        logger.info(
            "[%d/%d] %s — is_valid_json=%s",
            i,
            len(examples),
            ex["contract_id"][:60],
            rec["is_valid_json"],
        )

    write_predictions(records, output_path)
    n_valid = sum(1 for r in records if r["is_valid_json"])
    logger.info(
        "Wrote %d predictions to %s (json_validity_rate=%.1f%%)",
        len(records),
        output_path,
        100 * n_valid / len(records),
    )
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Constrained-decoding baseline evaluator.")
    parser.add_argument("--input", default="data/processed/test.jsonl")
    parser.add_argument("--train", default="data/processed/train.jsonl")
    parser.add_argument("--output", default="data/results/constrained_baseline_predictions.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args(argv)

    schema_description = build_schema_description()
    schema = build_json_schema()

    train_rows = load_train_rows(args.train)
    if not train_rows:
        logger.error("No train rows found at %s — required for few-shot.", args.train)
        return 1

    picks = select_few_shot_examples(train_rows)
    few_shot_block = "\n\n".join(format_few_shot_example(p) for p in picks)

    def builder(contract_text: str) -> str:
        return build_strong_prompt(contract_text, schema_description, few_shot_block)

    return run_constrained_baseline(
        prompt_builder=builder,
        schema=schema,
        test_path=args.input,
        output_path=args.output,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(main())
