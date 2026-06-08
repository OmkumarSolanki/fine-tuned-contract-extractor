"""Load the fine-tuned model and run a single extraction — a worked example.

This is the smallest end-to-end demonstration of the serving stack *without*
the HTTP layer: it loads the LoRA adapter (default: the published Hub repo),
builds the training-format prompt, generates, and prints the parsed 12-field
extraction.

GPU-only (Unsloth + bitsandbytes). The heavy imports live inside ``main`` so
this module imports cleanly on CPU and its pure helpers stay unit-testable.

Usage::

    python scripts/run_local_inference.py --contract-file path/to/contract.txt
    python scripts/run_local_inference.py --text "This Agreement ..."
    EXTRACTOR_ADAPTER_PATH=solankiom/llama-3.1-8b-contract-extractor \
        python scripts/run_local_inference.py --text "..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# DEFAULT_ADAPTER_PATH is a plain string constant — safe to import on CPU.
from extractor.inference.model_loader import DEFAULT_ADAPTER_PATH

# A short, self-contained sample so the script does something useful with no args.
SAMPLE_CONTRACT = (
    "CO-BRANDING AGREEMENT. This Co-Branding Agreement (the \"Agreement\") is "
    "entered into as of January 1, 2020 (the \"Effective Date\") by and between "
    "Acme Corporation, a Delaware corporation (\"Acme\"), and Beta LLC, a "
    "California limited liability company (\"Beta\"). This Agreement shall be "
    "governed by the laws of the State of New York. The initial term is two (2) "
    "years and shall automatically renew for successive one-year terms unless "
    "either party gives ninety (90) days written notice prior to expiration."
)


def resolve_adapter_path(cli_value: str | None) -> str:
    """Resolve the adapter path: CLI flag > ``EXTRACTOR_ADAPTER_PATH`` > default.

    Pure (env-reading) helper so the precedence logic is unit-testable without
    loading a model.
    """
    if cli_value:
        return cli_value
    return os.environ.get("EXTRACTOR_ADAPTER_PATH") or DEFAULT_ADAPTER_PATH


def format_extraction(raw_output: str) -> str:
    """Pretty-print model output as 2-space-indented JSON.

    Falls back to returning the raw string unchanged when the output is not
    valid JSON (so the caller can still see what the model produced).
    """
    try:
        parsed = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return raw_output
    return json.dumps(parsed, indent=2, ensure_ascii=False)


def _read_contract(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.contract_file:
        with open(args.contract_file, encoding="utf-8") as fh:
            return fh.read()
    return SAMPLE_CONTRACT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one contract extraction locally.")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--text", help="Contract text to extract from.")
    src.add_argument("--contract-file", help="Path to a UTF-8 contract text file.")
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="LoRA adapter path or Hub repo id (default: EXTRACTOR_ADAPTER_PATH or built-in default).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048, help="Generation budget (default: 2048)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    adapter_path = resolve_adapter_path(args.adapter_path)
    contract_text = _read_contract(args)

    # Heavy imports deferred to runtime (GPU-only).
    from extractor.inference.model_loader import load_generator  # noqa: PLC0415
    from extractor.inference.prompt import build_messages  # noqa: PLC0415

    print(f"Loading generator from: {adapter_path}", file=sys.stderr)
    generator = load_generator(adapter_path)

    messages = build_messages(contract_text)
    raw_output, n_tokens = generator.generate(messages, args.max_new_tokens)

    print(format_extraction(raw_output))
    print(f"\n[{n_tokens} tokens generated]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
