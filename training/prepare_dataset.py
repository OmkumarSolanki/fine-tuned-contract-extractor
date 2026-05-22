"""Convert ``data/raw/cuad_parsed.jsonl`` into ChatML training splits.

Each line of the output JSONLs has the shape::

    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "Extract structured clauses ..."},
            {"role": "assistant", "content": "<compact JSON of the 12 fields>"}
        ],
        "contract_id": "<contract title>"
    }

The Llama 3.1 chat template is applied at training time by ``SFTTrainer``;
we keep the structured ``messages`` representation here so assistant-only loss
masking stays clean and the files remain debuggable. We only render once at
startup as a sanity check, so any drift in the tokenizer's chat template
becomes visible in the logs.

Tokenizer source priority (per project plan):

1. ``meta-llama/Llama-3.1-8B-Instruct`` (gated) if ``HF_TOKEN`` is set.
2. ``unsloth/Meta-Llama-3.1-8B-Instruct`` (public mirror) otherwise.

CLI:

    python training/prepare_dataset.py [--input PATH] [--output-dir PATH] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

from extractor.schemas import ContractExtraction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a legal contract analyst. Extract structured clauses from contracts."
)
USER_PROMPT_TEMPLATE = "Extract structured clauses from this contract:\n\n{contract_text}"

PRIMARY_TOKENIZER = "meta-llama/Llama-3.1-8B-Instruct"
FALLBACK_TOKENIZER = "unsloth/Meta-Llama-3.1-8B-Instruct"

MAX_TOTAL_TOKENS = 8000
HEAD_TOKENS = 5000
TAIL_TOKENS = 3000
TRUNC_MARKER = "\n[...TRUNCATED...]\n"

SEED = 42


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------


def load_tokenizer() -> tuple[Any, str]:
    """Load the Llama 3.1 chat tokenizer with HF_TOKEN→unsloth fallback.

    Returns the tokenizer and the source string actually used (for logging).
    Raises ``RuntimeError`` if both sources fail.
    """
    # Lazy import so this module can be imported in test environments without
    # transformers (which is heavy and not always available).
    from transformers import AutoTokenizer

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # noqa: BLE001
        # python-dotenv not installed → fall back to whatever's in os.environ.
        pass

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            tok = AutoTokenizer.from_pretrained(PRIMARY_TOKENIZER, token=hf_token)
            logger.info(
                "Loaded tokenizer from %s (gated, via HF_TOKEN)", PRIMARY_TOKENIZER
            )
            return tok, PRIMARY_TOKENIZER
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load %s with HF_TOKEN (%s); falling back to %s.",
                PRIMARY_TOKENIZER,
                exc,
                FALLBACK_TOKENIZER,
            )
    else:
        logger.info(
            "HF_TOKEN not set; using public mirror %s.", FALLBACK_TOKENIZER
        )

    try:
        tok = AutoTokenizer.from_pretrained(FALLBACK_TOKENIZER)
        logger.info(
            "Loaded tokenizer from %s (unsloth mirror)", FALLBACK_TOKENIZER
        )
        return tok, FALLBACK_TOKENIZER
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to load both the gated meta-llama tokenizer and the unsloth "
            "fallback. Either set HF_TOKEN (after accepting the Llama 3.1 license "
            "at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) or check "
            f"your network connection. Last error: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def truncate_text(
    text: str,
    tokenizer: Any,
    max_total: int = MAX_TOTAL_TOKENS,
    head: int = HEAD_TOKENS,
    tail: int = TAIL_TOKENS,
) -> str:
    """Head + tail truncation. Returns ``text`` unchanged if under budget.

    Long contracts often have key clauses (governing law, uncapped liability,
    liability caps) at the end, so we keep ``head`` tokens from the start and
    ``tail`` tokens from the end joined by :data:`TRUNC_MARKER`.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_total:
        return text
    head_text = tokenizer.decode(ids[:head], skip_special_tokens=True)
    tail_text = tokenizer.decode(ids[-tail:], skip_special_tokens=True)
    return head_text + TRUNC_MARKER + tail_text


def compact_json(annotations: dict) -> str:
    """Serialize annotations as compact JSON in canonical field order.

    The key order matches :attr:`ContractExtraction.model_fields` so the
    training target is byte-identical for equivalent inputs.
    """
    ordered = {
        field: annotations.get(field) for field in ContractExtraction.model_fields
    }
    # ``parties`` defaults to [] when missing in source; preserve that.
    if ordered["parties"] is None:
        ordered["parties"] = []
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def build_messages(
    contract_id: str,
    contract_text: str,
    annotations: dict,
    tokenizer: Any,
) -> dict:
    """Build a single training row: messages list + contract_id.

    Validates ``annotations`` against :class:`ContractExtraction`. On failure,
    raises ``ValueError`` with a short reason; the caller is expected to log
    and drop the row.
    """
    try:
        extraction = ContractExtraction.model_validate(annotations)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).replace("\n", " | ")
        raise ValueError(msg[:200] + ("…" if len(msg) > 200 else "")) from exc

    truncated = truncate_text(contract_text, tokenizer)
    assistant_content = compact_json(extraction.model_dump())

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(contract_text=truncated),
            },
            {"role": "assistant", "content": assistant_content},
        ],
        "contract_id": contract_id,
    }


def split_indices(n: int, seed: int = SEED) -> tuple[list[int], list[int], list[int]]:
    """Deterministic 80/10/10 index split for ``n`` items.

    Returns ``(train_idx, val_idx, test_idx)`` after a seeded shuffle.
    """
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def split_rows(rows: list[dict], seed: int = SEED) -> tuple[list[dict], list[dict], list[dict]]:
    """80/10/10 split of ``rows`` with deterministic shuffle."""
    train_idx, val_idx, test_idx = split_indices(len(rows), seed=seed)
    train = [rows[i] for i in train_idx]
    val = [rows[i] for i in val_idx]
    test = [rows[i] for i in test_idx]
    return train, val, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _log_chat_template_preview(tokenizer: Any, sample_messages: list[dict]) -> None:
    """Render one example through the tokenizer's chat template and log a preview."""
    try:
        rendered = tokenizer.apply_chat_template(
            sample_messages, tokenize=False, add_generation_prompt=False
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not render chat template preview: %s", exc)
        return
    head_preview = rendered[:200].replace("\n", "\\n")
    tail_preview = rendered[-200:].replace("\n", "\\n")
    logger.info("Chat template head: %s", head_preview)
    logger.info("Chat template tail: %s", tail_preview)
    logger.info("Rendered length: %d chars", len(rendered))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input",
        default="data/raw/cuad_parsed.jsonl",
        help="Path to ingest_cuad.py output. Default: data/raw/cuad_parsed.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for train.jsonl/val.jsonl/test.jsonl. Default: data/processed",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for the 80/10/10 split. Default: {SEED}",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(
            "Input file not found: %s. Run `python training/ingest_cuad.py` first.",
            input_path,
        )
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, source = load_tokenizer()
    logger.info("Tokenizer source: %s", source)

    raw_rows = _read_jsonl(input_path)
    logger.info("Read %d raw contracts from %s", len(raw_rows), input_path)

    # Build ChatML rows; drop invalid ones with a single log line each.
    built: list[dict] = []
    n_dropped = 0
    for raw in raw_rows:
        contract_id = raw.get("contract_id", "<unknown>")
        try:
            built.append(
                build_messages(
                    contract_id=contract_id,
                    contract_text=raw["contract_text"],
                    annotations=raw["annotations"],
                    tokenizer=tokenizer,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dropped %s: %s", contract_id, exc)
            n_dropped += 1

    logger.info("Built %d ChatML rows (%d dropped)", len(built), n_dropped)

    if built:
        _log_chat_template_preview(tokenizer, built[0]["messages"])

    train, val, test = split_rows(built, seed=args.seed)
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "val.jsonl", val)
    _write_jsonl(output_dir / "test.jsonl", test)

    logger.info(
        "Train: %d, Val: %d, Test: %d, Dropped: %d (output_dir=%s)",
        len(train),
        len(val),
        len(test),
        n_dropped,
        output_dir,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    sys.exit(main())
