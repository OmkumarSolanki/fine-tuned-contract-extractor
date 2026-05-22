"""Shared utilities for the baseline evaluators.

Both ``eval_base.py`` and ``eval_prompt_baseline.py`` use the same:

- test-set loader (reads ``data/processed/test.jsonl``)
- contract-text extractor (strips the training-time user-prompt wrapper)
- prediction parser (model string → ``ContractExtraction`` or ``None``)
- prediction-record shape (``{contract_id, raw_output, parsed, is_valid_json}``)
- output writer (single JSON array, not JSONL)

The model-load and generation helpers (:func:`load_model`, :func:`generate_one`)
import ``transformers`` and ``torch`` lazily so this module stays test-friendly:
the pure-Python helpers don't trigger any heavy imports at module-load time, and
the unit tests never have to touch a real model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from extractor.schemas import ContractExtraction

logger = logging.getLogger(__name__)


# Must match training/prepare_dataset.py:USER_PROMPT_TEMPLATE so the baseline
# evaluators can recover the raw contract text from a wrapped user message.
USER_PROMPT_PREFIX = "Extract structured clauses from this contract:\n\n"


# ---------------------------------------------------------------------------
# Pure-Python helpers (always safe to import; tested in isolation)
# ---------------------------------------------------------------------------


def extract_contract_text(user_message_content: str) -> str:
    """Recover the raw contract body from a training-style user message.

    The training pipeline prepends a fixed prefix; the baseline evaluators
    need the raw body so they can wrap it in their own prompts. If the prefix
    is absent (e.g. malformed input), returns the input unchanged.
    """
    if user_message_content.startswith(USER_PROMPT_PREFIX):
        return user_message_content[len(USER_PROMPT_PREFIX):]
    return user_message_content


def parse_prediction(raw_output: str) -> Optional[dict]:
    """Try to parse a model output string as a valid extraction.

    Returns the dict on success (validates against
    :class:`~extractor.schemas.ContractExtraction`). Returns ``None`` if the
    output is not parseable JSON or fails schema validation. Never raises.
    """
    try:
        data = json.loads(raw_output)
        ContractExtraction.model_validate(data)
        return data
    except Exception:
        return None


def make_prediction_record(contract_id: str, raw_output: str) -> dict:
    """Build the canonical baseline prediction record.

    Even invalid predictions are recorded with ``is_valid_json=False`` and
    ``parsed=None``: the JSON validity rate is part of the comparison metric,
    so dropping invalid rows would silently inflate it.
    """
    parsed = parse_prediction(raw_output)
    return {
        "contract_id": contract_id,
        "raw_output": raw_output,
        "parsed": parsed,
        "is_valid_json": parsed is not None,
    }


def load_test_examples(
    path: Path | str,
    limit: Optional[int] = None,
) -> list[dict]:
    """Read ``test.jsonl`` into ``[{contract_id, contract_text}, ...]``.

    Strips the training-time user-prompt wrapper so each baseline can
    construct its own prompt around the raw contract body.
    """
    path = Path(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            user_msg = row["messages"][1]["content"]
            rows.append(
                {
                    "contract_id": row["contract_id"],
                    "contract_text": extract_contract_text(user_msg),
                }
            )
    if limit is not None:
        rows = rows[:limit]
    return rows


def write_predictions(records: list[dict], path: Path | str) -> None:
    """Write the prediction records as a single indented JSON array.

    Baseline prediction outputs are JSON, not JSONL, so they can be diffed
    and inspected with ``jq`` end-to-end. The directory is created on demand.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Model + generation (heavy; lazy-imported)
# ---------------------------------------------------------------------------


PRIMARY_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FALLBACK_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"


def load_model(model_name: Optional[str] = None) -> tuple[Any, Any]:
    """Load tokenizer + base model in bf16 with HF_TOKEN→unsloth fallback.

    Mirrors the loader strategy in ``training/prepare_dataset.py``: tries the
    gated ``meta-llama/...`` repo first if ``HF_TOKEN`` is set, then falls back
    to the public ``unsloth/...`` mirror. Both ship the same weights and chat
    template.

    Lazy-imports ``transformers`` and ``torch`` so unit tests can mock the
    function without paying the import cost.

    Parameters
    ----------
    model_name:
        Override the primary repo. Useful only for debugging; production runs
        should let the default chain pick.
    """
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # noqa: BLE001
        # python-dotenv not installed → fall back to whatever's in os.environ.
        pass

    primary = model_name or PRIMARY_MODEL
    candidates: list[tuple[str, Optional[str], str]] = []
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        candidates.append((primary, hf_token, "gated, via HF_TOKEN"))
    candidates.append((FALLBACK_MODEL, None, "unsloth mirror"))

    last_exc: Optional[Exception] = None
    for repo, token, label in candidates:
        try:
            kw: dict[str, Any] = {}
            if token:
                kw["token"] = token
            tokenizer = AutoTokenizer.from_pretrained(repo, **kw)
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                **kw,
            )
            logger.info("Loaded base model from %s (%s)", repo, label)
            return tokenizer, model
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s (%s): %s", repo, label, exc)
            last_exc = exc

    raise RuntimeError(
        "Failed to load Llama 3.1 8B Instruct from both the gated meta-llama "
        "source and the unsloth fallback. Either set HF_TOKEN (after accepting "
        "the Llama 3.1 license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) "
        f"or check your network. Last error: {last_exc}"
    )


def generate_one(
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int = 2048,
) -> str:
    """Single-prompt greedy decode. Returns the model's continuation only.

    ``do_sample=False`` is the standard "temperature 0" greedy setting — fully
    deterministic given (model, tokenizer, prompt). The prompt tokens are
    stripped from the output so the returned string is exactly what the model
    generated.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# End-to-end driver
# ---------------------------------------------------------------------------


def run_baseline(
    prompt_builder: Callable[[str], str],
    test_path: Path | str,
    output_path: Path | str,
    *,
    limit: Optional[int] = None,
    max_new_tokens: int = 2048,
    tokenizer: Optional[Any] = None,
    model: Optional[Any] = None,
) -> int:
    """Drive a baseline evaluation end-to-end.

    Loads the test set, iterates each contract through ``prompt_builder`` and
    the model, and writes the prediction records (including invalid ones) to
    ``output_path``.

    If ``tokenizer`` and ``model`` are both supplied (the unit-test path), no
    model load is attempted. Otherwise both are loaded fresh via
    :func:`load_model`.

    Returns the process exit code (``0`` on success, ``1`` on input error).
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
        raw_output = generate_one(tokenizer, model, prompt, max_new_tokens)
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
