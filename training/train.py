"""Phase 6 — QLoRA fine-tuning driver for the contract extractor.

Fine-tunes Llama 3.1 8B Instruct on the deterministic 408/51/51 ChatML splits
produced by ``training/prepare_dataset.py``, using **Unsloth** (4-bit base +
fast LoRA kernels) and **TRL**'s ``SFTTrainer`` with **assistant-only loss** —
only the JSON answer contributes to the loss, never the contract text in the
prompt. The output is a small LoRA adapter saved under ``checkpoints/``.

WHERE TO RUN
------------
Linux + CUDA only (the 4-bit path needs bitsandbytes + a GPU). The recommended
platform is a RunPod A100 80GB on the
``runpod/pytorch:2.5.0-py3.11-cuda12.4.1-devel-ubuntu22.04`` template.

Pre-flight on the pod::

    pip install -e ".[dev]"          # base deps (incl. pyyaml, accelerate, datasets<4)
    pip install unsloth              # GPU training stack (pulls trl, peft, bitsandbytes)
    python training/ingest_cuad.py && python training/prepare_dataset.py   # rebuild splits

Then::

    python training/train.py                                   # uses training/configs/llama_8b_qlora.yaml
    python training/train.py --config path/to/other.yaml       # override config

DESIGN NOTE
-----------
The heavy imports (``unsloth``, ``trl``, ``torch``, ``datasets``) are performed
lazily inside :func:`main`, so this module imports cleanly on CPU with no GPU
stack installed. The pure-Python helpers below (config loading, SFT-kwargs
mapping, chat-template rendering) are unit-tested without a model.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = "training/configs/llama_8b_qlora.yaml"

# Fallback mirror for the gated base model, mirroring prepare_dataset.py /
# evaluation/_runner.py. Both ship identical weights + chat template.
FALLBACK_BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"

# Llama 3.1 chat-template markers used to mask everything except the assistant
# turn (assistant-only loss). These must match the template applied by
# ``tokenizer.apply_chat_template`` in :func:`render_text`.
LLAMA31_INSTRUCTION_PART = "<|start_header_id|>user<|end_header_id|>\n\n"
LLAMA31_RESPONSE_PART = "<|start_header_id|>assistant<|end_header_id|>\n\n"


# ---------------------------------------------------------------------------
# Pure-Python helpers (always safe to import; unit-tested without a model)
# ---------------------------------------------------------------------------


def load_config(path: Path | str) -> dict:
    """Load and lightly validate the YAML training config.

    Ensures the four required top-level sections (``model``, ``data``,
    ``lora``, ``training``) are present so a misconfigured run fails fast with
    a clear message instead of a deep ``KeyError`` mid-setup.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")
    for section in ("model", "data", "lora", "training"):
        if section not in cfg:
            raise ValueError(f"Config is missing required section: '{section}'")
    return cfg


def build_sft_kwargs(cfg: dict) -> dict:
    """Map the ``training:`` block of the config to TRL ``SFTConfig`` kwargs.

    Kept pure (returns a plain dict) so the mapping — including the
    best-checkpoint wiring — is unit-testable without importing TRL.

    ``load_best_model_at_end=True`` + ``metric_for_best_model="eval_loss"``
    (lower is better) is what makes the trainer keep the *lowest-eval-loss*
    checkpoint rather than the last one. For this to be valid, ``save_steps``
    must be an integer multiple of ``eval_steps`` and the strategies must
    match — both enforced here.
    """
    t = cfg["training"]

    eval_steps = int(t["eval_steps"])
    save_steps = int(t["save_steps"])
    if t["eval_strategy"] != t["save_strategy"]:
        raise ValueError(
            "eval_strategy and save_strategy must match for load_best_model_at_end "
            f"(got {t['eval_strategy']!r} vs {t['save_strategy']!r})."
        )
    if t["save_strategy"] == "steps" and save_steps % eval_steps != 0:
        raise ValueError(
            f"save_steps ({save_steps}) must be a multiple of eval_steps "
            f"({eval_steps}) so the best checkpoint aligns with an eval."
        )

    return {
        "output_dir": t["output_dir"],
        "num_train_epochs": int(t["num_epochs"]),
        "per_device_train_batch_size": int(t["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(t["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(t["gradient_accumulation_steps"]),
        # YAML may parse "2.0e-4" as a string depending on form; force float.
        "learning_rate": float(t["learning_rate"]),
        "lr_scheduler_type": t["lr_scheduler_type"],
        "warmup_ratio": float(t["warmup_ratio"]),
        "logging_steps": int(t["logging_steps"]),
        "eval_strategy": t["eval_strategy"],
        "eval_steps": eval_steps,
        "save_strategy": t["save_strategy"],
        "save_steps": save_steps,
        "save_total_limit": int(t["save_total_limit"]),
        "bf16": bool(t["bf16"]),
        "optim": t["optim"],
        "weight_decay": float(t["weight_decay"]),
        "seed": int(t["seed"]),
        # keep the lowest-eval-loss checkpoint, not the last
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
    }


def render_text(example: dict, tokenizer: Any) -> dict:
    """Render one ChatML row to a single training string via the chat template.

    Applies the *same* Llama 3.1 chat template used at data-prep time, so the
    text the model trains on is byte-identical to what it will see at
    inference. ``add_generation_prompt=False`` because the assistant turn is
    already present in the row.

    Returns ``{"text": <rendered string>}`` for use as TRL's
    ``dataset_text_field``.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Training orchestration (heavy; lazy-imported)
# ---------------------------------------------------------------------------


def _load_model_and_tokenizer(cfg: dict) -> tuple[Any, Any]:
    """Load the 4-bit base model + tokenizer via Unsloth, with mirror fallback.

    Tries the configured (gated) base first; on failure falls back to the
    public unsloth mirror. Lazy-imports ``unsloth`` so the module stays
    CPU-importable.
    """
    from unsloth import FastLanguageModel  # noqa: PLC0415

    primary = cfg["model"]["base_model"]
    max_seq_length = int(cfg["model"]["max_seq_length"])

    for repo in (primary, FALLBACK_BASE_MODEL):
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=repo,
                max_seq_length=max_seq_length,
                dtype=None,          # auto-detect bf16 on supported GPUs
                load_in_4bit=True,
            )
            logger.info("Loaded 4-bit base model from %s", repo)
            return model, tokenizer
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", repo, exc)

    raise RuntimeError(
        "Could not load the base model from either the gated repo or the "
        "unsloth mirror. Set HF_TOKEN (after accepting the Llama 3.1 license) "
        "or check the network."
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for the contract extractor.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the YAML training config. Default: {DEFAULT_CONFIG_PATH}",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    # Heavy imports happen here so the module stays CPU-importable for tests.
    import torch  # noqa: F401, PLC0415  (imported for side effects / availability check)
    from datasets import load_dataset  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415
    from unsloth import FastLanguageModel  # noqa: PLC0415
    from unsloth.chat_templates import train_on_responses_only  # noqa: PLC0415

    # 1) Base model (4-bit) + tokenizer
    model, tokenizer = _load_model_and_tokenizer(cfg)

    # 2) Attach LoRA adapters
    lora = cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(cfg["training"]["seed"]),
    )

    # 3) Data — load the ChatML JSONL splits and render to a single text field
    data_files = {
        "train": cfg["data"]["train_path"],
        "validation": cfg["data"]["val_path"],
    }
    ds = load_dataset("json", data_files=data_files)
    ds = ds.map(lambda ex: render_text(ex, tokenizer), desc="Applying chat template")

    # 4) Trainer
    sft_config = SFTConfig(
        **build_sft_kwargs(cfg),
        max_seq_length=int(cfg["model"]["max_seq_length"]),
        dataset_text_field="text",
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        args=sft_config,
    )

    # 5) Assistant-only loss: mask everything but the assistant turn so the
    #    contract text in the prompt never contributes to the loss.
    trainer = train_on_responses_only(
        trainer,
        instruction_part=LLAMA31_INSTRUCTION_PART,
        response_part=LLAMA31_RESPONSE_PART,
    )

    # 6) Train
    logger.info("Starting training…")
    trainer.train()

    # 7) Save the (best) adapter + tokenizer
    out_dir = Path(cfg["training"]["output_dir"]) / "final-adapter"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Saved LoRA adapter + tokenizer to %s", out_dir)

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    raise SystemExit(main())
