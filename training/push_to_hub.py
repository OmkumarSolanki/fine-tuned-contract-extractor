"""Publish the QLoRA LoRA adapter to the Hugging Face Hub (Phase 10).

Uploads three things to ``solankiom/llama-3.1-8b-contract-extractor``:

1. the LoRA adapter (``adapter_config.json`` + ``adapter_model.safetensors``),
2. the tokenizer files saved alongside it,
3. a **model card** (``README.md``) built from the *real* metrics committed under
   ``data/results/`` — the three-way comparison and the training summary — so the
   card never carries placeholder numbers.

Design
------
- ``build_model_card(...)`` is **pure** (dicts in, markdown out) and is what the
  test-suite exercises on the real summary files — no network, no model load.
- The actual upload (``huggingface_hub``) is isolated in ``upload_to_hub(...)``
  with a lazy import, so importing this module stays light and CPU-safe.
- ``--dry-run`` writes the generated card next to the adapter and prints it
  **without** touching the network — use it to preview before publishing.

CLI::

    # preview the card only (no upload)
    python training/push_to_hub.py --dry-run

    # publish (needs HF_TOKEN with write scope, or --token)
    python training/push_to_hub.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "solankiom/llama-3.1-8b-contract-extractor"
DEFAULT_ADAPTER_PATH = "checkpoints/contract-extractor/final-adapter"
DEFAULT_COMPARISON_SUMMARY = "data/results/comparison_summary.json"
DEFAULT_TRAINING_SUMMARY = "data/results/training_summary.json"

# The adapter was trained on the Unsloth 4-bit mirror; weights are identical to
# the gated meta-llama release.
BASE_MODEL = "unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit"
BASE_MODEL_OFFICIAL = "meta-llama/Llama-3.1-8B-Instruct"
CUAD_DATASET = "theatticusproject/cuad-qa"

# Canonical 12-field order (kept in sync with extractor/schemas.py).
FIELD_ORDER = [
    "document_name",
    "parties",
    "agreement_date",
    "effective_date",
    "expiration_date",
    "governing_law",
    "renewal_term",
    "notice_period_to_terminate_renewal",
    "exclusivity",
    "non_compete",
    "cap_on_liability",
    "uncapped_liability",
]


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file into a dict (raises if missing/invalid)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pct(rate: float) -> str:
    """Format a 0..1 rate as a whole-number percentage string, e.g. ``96%``."""
    return f"{round(rate * 100)}%"


def _model_prompts() -> tuple[str, str]:
    """Return the exact training-time (system, user-template) prompts.

    Imported from ``training.prepare_dataset`` so the card can't drift from what
    the model was actually trained/served with.
    """
    from training.prepare_dataset import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # noqa: PLC0415

    return SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def _frontmatter(repo_id: str) -> str:
    """YAML frontmatter block for the Hub model card."""
    return (
        "---\n"
        f"base_model: {BASE_MODEL}\n"
        "library_name: peft\n"
        "license: mit\n"
        "pipeline_tag: text-generation\n"
        "language:\n- en\n"
        "tags:\n"
        "- lora\n- qlora\n- unsloth\n- trl\n- peft\n- legal\n"
        "- contract-extraction\n- llama-3.1\n"
        "datasets:\n"
        f"- {CUAD_DATASET}\n"
        "---\n"
    )


def _three_way_table(models: dict[str, Any]) -> str:
    """Render the headline three-way validity + overall_f1 table."""
    rows = [
        ("Naive baseline", "naive"),
        ("Strong-prompt baseline", "strong_prompt"),
        ("**Fine-tuned (this adapter)**", "finetuned"),
    ]
    n = models.get("finetuned", {}).get("n_contracts", 51)
    lines = [
        f"| Model | JSON-validity ({n} contracts) | `overall_f1` (CAVEATED) |",
        "|-------|:---:|:---:|",
    ]
    for label, key in rows:
        m = models[key]
        valid = m["json_valid"]
        rate = m["json_validity_rate"]
        f1 = m["per_field_match_rate_CAVEATED"]["overall_f1"]
        lines.append(f"| {label} | {valid} / {n} (**{_pct(rate)}**) | {f1:.4f} |")
    return "\n".join(lines)


def _per_field_table(finetuned: dict[str, Any]) -> str:
    """Render the fine-tuned per-field match-rate table (12 fields)."""
    field_rates = finetuned["per_field_match_rate_CAVEATED"]
    lines = ["| Field | Match rate |", "|-------|:---:|"]
    for field in FIELD_ORDER:
        if field in field_rates:
            lines.append(f"| `{field}` | {field_rates[field]:.3f} |")
    return "\n".join(lines)


def build_model_card(
    comparison: dict[str, Any],
    training: dict[str, Any],
    repo_id: str = DEFAULT_REPO_ID,
) -> str:
    """Build the full model-card markdown from the real metric summaries.

    Pure function: dicts in, markdown out. ``comparison`` is the parsed
    ``comparison_summary.json`` and ``training`` the parsed
    ``training_summary.json``.
    """
    models = comparison["models"]
    ft = models["finetuned"]
    lora = training["lora"]
    opt = training["optimization"]
    res = training["results"]
    system_prompt, user_template = _model_prompts()

    trainable = lora["trainable_params"]
    total = lora["total_params"]
    runtime_min = round(res["train_runtime_seconds"] / 60)

    card = f"""{_frontmatter(repo_id)}
# Llama 3.1 8B — Contract Clause Extractor (QLoRA adapter)

A LoRA adapter that fine-tunes **{BASE_MODEL_OFFICIAL}** to extract **12
commercially-critical contract clauses** as strict JSON, trained on the
[CUAD](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus
Dataset). Fine-tuning lifts schema-valid JSON output from **{_pct(models["naive"]["json_validity_rate"])} / {_pct(models["strong_prompt"]["json_validity_rate"])}**
(naive / strong-prompt baselines) to **{_pct(ft["json_validity_rate"])}** on a held-out test set.

- **Base model:** `{BASE_MODEL}` (4-bit; identical weights to `{BASE_MODEL_OFFICIAL}`)
- **Method:** QLoRA (Unsloth 4-bit base + LoRA) via TRL `SFTTrainer`, assistant-only loss
- **Task:** structured legal contract clause extraction (12 fields)
- **Language:** English
- **License:** MIT (adapter weights). CUAD data is CC BY 4.0 — see *License & Data*.
- **Code:** https://github.com/OmkumarSolanki/fine-tuned-contract-extractor

## The 12 fields

`document_name`, `parties`, `agreement_date`, `effective_date`, `expiration_date`,
`governing_law`, `renewal_term`, `notice_period_to_terminate_renewal`,
`exclusivity`, `non_compete`, `cap_on_liability`, `uncapped_liability`.

All non-list fields are `null` when the contract doesn't address the topic;
`parties` is a (possibly empty) list of strings.

## How to use

This is a PEFT/LoRA adapter — load the base model, then apply the adapter. Use
the **exact training prompt** (below); a different prompt degrades accuracy.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "{BASE_MODEL_OFFICIAL}", device_map="auto", load_in_4bit=True
)
model = PeftModel.from_pretrained(base, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

SYSTEM_PROMPT = {system_prompt!r}
USER_PROMPT_TEMPLATE = {user_template!r}

contract_text = "AGREEMENT made as of January 1, 2024, between Acme Corp and Beta Inc. ..."
messages = [
    {{"role": "system", "content": SYSTEM_PROMPT}},
    {{"role": "user", "content": USER_PROMPT_TEMPLATE.format(contract_text=contract_text)}},
]
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)
out = model.generate(input_ids=input_ids, max_new_tokens=2048, do_sample=False)
print(tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True))
# -> compact JSON with the 12 fields
```

> Unsloth users can instead load this repo id directly with
> `FastLanguageModel.from_pretrained(model_name="{repo_id}", load_in_4bit=True)`.

## Training

QLoRA on the 408/51/51 ChatML split (seed 42), assistant-only loss, on 1× {training["hardware"].split("(")[0].strip() if "(" in training["hardware"] else training["hardware"]}.

| Hyperparameter | Value |
|----------------|-------|
| LoRA rank / alpha / dropout | {lora["r"]} / {lora["alpha"]} / {lora["dropout"]} |
| Target modules | {lora["target_modules"]} projection modules |
| Trainable params | {trainable:,} / {total:,} (**{lora["trainable_pct"]}%**) |
| Epochs / steps | {opt["epochs"]} / {opt["total_steps"]} |
| Effective batch | {opt["effective_batch_size"]} (1 × grad-accum {opt["gradient_accumulation_steps"]}) |
| Optimizer / LR | `{opt["optimizer"]}`, {opt["learning_rate"]} ({opt["lr_scheduler"]}) |
| Precision | {opt["precision"]} |
| Best val `eval_loss` | **{res["best_eval_loss"]}** (step {res["best_eval_step"]}, kept via `load_best_model_at_end`) |
| Final mean `train_loss` | {res["final_train_loss_mean"]} |
| Runtime | ~{runtime_min} min |

## Evaluation

Held-out **{ft["n_contracts"]}-contract** test set, greedy decoding (deterministic).
The reportable metric is **JSON-validity** — the fraction of outputs that parse
as JSON *and* validate against the 12-field schema.

{_three_way_table(models)}

> **Read the per-field F1 with the validity rate, never alone.** Schema-invalid
> predictions are scored as empty extractions; because many CUAD gold fields are
> null, an empty prediction scores "correct" on those sparse fields, which
> inflates the baselines' per-field numbers. The metric is an apples-to-apples
> extraction-quality measure only once a model mostly emits valid JSON — which
> is exactly what fine-tuning achieves here.

### Fine-tuned per-field match rate (CAVEATED)

{_per_field_table(ft)}

## Limitations

- English-only, trained on commercial contracts from CUAD; out-of-distribution
  documents (other languages, non-commercial agreements) will degrade.
- Long contracts are head+tail-truncated to an 8000-token budget at training
  time; extremely long inputs may still be truncated at inference.
- Not legal advice. Outputs must be reviewed by a qualified professional.
- No authentication is built into the reference serving layer — add it before
  any public deployment.

## License & Data

- **Adapter weights:** MIT © 2026 Om Solanki.
- **Base model:** subject to the [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE).
- **Training data:** [CUAD](https://www.atticusprojectai.org/cuad) (CC BY 4.0), via the public `{CUAD_DATASET}` mirror. No CUAD-derived contract text is redistributed in this repo.

## Acknowledgments

- **The Atticus Project** — for curating and releasing CUAD.
- **Meta AI** — for Llama 3.1 8B Instruct.
- **Unsloth AI** — for the 4-bit base mirror and fast QLoRA tooling.
- **Hugging Face** — for `transformers`, `peft`, `trl`, and the Hub.

```bibtex
@article{{hendrycks2021cuad,
  title   = {{CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review}},
  author  = {{Dan Hendrycks and Collin Burns and Anya Chen and Spencer Ball}},
  journal = {{arXiv preprint arXiv:2103.06268}},
  year    = {{2021}}
}}
```
"""
    return card


def write_card(card_text: str, adapter_path: str | Path) -> Path:
    """Write the model card to ``<adapter_path>/README.md`` and return its path."""
    out = Path(adapter_path) / "README.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card_text, encoding="utf-8")
    return out


def upload_to_hub(
    adapter_path: str | Path,
    repo_id: str,
    *,
    token: Optional[str] = None,
    private: bool = False,
) -> str:
    """Create the repo (if needed) and upload the adapter folder. Returns the URL.

    Lazy-imports ``huggingface_hub`` so the module stays CPU/CI-safe. Assumes the
    model card has already been written into ``adapter_path`` (see
    :func:`write_card`).
    """
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(adapter_path), repo_id=repo_id, repo_type="model")
    url = f"https://huggingface.co/{repo_id}"
    logger.info("Uploaded adapter to %s", url)
    return url


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish the LoRA adapter + model card to the HF Hub.")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    p.add_argument("--comparison-summary", default=DEFAULT_COMPARISON_SUMMARY)
    p.add_argument("--training-summary", default=DEFAULT_TRAINING_SUMMARY)
    p.add_argument("--token", default=None, help="HF token; defaults to HF_TOKEN env var.")
    p.add_argument("--private", action="store_true", help="Create the repo as private.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build + write the model card locally and print it; do NOT upload.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        logger.error("Adapter path not found: %s", adapter_path)
        return 1

    try:
        comparison = load_json(args.comparison_summary)
        training = load_json(args.training_summary)
    except FileNotFoundError as exc:
        logger.error("Missing metrics summary (%s). Run evaluation/compare.py first.", exc)
        return 1

    card = build_model_card(comparison, training, repo_id=args.repo_id)
    card_path = write_card(card, adapter_path)
    logger.info("Model card written to %s (%d chars)", card_path, len(card))

    if args.dry_run:
        print("\n" + "=" * 70 + "\nDRY RUN — model card preview (not uploaded):\n" + "=" * 70)
        print(card)
        return 0

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        logger.error("No HF token. Set HF_TOKEN or pass --token (needs write scope).")
        return 1

    url = upload_to_hub(adapter_path, args.repo_id, token=token, private=args.private)
    logger.info("Done. Adapter live at %s", url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
