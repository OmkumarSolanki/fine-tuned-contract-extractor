"""Unit tests for the training driver (``training/train.py``).

All tests are CPU-only and import nothing heavy: ``train.py`` keeps ``unsloth``,
``trl``, ``torch`` and ``datasets`` behind lazy imports inside ``main`` /
``_load_model_and_tokenizer``, so the pure-Python helpers exercised here run
without a GPU stack installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training import train


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPO_ROOT / "training" / "configs" / "llama_8b_qlora.yaml"


def _minimal_cfg() -> dict:
    """A complete, valid in-memory config for mapping tests."""
    return {
        "model": {"base_model": "meta-llama/Llama-3.1-8B-Instruct", "max_seq_length": 8192},
        "data": {"train_path": "data/processed/train.jsonl", "val_path": "data/processed/val.jsonl"},
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "training": {
            "output_dir": "checkpoints/contract-extractor",
            "num_epochs": 3,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2.0e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "logging_steps": 10,
            "eval_strategy": "steps",
            "eval_steps": 50,
            "save_strategy": "steps",
            "save_steps": 100,
            "save_total_limit": 2,
            "bf16": True,
            "optim": "adamw_8bit",
            "weight_decay": 0.0,
            "seed": 42,
        },
    }


class FakeTokenizer:
    """Records the last ``apply_chat_template`` call and returns a canned string."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return "RENDERED_CHAT_TEXT"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_reads_shipped_config():
    cfg = train.load_config(SHIPPED_CONFIG)
    assert cfg["model"]["base_model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg["model"]["max_seq_length"] == 8192
    assert cfg["lora"]["r"] == 16
    assert cfg["lora"]["target_modules"][0] == "q_proj"
    assert len(cfg["lora"]["target_modules"]) == 7


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        train.load_config(tmp_path / "nope.yaml")


def test_load_config_missing_section_raises(tmp_path):
    p = tmp_path / "partial.yaml"
    p.write_text("model:\n  base_model: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required section"):
        train.load_config(p)


def test_load_config_non_mapping_raises(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not parse to a mapping"):
        train.load_config(p)


# ---------------------------------------------------------------------------
# build_sft_kwargs
# ---------------------------------------------------------------------------


def test_build_sft_kwargs_core_mapping():
    kwargs = train.build_sft_kwargs(_minimal_cfg())
    assert kwargs["num_train_epochs"] == 3
    assert kwargs["per_device_train_batch_size"] == 1
    assert kwargs["gradient_accumulation_steps"] == 8
    assert kwargs["lr_scheduler_type"] == "cosine"
    assert kwargs["optim"] == "adamw_8bit"
    assert kwargs["seed"] == 42
    assert kwargs["bf16"] is True


def test_build_sft_kwargs_learning_rate_is_float():
    kwargs = train.build_sft_kwargs(_minimal_cfg())
    assert isinstance(kwargs["learning_rate"], float)
    assert kwargs["learning_rate"] == pytest.approx(2.0e-4)


def test_build_sft_kwargs_best_checkpoint_wiring():
    kwargs = train.build_sft_kwargs(_minimal_cfg())
    assert kwargs["load_best_model_at_end"] is True
    assert kwargs["metric_for_best_model"] == "eval_loss"
    assert kwargs["greater_is_better"] is False
    assert kwargs["report_to"] == "none"


def test_build_sft_kwargs_handles_string_scientific_lr():
    cfg = _minimal_cfg()
    cfg["training"]["learning_rate"] = "2.0e-4"  # some YAML forms parse this as str
    kwargs = train.build_sft_kwargs(cfg)
    assert kwargs["learning_rate"] == pytest.approx(2.0e-4)


def test_build_sft_kwargs_mismatched_strategies_raise():
    cfg = _minimal_cfg()
    cfg["training"]["save_strategy"] = "epoch"
    with pytest.raises(ValueError, match="must match"):
        train.build_sft_kwargs(cfg)


def test_build_sft_kwargs_save_not_multiple_of_eval_raises():
    cfg = _minimal_cfg()
    cfg["training"]["save_steps"] = 75  # not a multiple of eval_steps=50
    with pytest.raises(ValueError, match="multiple of eval_steps"):
        train.build_sft_kwargs(cfg)


def test_shipped_config_builds_valid_sft_kwargs():
    """The actual shipped YAML must produce a self-consistent kwargs set."""
    cfg = train.load_config(SHIPPED_CONFIG)
    kwargs = train.build_sft_kwargs(cfg)
    # save_steps is a multiple of eval_steps (precondition for best-checkpoint)
    assert kwargs["save_steps"] % kwargs["eval_steps"] == 0
    assert kwargs["load_best_model_at_end"] is True


# ---------------------------------------------------------------------------
# render_text
# ---------------------------------------------------------------------------


def test_render_text_returns_text_field():
    tok = FakeTokenizer()
    example = {"messages": [{"role": "user", "content": "hi"}], "contract_id": "X"}
    out = train.render_text(example, tok)
    assert out == {"text": "RENDERED_CHAT_TEXT"}


def test_render_text_passes_correct_template_flags():
    tok = FakeTokenizer()
    example = {"messages": [{"role": "system", "content": "s"}]}
    train.render_text(example, tok)
    call = tok.calls[-1]
    # assistant turn is already present, so no generation prompt; not tokenized
    assert call["add_generation_prompt"] is False
    assert call["tokenize"] is False
    assert call["messages"] == example["messages"]


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------


def test_llama31_markers_are_distinct_and_nonempty():
    assert train.LLAMA31_INSTRUCTION_PART
    assert train.LLAMA31_RESPONSE_PART
    assert train.LLAMA31_INSTRUCTION_PART != train.LLAMA31_RESPONSE_PART
    assert "assistant" in train.LLAMA31_RESPONSE_PART
    assert "user" in train.LLAMA31_INSTRUCTION_PART
