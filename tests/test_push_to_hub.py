"""Unit tests for ``training/push_to_hub.py`` (Phase 10 — Hub publishing).

CPU-only and offline: the model-card builder is pure (dicts in, markdown out),
and the upload path is exercised with a **fake `HfApi`** so no network or token
is needed. Covers card content from synthetic + real summaries, frontmatter,
`write_card`, arg parsing, the `--dry-run` flow, and upload orchestration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import push_to_hub as p


@pytest.fixture
def comparison() -> dict:
    """Minimal three-way comparison summary with known numbers."""
    return {
        "n_contracts": 51,
        "models": {
            "naive": {
                "n_contracts": 51,
                "json_valid": 0,
                "json_validity_rate": 0.0,
                "per_field_match_rate_CAVEATED": {"overall_f1": 0.4069},
            },
            "strong_prompt": {
                "n_contracts": 51,
                "json_valid": 6,
                "json_validity_rate": 0.11764705882352941,
                "per_field_match_rate_CAVEATED": {"overall_f1": 0.4139},
            },
            "finetuned": {
                "n_contracts": 51,
                "json_valid": 49,
                "json_validity_rate": 0.9607843137254902,
                "per_field_match_rate_CAVEATED": {
                    "document_name": 0.8627,
                    "parties": 0.7741,
                    "agreement_date": 0.8824,
                    "effective_date": 0.6471,
                    "expiration_date": 0.4706,
                    "governing_law": 0.6863,
                    "renewal_term": 0.8039,
                    "notice_period_to_terminate_renewal": 0.8039,
                    "exclusivity": 0.6667,
                    "non_compete": 0.7451,
                    "cap_on_liability": 0.6667,
                    "uncapped_liability": 0.7451,
                    "overall_f1": 0.7295,
                },
            },
        },
    }


@pytest.fixture
def training() -> dict:
    return {
        "hardware": "1x NVIDIA A100 80GB PCIe",
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": 7,
            "trainable_params": 41943040,
            "total_params": 8072204288,
            "trainable_pct": 0.52,
        },
        "optimization": {
            "epochs": 3,
            "total_steps": 153,
            "gradient_accumulation_steps": 8,
            "effective_batch_size": 8,
            "optimizer": "adamw_8bit",
            "learning_rate": 0.0002,
            "lr_scheduler": "cosine",
            "precision": "bf16",
        },
        "results": {
            "best_eval_loss": 0.2127,
            "best_eval_step": 100,
            "final_train_loss_mean": 0.1767,
            "train_runtime_seconds": 3268,
        },
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def test_pct_rounds_to_whole_percent():
    assert p._pct(0.0) == "0%"
    assert p._pct(0.11764705882352941) == "12%"
    assert p._pct(0.9607843137254902) == "96%"


def test_load_json_roundtrip(tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"a": 1}))
    assert p.load_json(f) == {"a": 1}


# ---------------------------------------------------------------------------
# build_model_card
# ---------------------------------------------------------------------------


def test_card_has_frontmatter(comparison, training):
    card = p.build_model_card(comparison, training)
    assert card.startswith("---\n")
    assert "base_model: unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit" in card
    assert "license: mit" in card
    assert "library_name: peft" in card
    assert "theatticusproject/cuad-qa" in card


def test_card_has_headline_numbers(comparison, training):
    card = p.build_model_card(comparison, training)
    # Validity lift in the summary line and the three-way table.
    assert "0% / 12%" in card
    assert "96%" in card
    assert "0.7295" in card  # fine-tuned overall_f1
    assert "0.4069" in card  # naive overall_f1


def test_card_three_way_table_lists_all_models(comparison, training):
    card = p.build_model_card(comparison, training)
    assert "Naive baseline" in card
    assert "Strong-prompt baseline" in card
    assert "Fine-tuned (this adapter)" in card
    assert "49 / 51" in card


def test_card_per_field_table_has_all_12_fields(comparison, training):
    card = p.build_model_card(comparison, training)
    for field in p.FIELD_ORDER:
        assert f"`{field}`" in card


def test_card_embeds_exact_training_prompt(comparison, training):
    from training.prepare_dataset import SYSTEM_PROMPT

    card = p.build_model_card(comparison, training)
    # The card must carry the real training prompt so users don't drift.
    assert SYSTEM_PROMPT in card
    assert "apply_chat_template" in card


def test_card_has_training_hyperparameters(comparison, training):
    card = p.build_model_card(comparison, training)
    assert "41,943,040" in card  # trainable params, comma-grouped
    assert "0.52%" in card
    assert "0.2127" in card  # best eval_loss
    assert "~54 min" in card  # 3268s -> 54 min


def test_card_has_license_and_acknowledgments(comparison, training):
    card = p.build_model_card(comparison, training)
    assert "MIT" in card
    assert "CC BY 4.0" in card
    assert "Atticus" in card
    assert "hendrycks2021cuad" in card


def test_card_uses_repo_id_in_usage(comparison, training):
    card = p.build_model_card(comparison, training, repo_id="someuser/some-repo")
    assert 'PeftModel.from_pretrained(base, "someuser/some-repo")' in card


def test_card_builds_from_real_committed_summaries():
    """The real text-free summaries under data/results/ build a valid card."""
    comp_path = Path(p.DEFAULT_COMPARISON_SUMMARY)
    train_path = Path(p.DEFAULT_TRAINING_SUMMARY)
    if not (comp_path.exists() and train_path.exists()):
        pytest.skip("real summary files not present")
    card = p.build_model_card(p.load_json(comp_path), p.load_json(train_path))
    assert "96%" in card
    assert card.startswith("---\n")


# ---------------------------------------------------------------------------
# write_card
# ---------------------------------------------------------------------------


def test_write_card_creates_readme(tmp_path):
    out = p.write_card("# hello", tmp_path / "adapter")
    assert out == tmp_path / "adapter" / "README.md"
    assert out.read_text() == "# hello"


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------


def test_parse_args_defaults():
    args = p.parse_args([])
    assert args.repo_id == p.DEFAULT_REPO_ID
    assert args.adapter_path == p.DEFAULT_ADAPTER_PATH
    assert args.dry_run is False
    assert args.private is False


def test_parse_args_overrides():
    args = p.parse_args(["--repo-id", "u/r", "--adapter-path", "/tmp/a", "--dry-run", "--private"])
    assert args.repo_id == "u/r"
    assert args.adapter_path == "/tmp/a"
    assert args.dry_run is True
    assert args.private is True


# ---------------------------------------------------------------------------
# main() flows
# ---------------------------------------------------------------------------


def _write_summaries(tmp_path, comparison, training) -> tuple[Path, Path]:
    comp = tmp_path / "comparison_summary.json"
    train = tmp_path / "training_summary.json"
    comp.write_text(json.dumps(comparison))
    train.write_text(json.dumps(training))
    return comp, train


def test_main_dry_run_writes_card_no_network(tmp_path, comparison, training, monkeypatch):
    adapter = tmp_path / "final-adapter"
    adapter.mkdir()
    comp, train = _write_summaries(tmp_path, comparison, training)

    # If anything tried to upload, this would make it fail loudly.
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _boom_hfapi())

    rc = p.main(
        [
            "--dry-run",
            "--adapter-path", str(adapter),
            "--comparison-summary", str(comp),
            "--training-summary", str(train),
        ]
    )
    assert rc == 0
    assert (adapter / "README.md").read_text().startswith("---\n")


def test_main_missing_adapter_returns_1(tmp_path, comparison, training):
    comp, train = _write_summaries(tmp_path, comparison, training)
    rc = p.main(
        [
            "--adapter-path", str(tmp_path / "does-not-exist"),
            "--comparison-summary", str(comp),
            "--training-summary", str(train),
        ]
    )
    assert rc == 1


def test_main_no_token_returns_1(tmp_path, comparison, training, monkeypatch):
    adapter = tmp_path / "final-adapter"
    adapter.mkdir()
    comp, train = _write_summaries(tmp_path, comparison, training)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    rc = p.main(
        [
            "--adapter-path", str(adapter),
            "--comparison-summary", str(comp),
            "--training-summary", str(train),
        ]
    )
    assert rc == 1


# ---------------------------------------------------------------------------
# upload_to_hub orchestration (mocked HfApi)
# ---------------------------------------------------------------------------


class FakeHfApi:
    instances: list = []

    def __init__(self, token=None):
        self.token = token
        self.create_calls: list = []
        self.upload_calls: list = []
        FakeHfApi.instances.append(self)

    def create_repo(self, repo_id=None, repo_type=None, private=False, exist_ok=False):
        self.create_calls.append(
            {"repo_id": repo_id, "repo_type": repo_type, "private": private, "exist_ok": exist_ok}
        )

    def upload_folder(self, folder_path=None, repo_id=None, repo_type=None):
        self.upload_calls.append(
            {"folder_path": folder_path, "repo_id": repo_id, "repo_type": repo_type}
        )


def _boom_hfapi():
    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("HfApi must not be constructed during a dry run")

    return Boom


def test_upload_to_hub_orchestration(tmp_path, monkeypatch):
    import huggingface_hub

    FakeHfApi.instances = []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)

    adapter = tmp_path / "final-adapter"
    adapter.mkdir()
    url = p.upload_to_hub(adapter, "u/r", token="tok", private=True)

    assert url == "https://huggingface.co/u/r"
    api = FakeHfApi.instances[-1]
    assert api.token == "tok"
    assert api.create_calls[0] == {
        "repo_id": "u/r",
        "repo_type": "model",
        "private": True,
        "exist_ok": True,
    }
    assert api.upload_calls[0]["folder_path"] == str(adapter)
    assert api.upload_calls[0]["repo_id"] == "u/r"


def test_main_full_upload_with_mocked_api(tmp_path, comparison, training, monkeypatch):
    import huggingface_hub

    FakeHfApi.instances = []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setenv("HF_TOKEN", "tok-from-env")

    adapter = tmp_path / "final-adapter"
    adapter.mkdir()
    comp, train = _write_summaries(tmp_path, comparison, training)

    rc = p.main(
        [
            "--adapter-path", str(adapter),
            "--comparison-summary", str(comp),
            "--training-summary", str(train),
        ]
    )
    assert rc == 0
    # Card was written into the adapter dir and the folder was uploaded.
    assert (adapter / "README.md").exists()
    api = FakeHfApi.instances[-1]
    assert api.token == "tok-from-env"
    assert api.upload_calls[0]["folder_path"] == str(adapter)
