# Development Guide

How to set up a development environment, run the test suite, run the data pipeline, and contribute to the project.

If you've never seen this codebase before, read [`ARCHITECTURE.md`](./ARCHITECTURE.md) first for the big picture, then come back here.

---

## 1. Prerequisites

- **Python 3.11+** (the project targets 3.11; 3.12, 3.13, 3.14 also work).
- **Git** for cloning and committing.
- **A Hugging Face account and access token** (free, optional) — only needed if you want to use the gated `meta-llama/Llama-3.1-8B-Instruct` tokenizer. If unset, the data pipeline automatically falls back to the public `unsloth/Meta-Llama-3.1-8B-Instruct` mirror.
- **No GPU is required.** All code in this repo is CPU-only.

---

## 2. Initial Setup

```bash
git clone <this-repo-url>
cd fine-tuned-contract-extractor

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package + dev dependencies
pip install -e ".[dev]"
```

### 2.1 macOS

All declared deps are pure-Python or have macOS arm64 wheels, so `pip install -e ".[dev]"` works out of the box on macOS.

### 2.2 Why `datasets<4.0`?

The CUAD-QA HuggingFace mirror (`theatticusproject/cuad-qa`) ships a Python loading script. `datasets` 4.x removed support for loading scripts. Until either (a) the mirror is converted to parquet or (b) we switch to a different CUAD source, we pin `datasets>=3.0.0,<4.0.0`. This is encoded in `pyproject.toml`.

### 2.3 Environment variables

```bash
cp .env.example .env
# Edit .env to set HF_TOKEN (optional — see below)
```

Recognized keys:

| Key | Used by | Purpose |
|-----|---------|---------|
| `HF_TOKEN` | `prepare_dataset.py` | Auth for the gated `meta-llama/Llama-3.1-8B-Instruct` tokenizer. Optional — if unset, the pipeline falls back to the public `unsloth/Meta-Llama-3.1-8B-Instruct` mirror. |

`.env` is gitignored. `.env.example` is a template; never put real secrets in it.

---

## 3. Running the Tests

```bash
pytest tests/ -v
```

You should see **254 passing tests** in well under one second.

### 3.1 Skipping network-dependent tests

We mark any test that requires internet access with `@pytest.mark.network`. None of the current tests need the network (helper-level only, with a fake tokenizer in `test_prepare_dataset.py`), but the marker is registered for future-proofing:

```bash
pytest tests/ -v -m "not network"      # skip network tests (default in CI)
pytest tests/ -v -m "network"          # run only network tests
```

### 3.2 Test layout

| File | Tests | What it covers |
|------|------:|----------------|
| `tests/test_schemas.py` | 13 | Pydantic models (`ContractExtraction`, `ExtractRequest`, `ExtractResponse`); field count + canonical order; validation behavior |
| `tests/test_metrics.py` | 27 | `is_valid_json`, `field_accuracy`, `parties_f1`, `overall_f1` — happy paths, edge cases, P/R imbalance |
| `tests/test_ingest_cuad.py` | 26 | All ingest helpers; both id forms (`__Cat_N` and `__Cat`); date parser; dedup; aggregation; text assembly |
| `tests/test_prepare_dataset.py` | 17 | `compact_json` ordering + UTF-8; `truncate_text` head+tail; `build_messages` validation; deterministic split |
| `tests/test_eval_base.py` | 26 | `_runner` pure helpers (parse, record shape, loaders, writer); naive prompt; `run_baseline` end-to-end (mocked model) |
| `tests/test_eval_prompt_baseline.py` | 26 | Schema-description rendering; deterministic few-shot selection (train-only); strong-prompt assembly |
| `tests/test_train.py` | 14 | Config loading (+failure modes); SFT-kwargs mapping + best-checkpoint guards; chat-template render flags; Llama 3.1 markers |
| `tests/test_eval_finetuned.py` | 12 | `load_test_messages` (system+user only, gold excluded); `run_finetuned` end-to-end (mocked `generate_chat`); CLI surface |
| `tests/test_compare.py` | 19 | `record_to_extraction` (valid/invalid→empty); `load_golds`; `score_model` (validity + F1, id alignment); `build_comparison`; table/CLI |
| `tests/test_api.py` | 19 | FastAPI endpoints with a mocked generator: `/health`, `/extract` (200, 422 missing/short, 503 no-model, 502 bad-output, 422-precedes-503), `/extract/stream` SSE, train/inference prompt parity, **API-key auth** (disabled-by-default, 401 missing/wrong key, 200 correct key, `/health` stays open) |
| `tests/test_observability.py` | 17 | Langfuse layer with a fake client: `RequestMetrics` math, no-op fallback when keys unset, process-wide singleton, v4/legacy SDK surfaces, error-swallowing, end-to-end traces |
| `tests/test_push_to_hub.py` | 19 | `build_model_card` from real summaries (dry-run, no network); `write_card`; `upload_to_hub` with a `FakeHfApi`; CLI/exit-code guards |
| `tests/test_scripts.py` | 19 | Phase 11 script helpers: `percentile` (interpolation, bounds, errors), `summarize` (latency/TTFT/tokens-per-sec), `resolve_adapter_path` precedence, `format_extraction` |

### 3.3 Adding a new test

- Put the test in the file matching its target module.
- If it doesn't need network or transformers, no marker is required.
- If it needs network (HuggingFace download, etc.), decorate with `@pytest.mark.network`.
- Use Pydantic v2's `pytest.raises(ValidationError)` for validation tests.
- Prefer property-style assertions (`assert x == 1.0`) over loose ones (`assert x > 0`).

---

## 4. Running the Data Pipeline

The pipeline has two scripts. They're idempotent and safe to re-run.

### 4.1 Quick smoke (~10 contracts, ~5 seconds)

```bash
python training/ingest_cuad.py --limit 10 --force
python training/prepare_dataset.py
```

Expected:
- `data/raw/cuad_parsed.jsonl` has 10 lines.
- `data/processed/{train,val,test}.jsonl` have 8 / 1 / 1 lines.

### 4.2 Full run (~510 contracts, ~75 seconds total)

```bash
python training/ingest_cuad.py
python training/prepare_dataset.py
```

Expected:
- `data/raw/cuad_parsed.jsonl` → ~510 lines.
- `data/processed/train.jsonl` → ~408 lines.
- `data/processed/val.jsonl` → ~51 lines.
- `data/processed/test.jsonl` → ~51 lines.

`ingest_cuad.py` is idempotent: by default it exits early if `data/raw/cuad_parsed.jsonl` already exists. Pass `--force` to overwrite.

### 4.3 Inspecting the output

```bash
# Top-level shape
$ wc -l data/processed/*.jsonl
   408 data/processed/train.jsonl
    51 data/processed/val.jsonl
    51 data/processed/test.jsonl

# Single row
$ head -1 data/processed/train.jsonl | jq '.contract_id'
$ head -1 data/processed/train.jsonl | jq '.messages | length'   # → 3
$ head -1 data/processed/train.jsonl | jq '.messages[2].content' | head -c 200

# How many fields are populated in the gold across the train set?
$ jq -c '.messages[2].content' data/processed/train.jsonl \
   | python -c "
import sys, json
counts = {}
for line in sys.stdin:
    obj = json.loads(json.loads(line))
    for k, v in obj.items():
        if k == 'parties':
            populated = bool(v)
        else:
            populated = v is not None
        counts[k] = counts.get(k, 0) + (1 if populated else 0)
total = sum(1 for _ in open('data/processed/train.jsonl'))
for k, n in sorted(counts.items()):
    print(f'{k:40s} {n:4d} / {total} ({100*n/total:.0f}%)')
"
```

---

## 5. Code Style

- **Line length: 100** (set in `pyproject.toml`).
- **Type hints:** required on public functions; prefer `from __future__ import annotations` so `dict[str, list[str]]` style works on older Python.
- **Docstrings:** required on public functions and classes. Module-level docstring required on every `.py` file in `extractor/`, `training/`, `evaluation/`.
- **Lints:** `ruff` is configured. Run `ruff check .` locally.

### 5.1 Conventions

- **Lazy imports for heavy dependencies.** `transformers` and `datasets` are imported inside the functions that need them, not at module top, so unit tests can import the source modules without pulling those packages.
- **Logging, not `print()`.** Every script uses `logging.basicConfig(level=INFO, format="%(asctime)s [%(levelname)s] %(message)s")` and `logger = logging.getLogger(__name__)`.
- **No emoji in source code.** Emoji are fine in markdown documentation but not in `.py` files.
- **Avoid global mutable state.** All scripts use `if __name__ == "__main__": sys.exit(main())` and pass arguments via `argparse`.

---

## 6. Project Structure Conventions

- **`extractor/`** is the typed data contract. Keep it lightweight — no heavy runtime deps.
- **`training/`** has scripts that produce artifacts (parsed data, ChatML splits). Heavier dependencies (e.g., `transformers`, `datasets`) are OK here.
- **`evaluation/`** has the pure-Python scoring functions.
- **`tests/`** mirror the source layout. One test file per source module.
- **`docs/`** has the public-facing technical documentation. Internal/personal notes go in `other/` (gitignored).

---

## 7. Common Tasks

### 7.1 Re-running the data pipeline after a CUAD update

```bash
# 1. Clear the HuggingFace cache for CUAD (forces a fresh download)
huggingface-cli delete-cache --pattern theatticusproject/cuad-qa

# 2. Re-run with --force
python training/ingest_cuad.py --force
python training/prepare_dataset.py

# 3. Verify
pytest tests/ -v
wc -l data/processed/*.jsonl
```

### 7.2 Switching to a different schema

If you want to add or remove fields in `ContractExtraction`:

1. Update `extractor/schemas.py`.
2. Update `EXPECTED_FIELDS` in `tests/test_schemas.py::test_field_count_and_order`.
3. Update `TARGET_CATEGORIES` (and `LIST_FIELDS` / `DATE_FIELDS` if applicable) in `training/ingest_cuad.py`.
4. Re-run the data pipeline.
5. Run the tests; they should all still pass.

See [`SCHEMA.md`](./SCHEMA.md) §6 for more on extending the schema.

### 7.3 Debugging a single contract

```python
import json
from extractor.schemas import ContractExtraction

target_id = "LIMEENERGYCO_..."
with open("data/raw/cuad_parsed.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        if rec["contract_id"] == target_id:
            extraction = ContractExtraction.model_validate(rec["annotations"])
            print(extraction.model_dump_json(indent=2))
            break
```

### 7.4 Generating a smaller subset for quick experimentation

```bash
# Extract the first 20 contracts as a "tiny" test set
python training/ingest_cuad.py --limit 20 --force --output data/raw/cuad_tiny.jsonl
python training/prepare_dataset.py --input data/raw/cuad_tiny.jsonl --output-dir data/processed_tiny
```

---

## 8. Continuous Integration

The CI workflow (`.github/workflows/ci.yml`) runs three jobs on every push to `main` and on pull requests:

1. **lint** — `ruff check .`
2. **test** — set up Python 3.11, `pip install -e ".[dev]"`, then `pytest tests/ -v -m "not network"`
3. **docker-build** — build the serving image with Buildx (GHA layer cache) to verify the `Dockerfile`; the container is not run (the model is GPU-only).

---

## 9. Contributing

This is currently a single-author portfolio project, but contributions are welcome — especially:

- **Schema improvements** — argued field additions / removals based on real M&A diligence experience.
- **Mapping rule refinements** — if you have a better idea for handling multi-span fields than "longest non-empty span", open an issue with examples.
- **Evaluation methodology** — particularly around the narrative fields where strict equality is harsh.
- **Documentation** — typos, unclear sections, missing context.

### 9.1 Pull-request checklist

Before opening a PR:

- [ ] Tests pass (`pytest tests/ -v`).
- [ ] New code has tests.
- [ ] Lints clean (`ruff check .`).
- [ ] Docs updated if behavior changed.

### 9.2 Decision-making

Architectural decisions that affect data shape (schema, prompts, truncation rule, JSON serialization, split logic) are *load-bearing* — anything trained or evaluated on this dataset depends on them. They are documented in [`DECISIONS.md`](./DECISIONS.md). Don't change them silently. Open an issue, agree on the change, then update the decision record alongside the code.

---

## 10. Where to Go Next

- **Want the conceptual overview?** [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- **Want to understand the 12 fields?** [`SCHEMA.md`](./SCHEMA.md)
- **Want to know how the pipeline works internally?** [`DATA_PIPELINE.md`](./DATA_PIPELINE.md)
- **Want to know the locked architectural decisions?** [`DECISIONS.md`](./DECISIONS.md)
