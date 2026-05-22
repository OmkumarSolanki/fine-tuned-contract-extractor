# Contract Extractor

A reproducible data pipeline and 12-field extraction schema for fine-tuning instruction-tuned LLMs on the [Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad).

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-135%20passing-green.svg)](#development)

> 510 CUAD contracts → 408/51/51 ChatML train/val/test splits, with Llama 3.1 chat-template-aware truncation, deterministic seeding, a 12-field Pydantic schema, a pure-Python metrics module, and two baseline evaluators (naive prompt + strong prompt). 135 unit tests cover the schema, metrics, pipeline helpers, and baseline evaluators.

---

## Table of contents

- [What this repo contains](#what-this-repo-contains)
- [Quick start](#quick-start)
- [The 12 fields](#the-12-fields)
- [Project structure](#project-structure)
- [The data pipeline at a glance](#the-data-pipeline-at-a-glance)
- [Metrics module](#metrics-module)
- [Tech stack](#tech-stack)
- [Documentation](#documentation)
- [Development](#development)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## What this repo contains

Three small, well-tested pieces of code:

1. **A typed schema** (`extractor/schemas.py`) — `ContractExtraction` (12 fields, canonical order), plus `ExtractRequest` / `ExtractResponse` shapes that round out the data contract for any HTTP serving layer that might consume it.
2. **A two-step data pipeline** (`training/`) — turns the public CUAD-QA dataset on Hugging Face into ChatML-formatted JSONL splits suitable for fine-tuning instruction-tuned LLMs.
3. **A pure-Python metrics module** (`evaluation/metrics.py`) — `is_valid_json`, `field_accuracy`, `parties_f1`, `overall_f1`. Locks down what "correct output" means so any model trained on this dataset can be scored against the same definitions.

Everything is covered by 135 unit tests that run in under a second on CPU, with no network or GPU required.

### Source data

The data product is **510 expert-annotated commercial contracts** from the [Contract Understanding Atticus Dataset (CUAD) v1](https://www.atticusprojectai.org/cuad), accessed through the public `theatticusproject/cuad-qa` HuggingFace mirror (CC BY 4.0). We pool CUAD's SQuAD-style train+test splits, regroup back to per-contract structured annotations under our 12-field schema, head+tail-truncate long contracts to an 8000-token budget, and deterministically split 80/10/10 into ChatML JSONL.

For the full breakdown — source format, mapping rules, real coverage statistics per field, known limitations, inspection recipes, and reproducibility guarantees — read [`docs/DATASET.md`](docs/DATASET.md).

---

## Quick start

```bash
# 1. Clone and set up
git clone <this-repo-url>
cd fine-tuned-contract-extractor
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. (Optional) Configure environment
cp .env.example .env
# Edit .env: set HF_TOKEN if you want to use the gated meta-llama tokenizer.
# Without it, the pipeline falls back to the public unsloth/Meta-Llama-3.1-8B-Instruct mirror.

# 3. Run the test suite (135 tests, no network or GPU required)
pytest tests/ -v

# 4. Smoke run (≈10 contracts, ~5 sec)
python training/ingest_cuad.py --limit 10 --force
python training/prepare_dataset.py
# → data/processed/{train,val,test}.jsonl with 8 / 1 / 1 rows

# 5. Full run (≈510 contracts, ~75 sec)
python training/ingest_cuad.py --force
python training/prepare_dataset.py
# → data/processed/{train,val,test}.jsonl with 408 / 51 / 51 rows
```

### Inspect a generated training example

```bash
$ head -1 data/processed/train.jsonl | jq '.contract_id'
"RaeSystemsInc_20001114_10-Q_EX-10.57_2631790_EX-10.57_Co-Branding Agreement"

$ head -1 data/processed/train.jsonl | jq '.messages | length'
3

$ head -1 data/processed/train.jsonl | jq '.messages[2].content' | head -c 200
"{\"document_name\":\"CO-BRANDING AGREEMENT (FORM)\",\"parties\":[\"Solutions Media...
```

Each row of the processed JSONL has a 3-message ChatML structure (`system` / `user` / `assistant`) plus a `contract_id` for traceability. The assistant turn is compact one-line JSON in canonical 12-field order.

---

## The 12 fields

`ContractExtraction` (defined in [`extractor/schemas.py`](extractor/schemas.py)) covers 12 commercially-critical clause categories drawn from CUAD's 41:

| Field | Type | Description |
|-------|------|-------------|
| `document_name` | str? | Title of the contract |
| `parties` | list[str] | All named parties to the contract |
| `agreement_date` | str? | Date the agreement was signed (ISO YYYY-MM-DD where possible) |
| `effective_date` | str? | Date the agreement becomes effective |
| `expiration_date` | str? | Date the agreement expires, if specified |
| `governing_law` | str? | Choice-of-law clause |
| `renewal_term` | str? | Auto-renewal mechanics, if any |
| `notice_period_to_terminate_renewal` | str? | Required notice to prevent automatic renewal |
| `exclusivity` | str? | Exclusivity restrictions (territorial, customer, product) |
| `non_compete` | str? | Non-compete restrictions |
| `cap_on_liability` | str? | Maximum monetary liability or formula |
| `uncapped_liability` | str? | Carve-outs from any cap on liability (e.g., gross negligence, willful misconduct, IP infringement, confidentiality breach) |

All non-list fields are `Optional[str]` — `None` when the contract doesn't address the topic. `parties` is `List[str]` — the empty list when no parties are named.

For the legal context behind each field and why these 12 were chosen out of CUAD's 41 categories, see [`docs/SCHEMA.md`](docs/SCHEMA.md).

---

## Project structure

```
fine-tuned-contract-extractor/
├── README.md                    # ◀── You are here
├── LICENSE                      # MIT
├── pyproject.toml               # Dependencies + package metadata
├── .env.example                 # Template for HF_TOKEN
├── .github/workflows/ci.yml     # Lint + test on push/PR
│
├── docs/                        # Public technical documentation
│   ├── ARCHITECTURE.md
│   ├── DATASET.md
│   ├── DATA_PIPELINE.md
│   ├── SCHEMA.md
│   ├── DEVELOPMENT.md
│   └── DECISIONS.md
│
├── extractor/                   # Pydantic models — the data contract
│   └── schemas.py
│
├── training/                    # Pipelines that produce the JSONL splits
│   ├── ingest_cuad.py           # CUAD-QA → cuad_parsed.jsonl
│   └── prepare_dataset.py       # cuad_parsed.jsonl → train/val/test splits
│
├── evaluation/                  # Scoring + baseline evaluators
│   ├── metrics.py               # is_valid_json, parties_f1, overall_f1, ...
│   ├── _runner.py               # Shared helpers + lazy-loaded model+generation
│   ├── eval_base.py             # Naive-prompt baseline
│   └── eval_prompt_baseline.py  # Strong-prompt baseline (schema + few-shot)
│
├── tests/                       # pytest — 135 tests across 6 files
│   ├── test_schemas.py                  # 13 tests
│   ├── test_metrics.py                  # 27 tests
│   ├── test_ingest_cuad.py              # 26 tests
│   ├── test_prepare_dataset.py          # 17 tests
│   ├── test_eval_base.py                # 26 tests
│   └── test_eval_prompt_baseline.py     # 26 tests
│
└── data/                        # Generated artifacts (gitignored)
    ├── raw/cuad_parsed.jsonl
    ├── processed/{train,val,test}.jsonl
    └── results/{base,prompt_baseline}_predictions.json   # produced by the baseline evaluators
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a per-module breakdown of responsibilities.

---

## The data pipeline at a glance

- **3-message ChatML rows.** Each example is a `{system, user, assistant}` triple. The system prompt is fixed; the user prompt wraps the (possibly truncated) contract text; the assistant turn is the gold extraction as compact one-line JSON.
- **Head + tail truncation.** Contracts longer than 8000 tokens are cut to the first 5000 + last 3000 tokens, joined by a literal `\n[...TRUNCATED...]\n` marker. The risk-allocation clauses we care about (governing law, uncapped liability, liability caps) tend to live at the end of contracts, so head-only truncation would systematically lose them.
- **Tokenizer fallback.** The pipeline tries the gated `meta-llama/Llama-3.1-8B-Instruct` tokenizer first (requires `HF_TOKEN`), then falls back to the public `unsloth/Meta-Llama-3.1-8B-Instruct` mirror. Both ship the same chat template.
- **Deterministic 80/10/10 split.** `random.Random(42).shuffle(...)` over the 510 contracts produces a 408 / 51 / 51 split, byte-identical across runs.
- **Validation gates.** Annotations are validated against the Pydantic schema both in `ingest_cuad.py` (before writing the parsed JSONL) and in `prepare_dataset.py` (before assembling the ChatML row). Any failure is logged and the row is dropped.

See [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) for the full mapping rules and edge cases.

---

## Metrics module

`evaluation/metrics.py` provides four pure-Python functions:

| Function | Returns | What it measures |
|----------|---------|------------------|
| `is_valid_json(prediction_str)` | `bool` | Does the model output parse as JSON and validate against `ContractExtraction`? |
| `field_accuracy(pred, gold, field)` | `float ∈ {0.0, 1.0}` | Case- and whitespace-insensitive equality on a single non-list field. |
| `parties_f1(pred, gold)` | `float ∈ [0.0, 1.0]` | Set-based F1 on the `parties` list (case-insensitive). |
| `overall_f1(predictions, golds)` | `dict` | Per-field mean scores plus an `overall_f1` aggregate. |

These are the contract for "correct output" — anything trained on this dataset can be scored against the same definitions.

---

## Tech stack

- **Python 3.11+**
- **[Pydantic v2](https://docs.pydantic.dev/)** — schema definition and validation
- **[HuggingFace Datasets](https://huggingface.co/docs/datasets) `<4.0`** — loading the CUAD-QA mirror (pinned because v4.x dropped script-based loaders; see [ADR-005](docs/DECISIONS.md))
- **[HuggingFace Transformers](https://huggingface.co/docs/transformers)** — Llama 3.1 chat tokenizer
- **[python-dateutil](https://dateutil.readthedocs.io/)** — fuzzy date parsing in CUAD ingestion
- **[python-dotenv](https://github.com/theskumar/python-dotenv)** — `.env` loading
- **[tqdm](https://tqdm.github.io/)** — ingest progress bar
- **[Jinja2](https://jinja.palletsprojects.com/)** — chat-template rendering at the tokenizer level
- **[pytest](https://docs.pytest.org/)** — test runner (135 tests today)
- **[ruff](https://docs.astral.sh/ruff/)** — linting (configured in `pyproject.toml`)

---

## Documentation

The repository ships with a full set of technical documents in [`docs/`](docs/). They cross-link to each other; pick whichever matches your goal.

| If you want to… | Read this |
|------------------|-----------|
| Understand how the system fits together | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Understand the dataset itself: source, mapping rules, coverage stats, limitations | [`docs/DATASET.md`](docs/DATASET.md) |
| Understand how the data pipeline works (deep dive) | [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) |
| Understand the 12 schema fields and their legal meaning | [`docs/SCHEMA.md`](docs/SCHEMA.md) |
| Set up a dev environment, run tests, contribute | [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) |
| Know why each architectural decision was made | [`docs/DECISIONS.md`](docs/DECISIONS.md) |

---

## Development

For detailed setup, testing, and contribution instructions, see [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

The short version:

```bash
# Setup
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests (no network or GPU required)
pytest tests/ -v

# Run the data pipeline
python training/ingest_cuad.py
python training/prepare_dataset.py
```

### Continuous integration

Every push and pull request runs the test suite via GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). CI does `pip install -e ".[dev]"` then `pytest -m "not network"`.

### Contributing

Pull requests are welcome — see [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for the checklist. Architectural decisions (schema, prompts, truncation rules, JSON shape, splits) are recorded in [`docs/DECISIONS.md`](docs/DECISIONS.md) — propose a change there alongside any code change that touches them.

---

## License

- **Code:** [MIT License](LICENSE) — © 2026 Om Solanki.
- **CUAD data:** [Creative Commons CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (used at runtime via the HuggingFace mirror; not redistributed in this repo).
- **Llama 3.1 tokenizer:** Subject to the [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) — relevant only if you choose to download the gated `meta-llama/Llama-3.1-8B-Instruct` tokenizer with an `HF_TOKEN`. The public `unsloth/Meta-Llama-3.1-8B-Instruct` mirror is the default fallback.

---

## Acknowledgments

- **[The Atticus Project](https://www.atticusprojectai.org/)** — for curating, annotating, and releasing the [Contract Understanding Atticus Dataset (CUAD)](https://www.atticusprojectai.org/cuad). 510 expert-annotated commercial contracts is a remarkable public-good contribution to legal NLP.
- **[Meta AI](https://ai.meta.com/)** — for the [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) tokenizer and chat template.
- **[Unsloth AI](https://unsloth.ai/)** — for the public `unsloth/Meta-Llama-3.1-8B-Instruct` mirror.
- **[Hugging Face](https://huggingface.co/)** — for the model and dataset hub, `transformers`, and `datasets`.
- **CUAD authors** — Dan Hendrycks, Collin Burns, Anya Chen, Spencer Ball — for [the paper](https://arxiv.org/abs/2103.06268) and the open-data ethic.

If you use CUAD in derivative work, a citation to the paper is the right way to give credit:

```bibtex
@article{hendrycks2021cuad,
  title   = {CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
  author  = {Dan Hendrycks and Collin Burns and Anya Chen and Spencer Ball},
  journal = {arXiv preprint arXiv:2103.06268},
  year    = {2021}
}
```
