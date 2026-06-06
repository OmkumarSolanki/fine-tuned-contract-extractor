# Architecture

This document describes the system architecture of the Contract Extractor: how data flows through the project, how the modules are organized, and how each piece relates to the others.

> If you only read one architecture document, read this one. The other docs in this folder go deeper into specific subsystems.

---

## 1. The Big Picture

The repo is a **data pipeline plus a typed schema plus a metrics module**. It turns the public CUAD-QA dataset into ChatML-formatted JSONL splits suitable for fine-tuning instruction-tuned LLMs on the 12-field contract-extraction task it defines.

```
┌──────────────────────────┐
│         CUAD-QA          │  HuggingFace dataset (theatticusproject/cuad-qa)
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ training/ingest_cuad.py  │  Pool train+test → group by title → recover
│                          │  per-contract structured annotations.
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ data/raw/cuad_parsed.jsonl │  510 contracts, 1 row per contract
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ training/prepare_dataset.py │ Validate against schema → head+tail truncate →
│                          │  build ChatML messages → deterministic 80/10/10.
└────────────┬─────────────┘
             │
             ▼
┌────────────────────────────────────────────┐
│ data/processed/{train,val,test}.jsonl      │ 408 / 51 / 51 ChatML rows
└────────────────────────────────────────────┘
```

Two pure-Python modules sit alongside the pipeline and are consumed by it and by the test suite:

- **`extractor/schemas.py`** — defines `ContractExtraction` (12 fields), `ExtractRequest`, `ExtractResponse`. Used by both `ingest_cuad.py` and `prepare_dataset.py` to validate annotations before writing.
- **`evaluation/metrics.py`** — defines `is_valid_json`, `field_accuracy`, `parties_f1`, `overall_f1`. Pure functions, fully unit-tested, ready for use by anyone who wants to score predictions against the test split.

---

## 2. Repository Layout

```
fine-tuned-contract-extractor/
├── README.md                    # Project entry point
├── LICENSE                      # MIT
├── pyproject.toml               # Python deps + metadata
├── .env.example                 # Template for secrets
├── .gitignore                   # Includes data/**, .env, other/
├── .github/workflows/ci.yml     # Lint + test on push/PR
│
├── docs/                        # ◀── You are here
│   ├── ARCHITECTURE.md          # This file
│   ├── DATASET.md               # The dataset itself: source, mapping rules, coverage, limits
│   ├── DATA_PIPELINE.md         # Deep-dive on ingest + prepare
│   ├── SCHEMA.md                # The 12 fields, with legal context
│   ├── DEVELOPMENT.md           # Dev setup, testing, contributing
│   └── DECISIONS.md             # Locked architectural decisions
│
├── extractor/                   # Pydantic models — the data contract
│   └── schemas.py
│
├── training/                    # Data pipeline + QLoRA fine-tuning
│   ├── ingest_cuad.py           # CUAD-QA → cuad_parsed.jsonl
│   ├── prepare_dataset.py       # cuad_parsed.jsonl → train/val/test
│   ├── train.py                 # QLoRA fine-tuning driver (Unsloth + TRL)
│   └── configs/llama_8b_qlora.yaml
│
├── evaluation/                  # Scoring + baseline evaluators
│   ├── metrics.py               # is_valid_json, parties_f1, overall_f1, ...
│   ├── _runner.py               # Shared helpers + lazy-loaded model+generation
│   ├── eval_base.py             # Naive-prompt baseline
│   └── eval_prompt_baseline.py  # Strong-prompt baseline (schema + few-shot)
│
├── tests/                       # pytest suite (149 tests)
│   ├── test_schemas.py
│   ├── test_metrics.py
│   ├── test_ingest_cuad.py
│   ├── test_prepare_dataset.py
│   ├── test_eval_base.py
│   ├── test_eval_prompt_baseline.py
│   └── test_train.py
│
└── data/                        # Generated artifacts (gitignored)
    ├── raw/cuad_parsed.jsonl
    └── processed/{train,val,test}.jsonl
```

The split between `extractor/`, `training/`, and `evaluation/` is intentional: `extractor/` holds the typed data contract that's safe to depend on; `training/` holds heavier pipeline scripts; `evaluation/` holds pure scoring functions that can be reused without pulling pipeline dependencies.

---

## 3. Data Flow — In Detail

```
HuggingFace Hub
theatticusproject/cuad-qa
   │
   │  load_dataset(..., trust_remote_code=True)
   │  ├─ train: 22,450 rows
   │  └─ test:   4,182 rows
   ▼
training/ingest_cuad.py
   │
   │  • Pool train + test → 26,632 rows
   │  • Group by `title` → 510 unique contracts
   │  • For each contract:
   │      - extract_category_from_id(row["id"])
   │      - aggregate spans by 12-field rules:
   │          parties        → dedupe, case-insensitive
   │          dates          → longest span → dateutil.parser.parse(fuzzy=True)
   │          singular text  → longest non-empty span
   │      - assemble_contract_text() — dedupe contexts per chunk index
   │      - validate against ContractExtraction
   ▼
data/raw/cuad_parsed.jsonl (510 lines)
   │
   │  Each line: {"contract_id", "contract_text", "annotations"}
   ▼
training/prepare_dataset.py
   │
   │  • load_tokenizer():
   │      try meta-llama/Llama-3.1-8B-Instruct (gated, with HF_TOKEN)
   │      else fall back to unsloth/Meta-Llama-3.1-8B-Instruct
   │  • For each row:
   │      - truncate_text() if > 8000 tokens (head 5000 + tail 3000)
   │      - validate annotations → ContractExtraction
   │      - compact_json() of the 12 fields in canonical order
   │      - build_messages(): [system, user, assistant]
   │  • Sanity-check: render via tokenizer.apply_chat_template
   │  • Deterministic shuffle (seed=42), 80/10/10 split
   ▼
data/processed/
├── train.jsonl  (408 rows)
├── val.jsonl    ( 51 rows)
└── test.jsonl   ( 51 rows)
```

Each row of the processed JSONL has the shape:

```json
{
  "messages": [
    {"role": "system",    "content": "You are a legal contract analyst. Extract structured clauses from contracts."},
    {"role": "user",      "content": "Extract structured clauses from this contract:\n\n<contract text, possibly truncated>"},
    {"role": "assistant", "content": "<compact JSON of 12 fields>"}
  ],
  "contract_id": "<title from CUAD>"
}
```

We deliberately do not pre-render the chat template into a `text` field: keeping `messages` lets a downstream trainer (e.g., TRL's `SFTTrainer`) apply the template at training time and supports clean assistant-only loss masking. The tokenizer is loaded once at startup just to log a head/tail preview of the rendered template, so any drift is visible.

See [`DATA_PIPELINE.md`](./DATA_PIPELINE.md) for the full mapping rules and edge cases.

---

## 4. Module Responsibilities

### `extractor/`

The typed data contract.

- **`schemas.py`** — defines `ContractExtraction` (12 fields, declaration order is canonical and load-bearing), `ExtractRequest` (validates `contract_text` is at least 50 chars), `ExtractResponse` (extraction + timing metadata). Both `ingest_cuad.py` and `prepare_dataset.py` validate against `ContractExtraction` before writing JSONL.

### `training/`

Pipelines that produce the JSONL splits.

- **`ingest_cuad.py`** — pure-Python CUAD-QA → JSONL converter. Idempotent (`--force` to overwrite). Helper functions are pure and unit-tested without network.
- **`prepare_dataset.py`** — formats parsed CUAD into ChatML messages, applies the Llama 3.1 chat template (sanity preview only), splits 80/10/10 deterministically.

### `evaluation/`

Pure-Python scoring functions.

- **`metrics.py`** — `is_valid_json`, `field_accuracy`, `parties_f1`, `overall_f1`. All pure, fully unit-tested. Defines what "correct output" means for any model trained against this dataset.

### `tests/`

`pytest`-based suite (149 tests). Tests are split per-file mirroring the source layout. Helper-level tests do not require network or transformers — `transformers` is imported lazily inside `prepare_dataset.load_tokenizer` (and inside `evaluation/_runner.load_model`), and both the data-pipeline tests and the baseline evaluator tests run on CPU with a mocked or whitespace tokenizer. The training driver (`training/train.py`) keeps `unsloth`/`trl`/`torch` behind lazy imports too, so its config/mapping helpers are tested on CPU with no GPU stack.

See [`DEVELOPMENT.md`](./DEVELOPMENT.md) for the test breakdown and how to run them.

---

## 5. Why These Choices?

### 5.1 Why CUAD?

CUAD is the only large, high-quality, expert-annotated public dataset for contract review. 510 commercial contracts, 13,000+ labels across 41 categories, annotated by trained law students with attorney review. It's also Creative Commons (CC BY 4.0), which keeps downstream redistribution clean.

We use 12 of the 41 categories — the ones with broadest commercial relevance (parties, dates, governing law, uncapped liability, liability caps, etc.). See [`SCHEMA.md`](./SCHEMA.md) for the rationale on each.

### 5.2 Why the Llama 3.1 chat tokenizer?

The pipeline needs a chat template to render the system / user / assistant turns into the exact input format an instruction-tuned model expects. Llama 3.1's chat template is well-documented and supported by `transformers`. The official `meta-llama/Llama-3.1-8B-Instruct` tokenizer is gated, so the pipeline falls back to the public `unsloth/Meta-Llama-3.1-8B-Instruct` mirror, which ships an identical chat template. See [`DECISIONS.md`](./DECISIONS.md) ADR-006.

### 5.3 Why ChatML / `messages` instead of pre-rendered text?

Three reasons:

1. **Debuggability.** You can `jq '.messages[2].content'` and see the assistant target without parsing chat-template tokens.
2. **Loss masking.** A downstream trainer can mask everything except the assistant turn cleanly when given `messages`-style inputs.
3. **Future-proofing.** If you ever swap the chat template (e.g., to test against a different base model), the data file is reusable; only the rendering step changes.

### 5.4 Why head + tail truncation, and why 5000 + 3000?

Long contracts are not uniform in information density. The first ~3000 tokens contain the parties, dates, and document title. The **last** ~2000 tokens often contain governing law, liability caps, and uncapped-liability carve-outs — exactly the fields we care about. Head-only truncation would lose those.

Empirically (eyeballing CUAD), 5000 + 3000 keeps virtually all of the 12 categories visible while staying under the 8000-token training budget. We skip truncation entirely under 8000 tokens to avoid introducing the marker artifact when it isn't needed.

See [`DATA_PIPELINE.md`](./DATA_PIPELINE.md) §4 for the truncation algorithm.

---

## 6. Cross-Cutting Concerns

### 6.1 Determinism

Every randomness source has a fixed seed:

- 80/10/10 split: `random.Random(42).shuffle(indices)` in `prepare_dataset.py`.
- Title ordering in `ingest_cuad.py`: `sorted(titles)` before optional `--limit`.

Two runs of the data pipeline against the same CUAD-QA snapshot produce byte-identical splits.

### 6.2 Idempotency

- **`ingest_cuad.py`** — exits early if the output exists, with a log line. `--force` to overwrite. The HuggingFace download itself is cached by the `datasets` library.
- **`prepare_dataset.py`** — overwrites every time (cheap to re-run; ~60 sec for 510 contracts).

### 6.3 Validation gates

There are two validation gates between raw CUAD and the training data:

1. In **`ingest_cuad.py`**, `aggregate_contract()`'s output is validated against `ContractExtraction.model_validate()` before writing. Anything that fails is logged with a one-line reason and dropped.
2. In **`prepare_dataset.py`**, the same validation runs again on the input. Anything that fails (which would only be hand-edited bad data at this point) is logged with `Dropped {contract_id}: {reason}` and skipped.

In the smoke run, both gates pass cleanly — 510 in, 510 out, 0 dropped.

### 6.4 No secrets in code

- `HF_TOKEN` is read from `.env` via `python-dotenv`.
- `.env` is gitignored; `.env.example` is committed as a template.
- The repo never logs the value of any secret.

---

## 7. Where to Read More

The remaining docs in this folder go deeper: [`DATA_PIPELINE.md`](./DATA_PIPELINE.md) for the pipeline internals, [`SCHEMA.md`](./SCHEMA.md) for the 12 fields, [`DEVELOPMENT.md`](./DEVELOPMENT.md) for setup and contributing, [`DECISIONS.md`](./DECISIONS.md) for the locked architectural decisions.
