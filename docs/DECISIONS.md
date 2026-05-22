# Architectural Decisions

This is a log of consequential decisions made during the project, with their rationale and the alternatives considered. The format is loosely based on Architecture Decision Records (ADRs) — short, dated, status-tracked.

These decisions are **load-bearing**: they shape the data pipeline and schema, and anything trained or evaluated on this dataset depends on them. Don't change them silently. If you want to revise one, open an issue, agree on the change, then update both the ADR and the code in the same PR.

The decisions below were captured from a Q&A session at project start (2026-05-20). A human (Om) made each call; the rationale is recorded so future maintainers and Claude/agent sessions know not to relitigate them.

---

## ADR-001 — Use CUAD as the dataset

- **Date:** 2026-05-20
- **Status:** Accepted

### Context

We need an expert-annotated dataset of commercial contracts large enough to support fine-tuning, with a permissive license for redistribution.

### Decision

Use [CUAD v1](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus Dataset), accessed via the [`theatticusproject/cuad-qa`](https://huggingface.co/datasets/theatticusproject/cuad-qa) HuggingFace mirror.

### Rationale

- 510 commercial contracts, 13,000+ labeled clauses.
- Annotated by trained law students with attorney review.
- 41 clause categories — broad enough to support diverse extraction schemas.
- CC BY 4.0 license — compatible with publishing a fine-tuned adapter.
- Already on HuggingFace, with a SQuAD-style schema that fits standard tooling.

### Alternatives considered

- **EDGAR contract scraping + custom annotation.** Way too expensive and slow to annotate at sufficient quality.
- **CUAD's SquadV2 split directly from the original GitHub repo.** Would work but loses the convenience of `datasets.load_dataset`.

### Consequences

- We pin `datasets>=3.0.0,<4.0.0` because the CUAD-QA mirror ships a Python loading script and `datasets` 4.x removed loading-script support. See ADR-005.

---

## ADR-002 — Pick 12 categories from CUAD's 41

- **Date:** 2026-05-20
- **Status:** Accepted

### Context

Training a model on all 41 CUAD categories produces a wide, sparse output schema. Many categories are rare or domain-specific.

### Decision

Train on 12 categories, chosen for commercial impact, annotation density, and inferential difficulty: `document_name`, `parties`, `agreement_date`, `effective_date`, `expiration_date`, `governing_law`, `renewal_term`, `notice_period_to_terminate_renewal`, `exclusivity`, `non_compete`, `cap_on_liability`, `uncapped_liability`.

### Rationale

See [`SCHEMA.md`](./SCHEMA.md) §1 for the full reasoning. The mix of "easy" (document_name, dates) and "hard" (renewal_term, uncapped_liability) fields gives downstream evaluation a richer story.

### Alternatives considered

- **Train on all 41.** Sparser per-field metrics, harder to interpret.
- **Train on fewer (5–8).** Less commercial coverage; not enough variety to stress-test fine-tuning vs. prompt engineering.

### Consequences

- The 12 fields are committed to in the schema. Future schema changes should be a versioned `ContractExtractionV2` to keep evaluation runs comparable.

---

## ADR-003 — Use Llama 3.1 8B Instruct as the base model

- **Date:** 2026-05-20
- **Status:** Accepted

### Context

We need an open-weights base model that's small enough to fine-tune on a single A100 but large enough to produce reliable structured output on long legal text.

### Decision

Fine-tune `meta-llama/Llama-3.1-8B-Instruct`.

### Rationale

- 8B is the QLoRA sweet spot — fits in 80GB with batch_size=1 + grad_accum=8 + bf16, trains in <2 hours.
- 128k context window — every CUAD contract fits without aggressive chunking. (We still truncate at 8000 tokens to control training memory; see ADR-008.)
- Llama 3.1 chat template is well-supported in `transformers`, Unsloth, and TRL.
- Permissive Llama 3.1 license allows commercial use with derivative-works terms compatible with publishing LoRA adapters.

### Alternatives considered

- **Llama 3.1 70B.** Requires multi-GPU; too expensive for a portfolio project.
- **Mistral 7B / Qwen 2.5 7B / Gemma 2 9B.** All credible alternatives. Llama wins on ecosystem maturity and the most-supported chat template.

### Consequences

- Tokenizer access is gated through the meta-llama HF org; we have a fallback path via the `unsloth/Meta-Llama-3.1-8B-Instruct` mirror. See ADR-006.

---

## ADR-005 — Pin `datasets>=3.0.0,<4.0.0`

- **Date:** 2026-05-20
- **Status:** Accepted

### Context

The `datasets` library bumped to 4.0 and removed support for Python loading scripts. The CUAD-QA mirror (`theatticusproject/cuad-qa`) uses a loading script (`cuad-qa.py`).

### Decision

Pin `datasets>=3.0.0,<4.0.0` in `pyproject.toml`. Document the rationale in the README.

### Rationale

This is the smallest, most-targeted fix. Switching to a different CUAD source (e.g., the original GitHub JSON) would be more durable but adds engineering effort. If/when the CUAD-QA mirror is converted to parquet, we can drop the upper bound.

### Alternatives considered

- **Direct download from CUAD GitHub.** More durable, but requires writing a custom loader and losing the `datasets` library's caching/streaming features.
- **Convert the mirror locally to parquet.** Bypasses the issue but means each developer has to do it.

### Consequences

- A bug was found during smoke-testing (a developer's `pip install` resolved `datasets` to 4.x) — pinning it in `pyproject.toml` prevents this.

---

## ADR-006 — Tokenizer fallback chain (gated → unsloth mirror)

- **Date:** 2026-05-20
- **Status:** Accepted

### Context

`meta-llama/Llama-3.1-8B-Instruct` is **gated** on HuggingFace — to download the tokenizer, you must accept the Llama 3.1 license and provide an `HF_TOKEN`. This is friction for new contributors and CI.

### Decision

In `training/prepare_dataset.py::load_tokenizer()`:

1. If `HF_TOKEN` is set in env, try `AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN)`.
2. On failure or if no token, fall back to `AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")`. The unsloth mirror is public and ships an identical chat template.
3. Log the source actually used at startup, so any drift is visible.
4. If both fail (no token, no internet), raise `RuntimeError` with a remediation message.

### Rationale

- Lets new contributors run the data pipeline without setting up an HF account.
- The unsloth mirror is byte-for-byte the same chat template and tokenizer, so there's no quality difference.
- Logging the source means any future divergence between the gated and mirror tokenizers would surface immediately.

### Alternatives considered

- **Require `HF_TOKEN`.** Higher friction; would block CI without an org-level secret.
- **Always use the unsloth mirror.** Slightly worse provenance — using the official source when possible is cleaner.

### Consequences

- The `HF_TOKEN` env var is optional throughout this repo; the unsloth mirror covers all current code paths.

---

## ADR-007 — Field mapping rules: longest-span and case-insensitive dedup

- **Date:** 2026-05-20
- **Status:** Accepted

### Context

CUAD often returns multiple text spans for a single (contract, category) pair: long legal clauses, party-name variants, etc. We have to pick what becomes the gold annotation.

### Decision

- For `parties` (list field): collect every non-empty span; deduplicate case-insensitively; preserve original casing of first occurrence. So `["Acme", "ACME", "Beta"]` → `["Acme", "Beta"]`.
- For dates: take the LONGEST non-empty span; `dateutil.parser.parse(s, fuzzy=True)`; emit ISO `YYYY-MM-DD` if a real date is parsed, else return the raw stripped string.
- For all other singular string fields: take the LONGEST non-empty span; or `None` if all spans are empty.

### Rationale

- **Longest-span** is the simplest defensible choice for narrative fields. The longest span tends to be the most informative (full clause, not partial reference).
- **First-occurrence casing** for parties preserves the contract's actual party-name presentation (often the legal name in caps + a short alias in mixed case — we want the legal name).
- **Fuzzy date parsing with raw fallback** handles the variety of date formats in real contracts (`"5/15/2018"`, `"May 15, 2018"`, `"as of the Effective Date"`) without losing the original span when parsing fails.

### Alternatives considered

- **Concatenate multi-span singular fields with `" | "`.** Higher recall but produces noisy targets and longer assistant outputs.
- **First-span instead of longest-span.** Simpler but often picks a partial reference rather than the full clause.
- **Strict ISO-only dates with no fallback.** Would `null` out CUAD annotations like `"as of the Effective Date"`, losing useful information.

### Consequences

- `aggregate_contract` in `training/ingest_cuad.py` is a small pure function with deterministic behavior. The rules are tested explicitly in `tests/test_ingest_cuad.py`.

---

## ADR-008 — Head + tail truncation at 5000 + 3000 tokens

- **Date:** 2026-05-20
- **Status:** Accepted

### Context

CUAD contracts range from ~1,000 to >100,000 tokens. Llama 3.1 8B can handle 128k context for inference, but training memory scales with sequence length. We need a truncation rule.

### Decision

In `training/prepare_dataset.py::truncate_text()`:

- If the contract is ≤ 8000 tokens, return it unchanged.
- Otherwise, keep the first 5000 tokens and the last 3000 tokens, joined by the literal marker `\n[...TRUNCATED...]\n`.

### Rationale

- The first ~3000 tokens of a contract typically contain the parties, dates, and document title — easy fields that head-only would handle fine.
- The last ~2000 tokens typically contain governing law, liability caps, and uncapped-liability carve-outs — exactly the fields we care about. Head-only truncation would systematically lose them.
- 5000 + 3000 = 8000 tokens. Adding the marker (~5 tokens) keeps total well under the training budget.

### Alternatives considered

- **Head-only truncation.** Simpler but loses end-of-contract clauses; per ADR-007's choice of high-impact fields, this would hurt the metrics that matter most.
- **Answer-span-aware truncation** using CUAD's `answer_start` offsets. Would tighten the heuristic but adds significant complexity. Could be a future improvement.
- **Different head/tail ratio (e.g., 7000 + 1000).** Empirically too head-heavy; 5000 + 3000 balances preamble vs. boilerplate.

### Consequences

- The marker is a stable string (`\n[...TRUNCATED...]\n`) that the model will see consistently. Any future inference code must apply the same truncation rule for consistency with training data.

---

## ADR-010 — Output JSONL has `messages` shape, not pre-rendered `text`

- **Date:** 2026-05-20
- **Status:** Accepted

### Context

`SFTTrainer` accepts both pre-rendered `text` and `messages` format. We have to pick one.

### Decision

Each line of `data/processed/{train,val,test}.jsonl` has the shape:

```json
{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}], "contract_id": "..."}
```

`SFTTrainer` will apply the chat template at training time. We render the template once at startup as a sanity check (logging a head/tail preview), but never write rendered text to the file.

### Rationale

1. **Debuggability.** `jq '.messages[2].content' train.jsonl | head -1` shows the assistant target directly.
2. **Loss masking.** TRL can mask everything except the assistant turn cleanly when given `messages`.
3. **Future-proofing.** If we swap base models, only the rendering step changes; the data file is reusable.

### Alternatives considered

- **Pre-render to `{"text": "<full chat template string>"}`.** Loses the role structure, makes loss masking harder, makes the file opaque.
- **Write both `text` and `messages`.** Wastes space, and one of them inevitably gets stale.

### Consequences

- The JSONL files can be loaded by any tool that understands chat-message format. The chat template is applied at consumption time, not at write time.

---

## ADR-011 — Compact one-line JSON for the assistant target

- **Date:** 2026-05-20
- **Status:** Accepted

### Context

The assistant turn is the model's training target. How we serialize it (compact vs. pretty-printed, fenced vs. raw) affects what the model learns to emit.

### Decision

`json.dumps(obj, ensure_ascii=False, separators=(",", ":"))` — compact one-line JSON, no whitespace after `,` or `:`, UTF-8 preserved. Keys re-ordered to match `ContractExtraction.model_fields` declaration order.

### Rationale

- **Compact saves ~30% tokens** vs. `indent=2`. Fewer tokens → faster training, lower inference latency, less context budget consumed.
- **Canonical key order** makes the training target byte-deterministic for equivalent inputs (avoids spurious gradient updates).
- **`ensure_ascii=False`** preserves non-ASCII characters (`España` stays readable, not `\u00cd...`).
- **No Markdown fences** (e.g., `\`\`\`json ... \`\`\``) — adds tokens, fragile to extract from output.

### Alternatives considered

- **Pretty-printed JSON (`indent=2`).** Easier for humans, but ~3x more tokens.
- **Markdown-fenced JSON.** Common in chat models, but adds parsing fragility.

### Consequences

- Inference code must expect compact JSON output. Any future prediction parser should `json.loads(...)` the output and not assume any whitespace.

---

## ADR-012 — Deterministic 80/10/10 split with seed=42

- **Date:** 2026-05-20
- **Status:** Accepted

### Context

We need a stable train/val/test split so:
- The same contracts go to test across runs (otherwise eval numbers aren't comparable).
- The training set is unaffected by re-running `prepare_dataset.py`.

### Decision

In `training/prepare_dataset.py::split_indices()`:

```python
indices = list(range(n))
random.Random(42).shuffle(indices)
n_train, n_val = int(0.8 * n), int(0.1 * n)
return indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]
```

Seed = 42. Any downstream training run should use the same seed for reproducibility.

### Rationale

- `random.Random(seed).shuffle` is the simplest deterministic shuffle in stdlib.
- Floor rounding on train and val sizes; remainder goes to test (so train and val are exactly 80% / 10%, and test absorbs any rounding).
- Matching seeds across the data split and training removes a subtle bug class (silently different randomness sources).

### Alternatives considered

- **Stratified split by contract length / by document type.** Would reduce variance in per-field metrics. Over-engineering for v1.
- **Multiple seeds, ensemble eval.** Better statistics but multiplies cost.

### Consequences

- For 510 contracts → 408 / 51 / 51. Tested explicitly in `tests/test_prepare_dataset.py::test_split_indices_handles_uneven_counts`.

---

## ADR-013 — Validate annotations before writing

- **Date:** 2026-05-20
- **Status:** Accepted

### Context

The data pipeline writes JSONL files that downstream code (training, evaluation) reads. We want to catch malformed data at the producer, not the consumer.

### Decision

Both `training/ingest_cuad.py` and `training/prepare_dataset.py` validate every row against `ContractExtraction.model_validate()` before writing. On validation failure:

- Log a single line: `Dropped {contract_id}: {one-line reason}` (max 200 chars; not the full Pydantic trace).
- Skip the row.

### Rationale

- Catches schema violations at the earliest possible point.
- Single-line logs are scannable and debuggable.
- Truncated reason avoids cluttering logs with multi-line tracebacks.

### Alternatives considered

- **Crash on first invalid row.** Worse for batch processing.
- **Write everything and validate downstream.** Defers the bug, harder to debug.

### Consequences

- In the smoke run, 0 rows were dropped — the pipeline is well-behaved on real CUAD data.
- The validation gate is the same one the model's output will pass through at eval time, ensuring training and eval contracts are aligned.

