# The Dataset

This document is the dedicated reference for **the data itself** — what it is, where it comes from, how we reorganize it, what its real coverage looks like, and what its limits are. It complements [`DATA_PIPELINE.md`](./DATA_PIPELINE.md) (which is the engineering deep-dive on the *code* that produces the splits) and [`SCHEMA.md`](./SCHEMA.md) (which explains the legal meaning of each of the 12 extracted fields).

Read this first if you want to understand **the data product** this repo produces. Read `DATA_PIPELINE.md` if you want to understand the **scripts** that produce it.

---

## TL;DR

| Question | Answer |
|---|---|
| **What data?** | 510 expert-annotated commercial contracts from the Contract Understanding Atticus Dataset (CUAD v1), reorganized into a 12-field structured-extraction schema. |
| **From where?** | The public `theatticusproject/cuad-qa` HuggingFace mirror of CUAD v1. CC BY 4.0 licensed. |
| **What do we do?** | Pool the SQuAD-style rows back into per-contract annotations, validate against a Pydantic schema, head+tail-truncate long contracts, format as ChatML 3-message conversations, deterministically split 80/10/10. |
| **What you get** | `data/processed/{train,val,test}.jsonl` — 408 / 51 / 51 ChatML rows. Each is a `{system, user, assistant}` triple plus a `contract_id`. The assistant turn is compact one-line JSON in canonical 12-field order. |

---

## Table of contents

- [1. About CUAD](#1-about-cuad)
- [2. The CUAD-QA mirror on HuggingFace](#2-the-cuad-qa-mirror-on-huggingface)
- [3. From CUAD-QA rows to per-contract annotations](#3-from-cuad-qa-rows-to-per-contract-annotations)
- [4. The 12 chosen fields and the 29 we leave alone](#4-the-12-chosen-fields-and-the-29-we-leave-alone)
- [5. Mapping rules: CUAD spans → structured fields](#5-mapping-rules-cuad-spans--structured-fields)
- [6. Final output: ChatML JSONL splits](#6-final-output-chatml-jsonl-splits)
- [7. Coverage statistics (real numbers)](#7-coverage-statistics-real-numbers)
- [8. Known limitations](#8-known-limitations)
- [9. How to inspect and verify](#9-how-to-inspect-and-verify)
- [10. Reproducibility and provenance](#10-reproducibility-and-provenance)
- [11. Legal disclaimer](#11-legal-disclaimer)

---

## 1. About CUAD

The **Contract Understanding Atticus Dataset (CUAD) v1** is a corpus of **510 commercial legal contracts** with **13,000+ labeled clauses** across **41 clause categories**. It was released by the [Atticus Project](https://www.atticusprojectai.org/) in 2021 alongside the paper *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review* (Hendrycks, Burns, Chen, Ball — [arXiv:2103.06268](https://arxiv.org/abs/2103.06268)).

### 1.1 Why CUAD is unusual

Most public legal NLP datasets are either small, machine-generated, or stripped of meaningful annotations. CUAD is the rare combination of all three desirable properties at once:

1. **Expertly annotated.** Each contract was labeled by trained law students under the supervision of practicing attorneys. The 41 categories were chosen by attorneys who do contract review professionally — not by NLP researchers picking what looked tractable.
2. **At scale that supports fine-tuning.** 510 contracts is small for general NLP but large for expert-annotated legal data. The total ~13,000 labels distribute across 41 categories, giving most categories enough density to learn from.
3. **Permissively licensed.** CC BY 4.0 — so derivative datasets, fine-tuned models, and downstream tools can be redistributed.

The contracts themselves are real, publicly-filed commercial contracts from EDGAR (the SEC's filings system). Most are M&A-adjacent: licensing, distribution, joint ventures, supply, manufacturing, service agreements. Document types skew toward agreements that public companies file as material exhibits.

### 1.2 What CUAD is *not*

CUAD is not a balanced or general-purpose contract dataset. The corpus is biased in specific ways that matter for any model trained on it:

- **English only.** All 510 contracts are in US English.
- **US-law-centric.** Most contracts have a US state or federal choice-of-law clause. International contracts are present but rare.
- **Public-company biased.** Because the source is EDGAR, the contracts come from public filers and their counterparties. Small-business and consumer contracts are not represented.
- **Drafting-quality biased.** EDGAR-filed contracts are heavily lawyered, professionally drafted, and have boilerplate organized in predictable places. Pasted-together ad-hoc contracts (the kind extractors actually struggle with in practice) are not in the corpus.
- **Domain biased.** M&A, licensing, and commercial arrangements dominate. Employment contracts, consumer ToS, NDAs as standalone instruments, and bank-loan documents are underrepresented or absent.

A model trained only on CUAD will be very good at extracting clauses from EDGAR-style commercial contracts and **may degrade** on other contract types. We document this explicitly so anyone using or extending the dataset understands its scope.

### 1.3 License and citation

- **License.** CUAD is released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Derivative works (including this repo's processed splits) inherit the obligation to credit CUAD.
- **Citation.** Use the BibTeX entry in the repo's main [README](../README.md#acknowledgments) or directly:

  ```bibtex
  @article{hendrycks2021cuad,
    title   = {CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
    author  = {Dan Hendrycks and Collin Burns and Anya Chen and Spencer Ball},
    journal = {arXiv preprint arXiv:2103.06268},
    year    = {2021}
  }
  ```

- **What this repo redistributes.** Nothing of CUAD itself. The pipeline downloads CUAD-QA at runtime from HuggingFace and produces `data/raw/cuad_parsed.jsonl` and `data/processed/{train,val,test}.jsonl` locally. Those files are gitignored. Users regenerate them by running the pipeline.

---

## 2. The CUAD-QA mirror on HuggingFace

CUAD ships in two public forms: the original GitHub repository (`master_clauses.csv` + the contract texts) and a HuggingFace mirror that reformats it as a SQuAD-style extractive QA dataset.

We use the HuggingFace mirror — **[`theatticusproject/cuad-qa`](https://huggingface.co/datasets/theatticusproject/cuad-qa)** — because it integrates cleanly with `datasets.load_dataset` and ships the same underlying annotations, just in a per-(passage, question) row format.

### 2.1 Row schema

Every row of CUAD-QA has the following SQuAD-style shape:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Identifier of the row, encoding the contract title, the clause category, and (for chunked contracts) a chunk index. Format: `<contract_title>__<Category>_<chunk_index>` or `<contract_title>__<Category>` for short contracts. |
| `title` | `str` | Contract identifier — same value across all rows belonging to one contract. |
| `context` | `str` | A passage of the contract. Long contracts are split across multiple passages; this field is the passage text. |
| `question` | `str` | A natural-language question paraphrasing the clause category, e.g., *"Highlight the parts (if any) of this contract related to 'Document Name'..."*. |
| `answers.text` | `List[str]` | Zero or more verbatim text spans from `context` that answer the question. Empty list = "the question doesn't apply to this passage". |
| `answers.answer_start` | `List[int]` | Character offsets of those spans within `context`. |

### 2.2 Splits and row counts

| Split | Rows |
|-------|-----:|
| `train` | 22,450 |
| `test` | 4,182 |
| **Total** | **26,632** |

The `train`/`test` split that ships with CUAD-QA was designed for QA fine-tuning, not for contract-level evaluation. **We pool both splits** before regrouping into 510 per-contract annotations — see §3.

### 2.3 Why ~26,632 rows for 510 contracts?

CUAD has 41 categories and 510 contracts, which would give 510 × 41 = 20,910 (contract, category) pairs minimum. The actual row count is higher because **long contracts are chunked into multiple passages**, and each (chunk, question) pair becomes its own row. A contract that fits in 3 passages contributes 3 × 41 = 123 rows, not 41.

Average rows per contract: 26,632 / 510 ≈ **52 rows per contract**.

The chunk index is encoded in the row `id` suffix:

```
LIMEENERGYCO_..._DISTRIBUTOR AGREEMENT__Document Name_0
                                      ↑                ↑
                                   category        chunk index
```

For short contracts that fit in one passage, the chunk index is omitted entirely:

```
ACCELERATED..._JOINT VENTURE AGREEMENT__Document Name
                                      ↑
                                  category (no chunk suffix)
```

Both forms are handled by the regex in `training/ingest_cuad.py`.

### 2.4 Why we pin `datasets<4.0`

The CUAD-QA mirror ships a Python loading script (`cuad-qa.py`). The `datasets` library bumped to v4.0 and removed support for script-based loaders. Pinning `datasets>=3.0.0,<4.0.0` is the smallest stable fix until the mirror is converted to parquet. See [ADR-005](./DECISIONS.md#adr-005--pin-datasets300400) for the full rationale.

### 2.5 Local cache

The first run of `python training/ingest_cuad.py` downloads the dataset (~200 MB compressed; ~1.7 GB uncompressed) into `~/.cache/huggingface/datasets/`. Subsequent runs hit the cache — they're fast (~5 sec) and offline-safe.

---

## 3. From CUAD-QA rows to per-contract annotations

This is the conceptual transformation `training/ingest_cuad.py` performs. The output is `data/raw/cuad_parsed.jsonl` — one JSON object per contract, conforming to the `ContractExtraction` schema.

### 3.1 The pipeline in five steps

```
HuggingFace CUAD-QA
   ├─ train (22,450 rows)
   └─ test  (4,182 rows)
       │
       ▼  [1] Pool train+test
   26,632 (chunk, category) rows
       │
       ▼  [2] Group by `title`
   510 contracts × ~52 rows each
       │
       ▼  [3] For each contract: extract category from `id`,
       │     collect all answer spans for that category
       │     across all chunks
       │
       ▼  [4] Aggregate spans per the field-mapping rules
       │     (see §5: dedup, longest-span, fuzzy date parse)
       │
       ▼  [5] Validate against `ContractExtraction`;
       │     drop rows that fail validation (logged)
       │
   data/raw/cuad_parsed.jsonl
   (510 lines, one JSON object per contract)
```

### 3.2 Why pool train + test

The CUAD-QA `train`/`test` split is a *QA-task* split: rows from the same contract appear in both splits. If we kept it, our 510 contracts would have inconsistent annotation coverage (some chunks in `train`, others in `test`), breaking the per-contract aggregation step. Pooling first, then re-splitting at the contract level (§6.5), is the only way to get clean contract-level evaluation.

### 3.3 Concatenation order

When a contract is split into N chunks, we collect the contract's body text by concatenating the `context` fields of each chunk **in chunk-index order**. This preserves the original document order — the parties appear at the start, the signature page at the end. The same logic is applied to the answer spans: spans from chunk 0 come before spans from chunk 1, etc.

This matters for the head+tail truncation rule (§6.3): we want the *first* tokens of the document to actually be the document's preamble, and the *last* tokens to be the signature/risk-allocation block.

### 3.4 What's in `data/raw/cuad_parsed.jsonl`

One JSON object per line. Schema:

```json
{
  "contract_id": "<the CUAD `title` field>",
  "contract_text": "<concatenated chunk contexts in order>",
  "document_name": "<extracted name or null>",
  "parties": ["<deduped party 1>", "<deduped party 2>"],
  "agreement_date": "<ISO YYYY-MM-DD or raw fallback or null>",
  "effective_date": "...",
  "expiration_date": "...",
  "governing_law": "...",
  "renewal_term": "...",
  "notice_period_to_terminate_renewal": "...",
  "exclusivity": "...",
  "non_compete": "...",
  "cap_on_liability": "...",
  "uncapped_liability": "..."
}
```

This file is the **canonical structured representation** of CUAD for this project. Everything downstream — `prepare_dataset.py`, the splits, the eventual model training — derives from it.

---

## 4. The 12 chosen fields and the 29 we leave alone

CUAD has 41 categories. We extract 12. This is a deliberate scope decision documented in [ADR-002](./DECISIONS.md#adr-002--pick-12-categories-from-cuads-41).

### 4.1 The 12 we use

| Field (this repo) | CUAD category | Brief |
|---|---|---|
| `document_name` | `Document Name` | Title of the contract |
| `parties` | `Parties` | All named parties |
| `agreement_date` | `Agreement Date` | Date of signing |
| `effective_date` | `Effective Date` | When the agreement starts |
| `expiration_date` | `Expiration Date` | When the agreement ends, if specified |
| `governing_law` | `Governing Law` | Choice-of-law clause |
| `renewal_term` | `Renewal Term` | Auto-renewal mechanics |
| `notice_period_to_terminate_renewal` | `Notice Period To Terminate Renewal` | Required termination notice |
| `exclusivity` | `Exclusivity` | Territorial / customer / product exclusivity |
| `non_compete` | `Non-Compete` | Post-termination competition restrictions |
| `cap_on_liability` | `Cap On Liability` | Maximum monetary liability |
| `uncapped_liability` | `Uncapped Liability` | Carve-outs from any liability cap |

For the legal meaning of each field and the rationale for picking these 12 specifically, see [`SCHEMA.md`](./SCHEMA.md).

### 4.2 The 29 we leave alone (and why they exist)

These 29 CUAD categories are not extracted by this pipeline. Listing them explicitly so anyone extending the schema knows what's available:

```
Affiliate License-Licensee     Insurance                       Post-Termination Services
Affiliate License-Licensor     IP Ownership Assignment         Price Restrictions
Anti-Assignment                Irrevocable Or Perpetual License Revenue/Profit Sharing
Audit Rights                   Joint IP Ownership              ROFR/ROFO/ROFN
Change Of Control              License Grant                   Source Code Escrow
Competitive Restriction Excpt. Liquidated Damages              Termination For Convenience
Covenant Not To Sue            Minimum Commitment              Third Party Beneficiary
                               Most Favored Nation             Unlimited/All-You-Can-Eat-License
                               No-Solicit Of Customers         Volume Restriction
                               No-Solicit Of Employees         Warranty Duration
                               Non-Disparagement
                               Non-Transferable License
```

We left these alone because of the three filters in [ADR-002](./DECISIONS.md#adr-002--pick-12-categories-from-cuads-41): commercial impact, annotation density, inferential difficulty. Many of these are interesting but specialized (Source Code Escrow, ROFR/ROFO/ROFN), commercially low-impact (Affiliate License-Licensee), or so rare in the dataset that a metric over them would be dominated by absent values (Most Favored Nation, Audit Rights).

**Adding one is straightforward:** add the field to `extractor/schemas.py`, add the CUAD category string to `TARGET_CATEGORIES` in `training/ingest_cuad.py`, regenerate. See `docs/SCHEMA.md` §5 for the full procedure.

### 4.3 What we do *not* support changing without a versioned schema

The 12 fields are committed to in this repo's v0.1.0 schema. Any change to the field set or canonical order requires a versioned `ContractExtractionV2` to keep evaluation runs comparable across model versions. This is recorded in [ADR-002](./DECISIONS.md#adr-002--pick-12-categories-from-cuads-41).

---

## 5. Mapping rules: CUAD spans → structured fields

CUAD often returns multiple text spans for a single (contract, category) pair — long clauses split across paragraphs, party-name variants, dates referenced in different sections. We need deterministic rules for collapsing these into a single structured value per field.

### 5.1 `parties` (list field)

```
spans = ["Acme Corp", "ACME CORP", "Beta Inc", "Beta Inc"]
       ──┬──────────────────────────────────────────────
         │  collect non-empty
         │  case-insensitive dedup
         │  preserve casing of first occurrence
         ▼
result = ["Acme Corp", "Beta Inc"]
```

- Empty / whitespace-only spans dropped.
- Dedup is case-insensitive on stripped strings.
- The casing of the **first** occurrence is preserved. This usually means the legal name in caps (e.g., `"ACME CORPORATION"`) wins over a short alias (`"Acme"`) only if the legal name appeared first; otherwise the order in CUAD's annotation determines it.

### 5.2 Date fields (`agreement_date`, `effective_date`, `expiration_date`)

```
spans = ["May 15, 2018", "the Effective Date"]
       ──┬───────────────────────────────────
         │  pick longest non-empty span
         │  dateutil.parser.parse(s, fuzzy=True, default=1900-01-01)
         │  if parsed year != 1900 → emit ISO YYYY-MM-DD
         │  else → return raw stripped string
         ▼
result = "2018-05-15"   (or "the Effective Date" if unparseable)
```

The 1900 sentinel year is how we detect "dateutil filled in a year because the input had none" (a parse failure for our purposes). When that happens, the original text is returned so downstream consumers see CUAD's actual annotation rather than a fake date.

Examples of inputs and outputs:

| CUAD span | Output | Reason |
|---|---|---|
| `"May 15, 2018"` | `"2018-05-15"` | Clean parse. |
| `"5/15/2018"` | `"2018-05-15"` | Clean parse. |
| `"15 May 2018"` | `"2018-05-15"` | Clean parse. |
| `"as of the Effective Date"` | `"as of the Effective Date"` | Unparseable — kept verbatim. |
| `"during the term"` | `"during the term"` | Unparseable. |
| `""` (empty span) | `null` | No content. |

### 5.3 Other singular text fields

`document_name`, `governing_law`, `renewal_term`, `notice_period_to_terminate_renewal`, `exclusivity`, `non_compete`, `cap_on_liability`, `uncapped_liability` all use the same rule:

```
spans = ["short", "the longest, most informative span", "medium"]
       ──┬─────────────────────────────────────────────
         │  drop empty/whitespace-only
         │  pick LONGEST span (by character count)
         ▼
result = "the longest, most informative span"
```

The longest-span rule is a simple, defensible heuristic. The longest answer span tends to be the most informative — full clause text rather than a partial reference. We considered concatenating all spans with `" | "` (higher recall) and first-span (simpler), and rejected both as documented in [ADR-007](./DECISIONS.md#adr-007--field-mapping-rules-longest-span-and-case-insensitive-dedup).

### 5.4 Validation gate

Before writing each row to `cuad_parsed.jsonl`, the aggregated dict is validated against `ContractExtraction.model_validate(...)`. Any row that fails validation is **dropped** with a single log line:

```
[INFO] Dropped {contract_id}: {one-line reason}
```

In practice, no rows are dropped on real CUAD data. The validation gate exists to catch upstream changes (CUAD-QA reformat, bad span data) early.

---

## 6. Final output: ChatML JSONL splits

`training/prepare_dataset.py` reads `data/raw/cuad_parsed.jsonl` and produces three output files in `data/processed/`. Each file is JSONL — one JSON object per line.

### 6.1 The 3-message ChatML structure

Every row of `train.jsonl`, `val.jsonl`, `test.jsonl` looks like:

```json
{
  "messages": [
    {"role": "system",    "content": "<system prompt — fixed across all rows>"},
    {"role": "user",      "content": "Extract structured clauses from this contract:\n\n<contract text>"},
    {"role": "assistant", "content": "<compact one-line JSON in canonical 12-field order>"}
  ],
  "contract_id": "<original CUAD title>"
}
```

### 6.2 The system prompt (verbatim)

```
You are a legal contract analyst. Extract structured clauses from contracts.
```

That's it. Two sentences. The system prompt is **deliberately minimal** — we want the model to learn the schema from the assistant turn, not from elaborate instructions in the system prompt. A long system prompt would mean every inference call carries that token weight forever.

### 6.3 The user prompt template

```
Extract structured clauses from this contract:

<contract text — head+tail truncated if > 8000 tokens>
```

Truncation rule: if the contract is ≤ 8000 tokens (counted with the Llama 3.1 tokenizer), it's kept verbatim. Otherwise:

```
<first 5000 tokens of the contract>
\n[...TRUNCATED...]\n
<last 3000 tokens of the contract>
```

The marker is a stable literal string (`\n[...TRUNCATED...]\n`) so the model sees it consistently. The 5000+3000 split balances **preamble** (first ~3000 tokens — parties, dates, recitals) against **risk-allocation block** (last ~2000 tokens — governing law, liability caps, uncapped-liability carve-outs). Head-only truncation would systematically lose the back-of-document clauses, which are exactly the fields with the highest commercial impact. See [ADR-008](./DECISIONS.md#adr-008--head--tail-truncation-at-5000--3000-tokens).

### 6.4 The assistant turn (compact one-line JSON)

The assistant turn is the **training target**. It's serialized as:

```python
json.dumps(extraction, ensure_ascii=False, separators=(",", ":"))
```

That gives compact JSON with no whitespace after `,` or `:`, with non-ASCII characters preserved (e.g., `España` stays readable rather than `\u00cd...`), and with keys in **canonical declaration order** matching `ContractExtraction.model_fields`.

A real example, partially trimmed:

```
{"document_name":"CO-BRANDING AGREEMENT (FORM)","parties":["NETTAXI Online Communities, Inc.","Solutions Media, Inc."],"agreement_date":"1999-11-05","effective_date":"1999-11-05","expiration_date":null,"governing_law":"the laws of the State of California","renewal_term":null,"notice_period_to_terminate_renewal":null,"exclusivity":null,"non_compete":null,"cap_on_liability":null,"uncapped_liability":null}
```

Why compact? Token count. `indent=2` pretty-printing adds ~30% more tokens, slowing training, raising inference latency, and consuming more context budget per call. The choice is documented in [ADR-011](./DECISIONS.md#adr-011--compact-one-line-json-for-the-assistant-target).

Why no Markdown fences (`` ```json ... ``` ``)? They add tokens and make output parsing more fragile. The output is a bare JSON object; downstream code does `json.loads(prediction_str)` directly.

### 6.5 Splits: 408 / 51 / 51, deterministically seeded

```python
indices = list(range(510))
random.Random(42).shuffle(indices)
n_train, n_val = int(0.8 * 510), int(0.1 * 510)   # 408, 51
train_idx = indices[:408]
val_idx   = indices[408:459]
test_idx  = indices[459:]                          # 51 (remainder)
```

- **Seed = 42.** Any downstream training run should use the same seed for reproducibility.
- **Floor rounding for train and val** → test absorbs the remainder. This is why test gets 51 (not 50) when 510 doesn't divide cleanly.
- **Byte-stable.** Re-running `prepare_dataset.py` produces byte-identical output files. Deletes and re-creates would diff to nothing.

The same seed is used for both the data split and any downstream training-time randomization. This avoids a class of subtle bugs where independent randomness sources silently interact.

### 6.6 `contract_id` is *not* in the messages

The top-level `contract_id` is preserved for traceability — when a model misclassifies a row, you need to be able to look up the original contract — but it's *not* part of the messages and therefore not part of the model's training context. The model learns to extract from contract text, not from contract IDs.

---

## 7. Coverage statistics (real numbers)

These are real population counts measured against `data/processed/train.jsonl` after a full pipeline run. Numbers may shift by 1–2 if upstream CUAD-QA changes; the deterministic 80/10/10 split itself is byte-stable.

### 7.1 Per-field population in the train split (n=408)

| Field | Populated | % | Density tier |
|---|---:|---:|---|
| `document_name` | 408 | 100% | Always |
| `parties` | 407 | ~100% | Always |
| `agreement_date` | 372 | 91% | High |
| `governing_law` | 350 | 86% | High |
| `expiration_date` | 332 | 81% | High |
| `effective_date` | 312 | 76% | High |
| `cap_on_liability` | 227 | 56% | Medium |
| `exclusivity` | 144 | 35% | Medium |
| `renewal_term` | 144 | 35% | Medium |
| `non_compete` | 99 | 24% | Low |
| `uncapped_liability` | 91 | 22% | Low |
| `notice_period_to_terminate_renewal` | 89 | 22% | Low |

### 7.2 What this distribution implies

**For training.** The model will see `null` for low-density fields most of the time. This is *desirable* — it teaches the model conservative extraction (don't hallucinate when the contract doesn't address the topic) rather than over-extraction. Hallucination is the #1 failure mode for LLM-based clause extractors.

**For evaluation.** A naive "always-null" baseline already achieves ~78% accuracy on the lowest-density field (`notice_period_to_terminate_renewal`) just by being right when there's nothing to extract. This is why we use F1 (set-based for `parties`, exact-match for singular) rather than raw accuracy: F1 punishes false negatives on the populated cases without rewarding the trivial null-emitting baseline. See `evaluation/metrics.py` and the unit tests in `tests/test_metrics.py`.

**For dataset extension.** The 6 medium/low-density fields are exactly where adding more annotated contracts would have the highest marginal value. The 4 high-density fields are essentially saturated.

### 7.3 Why some fields are sparse

- `notice_period_to_terminate_renewal`: only present when the contract has an auto-renewal mechanism, which itself is in only 35% of contracts (cf. `renewal_term`). 22% is consistent with that — most renewal clauses do specify a notice period.
- `non_compete`: legitimately rare in commercial (non-employment) contracts. CUAD's contracts are mostly B2B, so non-competes are mostly limited to post-termination distributor-restraint clauses.
- `uncapped_liability`: only meaningful when there's a `cap_on_liability` to carve out from. ~22% / ~56% ≈ 40% of capped contracts also have an explicit carve-out clause, which matches expectations for well-drafted commercial contracts.
- `exclusivity` (35%): exclusivity is a premium business term — present mostly in distribution, license, and supply contracts, not service or employment contracts.

---

## 8. Known limitations

Documented honestly so anyone using or extending the dataset has eyes open.

### 8.1 What CUAD doesn't annotate

CUAD's 41 categories cover most commercially-critical clauses but not all. Categories *missing* from CUAD that a deal lawyer might still want extracted:

- **Indemnification (as a standalone category).** CUAD does not annotate indemnification clauses directly — it treats them only insofar as they appear as carve-outs from the liability cap (i.e., under `Uncapped Liability`). A model trained on CUAD will *not* learn to extract a contract's full indemnification clauses. This is a real gap — we considered it carefully when picking our 12 fields and chose `uncapped_liability` as the closest available alternative ([ADR-002](./DECISIONS.md), discussion of category swap).
- **Data protection / privacy.** Not a CUAD category. Add one would require manual annotation.
- **Force majeure.** Not a CUAD category. Same.
- **Most M&A-specific deal mechanics** (earn-outs, indemnity escrow caps, baskets). CUAD has *some* of these but coverage is uneven.

### 8.2 Truncation losses

Contracts longer than 8000 tokens get the middle removed by head+tail truncation. The marker `\n[...TRUNCATED...]\n` tells the model some content is missing. In the CUAD corpus, ~10–15% of contracts are long enough to be truncated; for those, we lose roughly the middle 30–60% of the document.

What this can mean in practice:
- A clause that lives only in the truncated middle is invisible to the model.
- Defined-term references that point into the truncated section (`"Acme" has the meaning ascribed in Section 3.2`) lose their definitions — though the section number stays in scope, the substantive definition may not.
- Section numbering becomes ambiguous: the model sees Section 1–N and Section M–end with a gap.

We accept these losses as the cost of a fixed 8000-token training budget. See [ADR-008](./DECISIONS.md#adr-008--head--tail-truncation-at-5000--3000-tokens) for the reasoning. A future improvement would be **answer-span-aware truncation** using CUAD's `answer_start` offsets — keep the truncation budget concentrated around the actual annotated spans rather than at the document boundaries. We chose not to do this in v0.1.0 because the head+tail heuristic is simpler, deterministic, and easier to debug.

### 8.3 Multi-span aggregation losses

When a CUAD category has multiple annotated spans for one contract, we keep only the longest. This loses any information present in the shorter spans that isn't subsumed by the longer one.

For most fields, the longest span is genuinely the most informative — it's usually the full sentence rather than a partial cross-reference. But edge cases exist: a contract with two separate liability caps (e.g., one for IP claims, one for everything else) will only have one extracted, even though both are independently meaningful.

### 8.4 Date parsing edge cases

We use `dateutil.parser.parse(s, fuzzy=True)` to normalize dates. `fuzzy=True` accepts surrounding non-date text, which helps with messy CUAD spans like `"as of May 15, 2018, by and between"`. But fuzzy parsing has known failure modes:

- Truly ambiguous numbers: `"5/6/2018"` → 2018-05-06 in US convention, which is what dateutil does, but **not what a UK contract would mean**. CUAD is US-biased so this is usually correct.
- Year-only or month-year-only inputs are interpreted with the day defaulted to the 1st. This is a parse heuristic, not a fact about the contract.
- Strings that contain a parseable date *plus* unrelated content (`"the date 30 days after the closing date of January 15, 2018"`) parse to `2018-01-15`, which may not be the intended semantic date.

In all of these cases we keep what `dateutil` returns. The validation gate doesn't reject ambiguous parses; it only rejects type errors.

### 8.5 Annotation noise

CUAD is expert-annotated but not perfect. The original CUAD paper explicitly acknowledges:

- A small percentage of categories show inter-annotator disagreement.
- Some categories (especially the more nuanced ones — `Most Favored Nation`, `Anti-Assignment` exceptions) have known annotation drift.
- The dataset was built with cost constraints; not every span was double-annotated.

For the 12 fields we extract, the categories we use are among CUAD's better-annotated ones, but treating CUAD annotations as "ground truth" is a useful approximation, not a fact.

### 8.6 Domain bias (recap)

CUAD is M&A-adjacent commercial contracts from EDGAR. A model trained on it should be expected to underperform on:

- Employment contracts (only obliquely represented).
- Consumer ToS / EULAs (not represented).
- Loan and security agreements (lightly represented).
- Real-estate documents (not represented).
- Non-English contracts (not represented).

Anyone using a CUAD-trained model in production should benchmark on their actual contract distribution, not assume CUAD generalizes.

---

## 9. How to inspect and verify

A few `jq` and Python recipes for common questions about the data. All commands assume you've run the full pipeline at least once.

### 9.1 Inspect a single training row

```bash
# Top-level shape
head -1 data/processed/train.jsonl | jq 'keys'
# → ["contract_id", "messages"]

# Contract id
head -1 data/processed/train.jsonl | jq -r '.contract_id'

# All three messages (system / user / assistant)
head -1 data/processed/train.jsonl | jq -r '.messages[].role'

# The assistant target (compact JSON), pretty-printed for reading
head -1 data/processed/train.jsonl | jq -r '.messages[2].content' | jq
```

### 9.2 Per-field population stats (the §7.1 table)

```bash
jq -c '.messages[2].content' data/processed/train.jsonl \
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

### 9.3 Find all rows where a specific field is populated

```bash
# E.g., contracts that have a non-null uncapped_liability
jq -c 'select((.messages[2].content | fromjson | .uncapped_liability) != null) | .contract_id' \
   data/processed/train.jsonl \
   | head -5
```

### 9.4 Token-length distribution (using the Llama tokenizer)

```bash
python -c "
from transformers import AutoTokenizer
import json, statistics
tok = AutoTokenizer.from_pretrained('unsloth/Meta-Llama-3.1-8B-Instruct')
lengths = []
with open('data/raw/cuad_parsed.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        lengths.append(len(tok.encode(obj['contract_text'], add_special_tokens=False)))
lengths.sort()
print(f'n={len(lengths)}')
print(f'min={lengths[0]}  p25={lengths[len(lengths)//4]}  median={lengths[len(lengths)//2]}  p75={lengths[3*len(lengths)//4]}  max={lengths[-1]}')
print(f'over 8000 tokens: {sum(1 for L in lengths if L > 8000)} / {len(lengths)} ({100*sum(1 for L in lengths if L > 8000)/len(lengths):.0f}%)')
"
```

### 9.5 Verify no rows were dropped

```bash
# Raw line count matches train+val+test
wc -l data/raw/cuad_parsed.jsonl data/processed/*.jsonl
# Expected: 510 raw, 408 train, 51 val, 51 test → 510 = 408+51+51
```

### 9.6 Verify canonical key order in every row of train

```bash
EXPECTED='document_name,parties,agreement_date,effective_date,expiration_date,governing_law,renewal_term,notice_period_to_terminate_renewal,exclusivity,non_compete,cap_on_liability,uncapped_liability'

bad=$(jq -r '.messages[2].content | fromjson | keys_unsorted | join(",")' data/processed/train.jsonl \
       | grep -vxc "$EXPECTED" || true)

[ "$bad" = "0" ] && echo "OK: all 408 rows have the canonical key order" || echo "$bad rows have wrong key order"
```

---

## 10. Reproducibility and provenance

### 10.1 Deterministic everything

| Step | Source of randomness | Seed/Determinism |
|---|---|---|
| CUAD-QA download | None | HuggingFace pins by revision hash |
| Pool train+test | None | Order is fixed by the dataset |
| Group by title | None | Title is a stable string |
| Span aggregation | None | Pure functions; same inputs → same outputs |
| Date parsing | `dateutil.parser` | Same library version → same parse outputs |
| Truncation | None | Token count is deterministic given a fixed tokenizer |
| Tokenizer | `meta-llama/...` (gated) or `unsloth/...` (mirror) | Same revision → same chat template |
| Train/val/test split | `random.Random(42).shuffle` | Seeded |

A re-run on the same machine, same CUAD-QA revision, same Python and library versions produces **byte-identical** output files. We don't currently emit a manifest, but `wc -l` + a `git diff` on the JSONL files is a sufficient regeneration check.

### 10.2 Tokenizer source logging

`prepare_dataset.py::load_tokenizer()` logs which tokenizer source was used at startup:

```
[INFO] Loaded tokenizer from meta-llama/Llama-3.1-8B-Instruct (gated)
```

or

```
[INFO] Loaded tokenizer from unsloth/Meta-Llama-3.1-8B-Instruct (unsloth mirror)
```

The two sources ship the same chat template, so output should be byte-identical regardless. Logging the source means any future divergence — say, the unsloth mirror updating its template before the gated source does — would be visible in run logs immediately.

### 10.3 What's redistributed and what isn't

| Artifact | Where it lives | Redistributed? |
|---|---|---|
| CUAD raw contracts | `~/.cache/huggingface/datasets/` (after download) | **No** — gitignored, not in this repo |
| `data/raw/cuad_parsed.jsonl` | `data/raw/` | **No** — gitignored, regenerated locally |
| `data/processed/{train,val,test}.jsonl` | `data/processed/` | **No** — gitignored, regenerated locally |
| The pipeline code | `training/`, `extractor/`, `evaluation/` | Yes — MIT licensed |
| Schema, mapping rules, splits | This repo | Yes — MIT licensed; data derives from CUAD CC BY 4.0 |

Any user who clones this repo regenerates the dataset from upstream CUAD-QA. We don't ship CUAD itself, which keeps the repo small and the CUAD attribution chain clean.

### 10.4 Verifying a regenerated run

```bash
# After re-running the pipeline, sanity-check matches:
md5sum data/processed/*.jsonl

# Compare to the most recent known-good values you kept from a previous run.
# These should match exactly across runs on the same upstream CUAD-QA revision.
```

If the MD5s differ, the upstream revision changed (or you tweaked something locally). The split itself is seeded; the only sources of variance are the upstream data and any local code edits.

---

## 11. Legal disclaimer

This project is a **data pipeline** and **schema specification** for legal contract extraction. It is not legal advice, and outputs of any model trained on this dataset should not be treated as a substitute for professional attorney review.

CUAD itself is annotated by trained law students under attorney supervision, but the annotations are intended to support **NLP research and tooling**, not to serve as legal opinions. The contracts in CUAD are real, publicly-filed instruments, but they are read and labeled at the level needed to train extractive models — not at the level a litigator preparing a brief would require.

If you intend to use a CUAD-trained model on real-world contracts in a way that affects legal decisions:

- Treat extracted clauses as a **first-pass review aid**, not as conclusive findings.
- Have human counsel verify any clause whose presence or wording would change a deal's economics.
- Be especially skeptical of fields with low population in the training distribution (§7.1) — the model has seen relatively few positive examples and is likelier to err on those.
- Benchmark on contracts representative of your actual use case, not just on the held-out CUAD test split (§8.6).

The schema, code, and metric definitions in this repo are released under the MIT license (see `LICENSE`). The CUAD data and any derived training files inherit the CC BY 4.0 attribution requirement.
