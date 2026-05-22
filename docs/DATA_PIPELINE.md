# Data Pipeline

This document is the deep-dive on how raw CUAD becomes ChatML training data. It complements [`ARCHITECTURE.md`](./ARCHITECTURE.md) (which has the high-level picture) and [`SCHEMA.md`](./SCHEMA.md) (which explains the 12 fields).

The pipeline has two scripts:

1. **`training/ingest_cuad.py`** — downloads CUAD-QA, recovers per-contract structured annotations, writes `data/raw/cuad_parsed.jsonl`.
2. **`training/prepare_dataset.py`** — formats each contract as a 3-message ChatML conversation, applies the Llama 3.1 chat template (sanity preview only), splits 80/10/10 deterministically.

---

## 1. Source: CUAD-QA on HuggingFace

The Contract Understanding Atticus Dataset ([`theatticusproject/cuad-qa`](https://huggingface.co/datasets/theatticusproject/cuad-qa)) is hosted as a SQuAD-style extractive QA dataset.

### 1.1 Shape

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | `<contract_title>__<category>_<chunk_index>` (chunk index optional for short contracts) |
| `title` | str | Contract identifier — same across all rows for one contract |
| `context` | str | A passage of the contract (long contracts are split into multiple chunks) |
| `question` | str | A natural-language question identifying the category |
| `answers.text` | List[str] | Zero or more text spans extracted from `context` |
| `answers.answer_start` | List[int] | Character offsets of those spans in `context` |

| Split | Rows |
|-------|-----:|
| `train` | 22,450 |
| `test` | 4,182 |

There are **510 unique contract titles** across the two splits combined.

### 1.2 Why are there ~52 rows per contract instead of 41?

CUAD has 41 categories. So 41 questions × 510 contracts = ~20,910 rows minimum. The actual count is 26,632. The excess comes from contracts that don't fit into a single passage and get **chunked**: each (chunk, question) pair becomes its own row, so a 3-chunk contract contributes 3 × 41 = 123 rows.

The chunk index is in the `id` suffix:

```
LIMEENERGYCO_..._DISTRIBUTOR AGREEMENT__Document Name_0
                                      ↑                ↑
                                   category        chunk index
```

For short contracts that fit in one passage, the chunk index is omitted entirely:

```
ACCELERATEDTECHNOLOGIESHOLDINGCORP_..._JOINT VENTURE AGREEMENT__Document Name
                                                                ↑
                                                            no _N suffix
```

The id parser (`extract_category_from_id`) handles both forms with the regex:

```python
_ID_REGEX = re.compile(r"__(?P<cat>.+?)(?:_(?P<chunk>\d+))?$")
```

Non-greedy `.+?` for the category, optional non-capturing `(?:_<digits>)?` for the chunk. When the chunk group is `None`, `extract_chunk_index_from_id` defaults to 0.

### 1.3 Why we pool train + test

CUAD-QA's own train/test split is for QA model training. We're doing a **structured extraction** task on whole contracts — the per-question split structure isn't useful to us. We pool both splits into one ~510-contract pool and do our own deterministic 80/10/10 contract-level split in `prepare_dataset.py` (with `seed=42`).

---

## 2. Step 1 — `training/ingest_cuad.py`

### 2.1 Purpose

Convert the per-(contract, chunk, question) row format into one row per contract, with full reconstructed text and the 12 schema fields filled in.

Output line shape:
```json
{
  "contract_id": "<CUAD title>",
  "contract_text": "<concatenated, deduplicated context chunks>",
  "annotations": {
    "document_name": "...",
    "parties": ["...", "..."],
    "agreement_date": "YYYY-MM-DD",
    "...": "..."
  }
}
```

### 2.2 Algorithm

```
1. Load the dataset:
       ds = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)
       pooled = concatenate_datasets([ds["train"], ds["test"]])

2. Group rows by `title`:
       by_title: dict[str, list[row]] = defaultdict(list)
       for row in pooled:
           by_title[row["title"]].append(row)

3. For each contract title:
       a. Build category → spans dict:
              for row in rows_for_title:
                  category = extract_category_from_id(row["id"])
                  if category in TARGET_CATEGORIES.values():
                      category_to_spans[category].extend(row["answers"]["text"])

       b. Apply 12-field rules:
              annotations = aggregate_contract(category_to_spans)

       c. Validate:
              ContractExtraction.model_validate(annotations)
          On failure, log "Skipped {title}: ..." and continue.

       d. Reconstruct contract text:
              contract_text = assemble_contract_text(rows_for_title)
          On empty text, log and skip.

       e. Write:
              {"contract_id": title,
               "contract_text": contract_text,
               "annotations": annotations}
```

### 2.3 The 12-field mapping rules (`aggregate_contract`)

For each of the 12 schema fields, we look up the corresponding CUAD category string and apply one of three rules.

#### List field: `parties`

CUAD often returns multiple text spans for parties (e.g., the company's full legal name plus its short alias). We collect them all, then deduplicate **case-insensitively** while preserving the original casing of the first occurrence.

```python
def dedupe_preserve_case(spans):
    seen = set()
    out = []
    for span in spans:
        stripped = span.strip()
        if not stripped:
            continue
        key = stripped.lower()
        if key not in seen:
            seen.add(key)
            out.append(stripped)
    return out
```

So `["Acme Corp", "ACME CORP", "Beta", "  beta  "]` becomes `["Acme Corp", "Beta"]`.

#### Date fields: `agreement_date`, `effective_date`, `expiration_date`

Dates in CUAD vary wildly — `"5/15/2018"`, `"May 15, 2018"`, `"the 15th day of May, 2018"`, `"as of the Effective Date"`. We:

1. Pick the **longest** non-empty span (richest content tends to parse best).
2. Run it through `dateutil.parser.parse(s, fuzzy=True, default=datetime(1900, 1, 1))`.
3. If the parsed year is **1900** (our sentinel), it means dateutil filled the year from the default — i.e., there was no real date in the span. Treat as a parse failure and return the raw stripped string.
4. Otherwise, return `parsed.date().isoformat()` → `"2018-05-15"`.

This gives us:
- `"2018-05-15"` → `"2018-05-15"`
- `"May 15, 2018"` → `"2018-05-15"`
- `"dated as of January 1, 2024"` → `"2024-01-01"` (fuzzy match)
- `"as of the Effective Date"` → `"as of the Effective Date"` (raw fallback — preserves source for downstream review)
- `"2018"` → `"2018-01-01"` (year-only is a real date)
- `None`, `""`, `"   "` → `None`

#### Other singular string fields: `governing_law`, `renewal_term`, `non_compete`, `cap_on_liability`, `uncapped_liability`, etc.

CUAD returns full-sentence spans for these, often multiple per contract. The longest is usually the most informative (it covers the whole clause rather than a partial reference). We pick the longest non-empty span, or `None` if all spans are empty.

```python
def pick_longest_span(spans):
    candidates = [s.strip() for s in spans if s and s.strip()]
    return max(candidates, key=len) if candidates else None
```

This is a deliberate choice — it could be argued that *concatenating* multiple spans gives higher recall, but it also makes the assistant target longer and noisier, slowing training and hurting JSON validity. The longest-span rule is the simplest reasonable choice, and the model learns to emit a representative clause.

### 2.4 Reconstructing full contract text (`assemble_contract_text`)

Each contract title has many rows (one per (chunk, question)). The actual contract text is split across the unique `context` values, one per chunk index. We:

1. Walk all rows for this title.
2. For each row, parse the chunk index out of `id`.
3. Keep one representative `context` per chunk index (drops duplicates from the per-question rows).
4. Sort by chunk index ascending.
5. Join with `\n\n`.

```python
def assemble_contract_text(rows_for_title):
    chunk_to_context = {}
    for row in rows_for_title:
        try:
            chunk_idx = extract_chunk_index_from_id(row["id"])
        except ValueError:
            continue
        if chunk_idx not in chunk_to_context:
            chunk_to_context[chunk_idx] = row.get("context", "") or ""
    return "\n\n".join(chunk_to_context[k] for k in sorted(chunk_to_context))
```

Rows with malformed ids are silently skipped — robustness over strictness here, since the chunk recovery is best-effort.

### 2.5 CLI

```bash
python training/ingest_cuad.py [--output PATH] [--limit N] [--force]
```

- `--output` — output path (default: `data/raw/cuad_parsed.jsonl`).
- `--limit N` — process only the first N titles (after sorting for determinism). Useful for fast local iteration. Default: process all 510.
- `--force` — overwrite the output if it exists. Default: skip with a log message.

### 2.6 Smoke output (real run, 2026-05-20)

```
$ python training/ingest_cuad.py --force
2026-05-20 20:17:43,140 [INFO] Loading theatticusproject/cuad-qa from Hugging Face …
2026-05-20 20:17:48,872 [INFO] Loaded splits: train=22450, test=4182
2026-05-20 20:17:48,873 [INFO] Pooled rows: 26632
2026-05-20 20:17:49,523 [INFO] Unique contract titles: 510
Parsing contracts: 100%|████████████| 510/510 [00:01<00:00, 441.14it/s]
2026-05-20 20:17:50,733 [INFO] Wrote 510 contracts to data/raw/cuad_parsed.jsonl (skipped 0)
```

510 / 510 contracts, 0 skips. End-to-end takes ~3 seconds when the HF download is cached, ~10 seconds on a cold cache.

---

## 3. Step 2 — `training/prepare_dataset.py`

### 3.1 Purpose

Read `data/raw/cuad_parsed.jsonl`, format each row as a 3-message ChatML conversation suitable for `SFTTrainer`, and split deterministically into train/val/test.

Output line shape:
```json
{
  "messages": [
    {"role": "system",    "content": "You are a legal contract analyst. ..."},
    {"role": "user",      "content": "Extract structured clauses from this contract:\n\n<text>"},
    {"role": "assistant", "content": "<compact JSON>"}
  ],
  "contract_id": "<title>"
}
```

### 3.2 Tokenizer loading

Llama 3.1 ships with a chat template baked into `tokenizer_config.json`. We need that template at training time. The official `meta-llama/Llama-3.1-8B-Instruct` is **gated** — you have to accept the license on the HF Hub, then provide an `HF_TOKEN`. The unsloth mirror `unsloth/Meta-Llama-3.1-8B-Instruct` is public and ships an identical chat template.

```python
def load_tokenizer():
    if os.environ.get("HF_TOKEN"):
        try:
            tok = AutoTokenizer.from_pretrained(
                PRIMARY_TOKENIZER, token=os.environ["HF_TOKEN"]
            )
            logger.info("Loaded tokenizer from %s (gated, via HF_TOKEN)", PRIMARY_TOKENIZER)
            return tok, PRIMARY_TOKENIZER
        except Exception as exc:
            logger.warning("Failed: %s; falling back to %s", exc, FALLBACK_TOKENIZER)
    tok = AutoTokenizer.from_pretrained(FALLBACK_TOKENIZER)
    logger.info("Loaded tokenizer from %s (unsloth mirror)", FALLBACK_TOKENIZER)
    return tok, FALLBACK_TOKENIZER
```

The script logs which source was actually used so any mismatch (gated vs mirror) is visible at startup. If both fail (no token, no network), we raise `RuntimeError` with a remediation message.

### 3.3 Prompts (LOCKED)

```python
SYSTEM_PROMPT = "You are a legal contract analyst. Extract structured clauses from contracts."

USER_PROMPT_TEMPLATE = "Extract structured clauses from this contract:\n\n{contract_text}"
```

These constants live at the top of `training/prepare_dataset.py`. Any code that runs inference against a model trained on this dataset must use these strings byte-for-byte. If they diverge, the model has been trained on prompts that inference never sends, and accuracy drops sharply.

### 3.4 Truncation

CUAD contracts vary from ~1,000 to ~100,000+ tokens. Llama 3.1 8B has a 128k context window, so inference can handle anything CUAD throws at it. **Training memory**, however, scales with sequence length, so we cap training sequences at 8000 tokens.

The naive choice is head truncation: keep the first 8000 tokens, drop the rest. But legal contracts pack their **risk-allocation clauses** (governing law, uncapped liability, liability caps — exactly what we extract) at the *end*, in the boilerplate. Head-only truncation would systematically lose them.

So we use **head + tail truncation**:

```python
def truncate_text(text, tokenizer, max_total=8000, head=5000, tail=3000):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_total:
        return text
    head_text = tokenizer.decode(ids[:head], skip_special_tokens=True)
    tail_text = tokenizer.decode(ids[-tail:], skip_special_tokens=True)
    return head_text + "\n[...TRUNCATED...]\n" + tail_text
```

If the contract fits in 8000 tokens, return it as-is (no marker artifact). Otherwise keep 5000 from the start (preamble, parties, dates, body) and 3000 from the end (governing law, uncapped liability, liability caps), separated by a literal `\n[...TRUNCATED...]\n` marker so the model knows there's a gap.

### 3.5 Compact JSON serialization

The assistant's training target is a compact one-line JSON with keys in canonical (`ContractExtraction.model_fields`) order:

```python
def compact_json(annotations):
    ordered = {field: annotations.get(field) for field in ContractExtraction.model_fields}
    if ordered["parties"] is None:
        ordered["parties"] = []
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
```

Two intentional choices:

1. **Compact form** (`separators=(",", ":")`) — no whitespace after `,` or `:`. This is shorter (~30% fewer tokens than pretty-printed) and trains faster. We can pretty-print at inference time if we want.
2. **`ensure_ascii=False`** — preserves UTF-8 characters (`España` stays `España`, not `\u00cd...`). Important because contract names sometimes include accented characters.

Field ordering is deterministic and tested:

```python
def test_compact_json_field_order_canonical():
    shuffled = {k: ann[k] for k in reversed(list(ann))}
    out = compact_json(shuffled)
    parsed = json.loads(out)
    assert list(parsed.keys()) == list(ContractExtraction.model_fields)
```

### 3.6 Building messages

```python
def build_messages(contract_id, contract_text, annotations, tokenizer):
    extraction = ContractExtraction.model_validate(annotations)  # may raise ValueError
    truncated = truncate_text(contract_text, tokenizer)
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": USER_PROMPT_TEMPLATE.format(contract_text=truncated)},
            {"role": "assistant", "content": compact_json(extraction.model_dump())},
        ],
        "contract_id": contract_id,
    }
```

Validation runs before any heavy work (truncation, JSON serialization), so invalid rows fail fast.

### 3.7 Chat-template sanity check at startup

After building the first row, we render it through the tokenizer's chat template and log a head/tail preview:

```python
rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
logger.info("Chat template head: %s", rendered[:200].replace("\n", "\\n"))
logger.info("Chat template tail: %s", rendered[-200:].replace("\n", "\\n"))
logger.info("Rendered length: %d chars", len(rendered))
```

In a real run, this surfaces something like:

```
Chat template head: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nYou are a legal contract analyst. Extract structured clauses from contracts.
Chat template tail: ...uncapped_liability":null}<|eot_id|>
Rendered length: 38624 chars
```

This is purely diagnostic — we do *not* write the rendered text to the output file. `SFTTrainer` will apply the same template at training time.

### 3.8 Splitting (deterministic, seed=42)

```python
def split_indices(n, seed=42):
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]
```

For n=510: 408 / 51 / 51. Two runs with the same seed produce identical splits (tested). The remainder rounding goes into the test set so train and val are exactly 80% and 10%.

### 3.9 CLI

```bash
python training/prepare_dataset.py [--input PATH] [--output-dir PATH] [--seed N]
```

- `--input` — input JSONL (default: `data/raw/cuad_parsed.jsonl`).
- `--output-dir` — directory for `train.jsonl` / `val.jsonl` / `test.jsonl` (default: `data/processed`).
- `--seed` — split seed (default: 42).

### 3.10 Smoke output (real run, 2026-05-20)

```
$ python training/prepare_dataset.py
[INFO] Loaded tokenizer from unsloth/Meta-Llama-3.1-8B-Instruct (unsloth mirror)
[INFO] Tokenizer source: unsloth/Meta-Llama-3.1-8B-Instruct
[INFO] Read 510 raw contracts from data/raw/cuad_parsed.jsonl
[INFO] Built 510 ChatML rows (0 dropped)
[INFO] Chat template head: <|begin_of_text|><|start_header_id|>system<|end_header_id|>...
[INFO] Chat template tail: ...uncapped_liability":null}<|eot_id|>
[INFO] Rendered length: 38624 chars
[INFO] Train: 408, Val: 51, Test: 51, Dropped: 0 (output_dir=data/processed)
```

---

## 4. Inspecting the Output

```bash
# How many contracts in each split?
$ wc -l data/processed/*.jsonl
   408 data/processed/train.jsonl
    51 data/processed/val.jsonl
    51 data/processed/test.jsonl

# What does one row look like?
$ head -1 data/processed/train.jsonl | jq '.contract_id'
"RaeSystemsInc_20001114_10-Q_EX-10.57_2631790_EX-10.57_Co-Branding Agreement"

$ head -1 data/processed/train.jsonl | jq '.messages | length'
3

$ head -1 data/processed/train.jsonl | jq '.messages[0]'
{
  "role": "system",
  "content": "You are a legal contract analyst. Extract structured clauses from contracts."
}

$ head -1 data/processed/train.jsonl | jq '.messages[2].content' | head -c 200
"{\"document_name\":\"CO-BRANDING AGREEMENT (FORM)\",\"parties\":[\"Solutions Media...
```

---

## 5. Edge Cases and How We Handle Them

| Edge case | Where it shows up | Handling |
|-----------|--------------------|----------|
| Single-chunk contract — id has no `_<N>` suffix | `extract_category_from_id` | Optional regex group; chunk index defaults to 0 |
| Same chunk content repeated across 41 questions for one contract | `assemble_contract_text` | Dedup by chunk index (`chunk_to_context[idx] = ctx` only if absent) |
| `dateutil` "successfully" parses a date-less span by falling back to defaults | `parse_date_loose` | Sentinel year 1900 in `default=`; if parsed year == 1900, return raw string |
| Multiple parties with different casings | `dedupe_preserve_case` | Lowercase+strip key for dedup; preserve first-seen casing |
| Contract over 8000 tokens | `truncate_text` | Head 5000 + `[...TRUNCATED...]` marker + tail 3000 |
| Contract under 8000 tokens | `truncate_text` | Returned unchanged; no marker artifact |
| Annotations fail Pydantic validation | both `ingest` and `prepare` | One-line log with contract_id + reason; row dropped |
| `HF_TOKEN` missing or invalid | `load_tokenizer` | Logs and falls back to unsloth mirror; same chat template |
| `jinja2` not installed | `_log_chat_template_preview` | Caught and logged as warning; pipeline continues |
| Empty / whitespace-only context for a title | `assemble_contract_text` returns `""` | Caller logs `Skipped {title}: empty contract text` and skips |

---

## 6. Tests

The data pipeline has **43 helper-level tests** (26 in `test_ingest_cuad.py`, 17 in `test_prepare_dataset.py`). All run without network or transformers:

- `test_ingest_cuad.py` uses synthetic dicts and exercises every helper (id parsing both forms, date parser with ISO/natural/fuzzy/fallback/empty inputs, dedup, longest-span, full aggregation, contract text assembly).
- `test_prepare_dataset.py` uses an `IdentityTokenizer` fake (whitespace-tokenized round-trip) so truncate_text and build_messages are testable without the real Llama tokenizer.

End-to-end behavior (against real CUAD-QA) was verified manually during a smoke run.

---

## 7. Cost / Performance Profile

| Operation | Cold cache | Warm cache | Output size |
|-----------|-----------:|-----------:|------------:|
| `ingest_cuad.py` (510 contracts) | ~10 sec (HF download ~5 sec) | ~3 sec | 27 MB |
| `prepare_dataset.py` (510 → 408/51/51) | ~75 sec (tokenizer download ~15 sec) | ~60 sec | 14 MB total |
| Full unit test suite | <1 sec | <1 sec | — |

Both scripts are CPU-only. No GPU is required.

