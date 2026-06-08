# Evaluation Methodology & Results

How we measure quality — the metrics, the honest framing around them, and the current results with
confidence intervals, a trivial-model floor, a safety (hallucination) number, and a significance
test. Metric implementations live in [`evaluation/metrics.py`](../evaluation/metrics.py); the deeper
analysis lives in [`evaluation/analysis.py`](../evaluation/analysis.py). All numbers below are
reproducible from the committed prediction files via `python -m evaluation.analysis`.

---

## 1. The question

> **Does fine-tuning Llama 3.1 8B on CUAD give a meaningful accuracy lift on structured 12-field
> clause extraction — over and above what careful prompt engineering already achieves?**

That is a deliberately hard bar, so the comparison is **three-way**:

1. **Naive baseline** — base Llama 3.1 8B Instruct, one-line prompt.
2. **Strong-prompt baseline** — same base model, but a carefully engineered prompt (full schema
   description, worked few-shot examples, JSON-only constraint, ISO-date directive).
3. **Fine-tuned** — base + LoRA adapter, run with the *same prompt template used during training*.

If the fine-tune beats the strong baseline by a clear, statistically real margin, fine-tuning was
worth it. If it's a wash, prompt engineering alone was enough. Both outcomes would be informative.

## 2. Test set & split integrity

The held-out test set is **51 contracts** (the 10% split from `prepare_dataset.py`, `seed=42`),
out of 510. The split is **by contract**, and we verify there is **zero overlap** between train,
val, and test — so a high score cannot be memorization:

```
train 408 · val 51 · test 51 · train∩test 0 · train∩val 0 · val∩test 0  → leakage-clean
```

(`evaluation.analysis.check_leakage`, asserted in the test suite.)

## 3. Metrics

- **JSON-validity rate** — fraction of outputs that parse as JSON *and* validate against the
  12-field `ContractExtraction` schema (`is_valid_json`). The strict, headline metric.
- **Per-field match + `overall_f1`** — case-insensitive exact match per field (`parties` scored by
  set F1), averaged (`overall_f1`). **Caveat:** a schema-invalid prediction is scored as an *empty*
  extraction, and many CUAD fields are genuinely null — so an all-null prediction scores "correct"
  on the sparse fields. That makes the raw `overall_f1` only meaningful **relative to the
  always-null floor below**, never alone.

## 4. The always-null floor (read this before the results)

A trivial model that answers `null` for **every field of every contract** scores:

> **always-null floor: `overall_f1` = 0.4069**

This is the single most important context for the table. Because so many fields are legitimately
empty, "always null" already earns 0.41 *for free*. So a model's real skill is its lift **above
0.4069**, not its absolute `overall_f1`.

## 5. Results (held-out 51-contract test set)

Greedy decoding (temperature 0) on an A100 80GB. 95% confidence intervals: **Wilson** for validity,
**percentile bootstrap** (10k resamples, seed 42) for `overall_f1`.

| Model | JSON-valid | 95% CI | `overall_f1` | 95% CI | Lift over null floor |
|-------|-----------|--------|-------------|--------|----------------------|
| Naive baseline | 0/51 (0%) | [0%, 7%] | 0.4069 | [0.359, 0.454] | **+0.000** (= the floor) |
| Strong-prompt baseline | 6/51 (12%) | [6%, 23%] | 0.4139 | [0.368, 0.459] | +0.007 |
| **Fine-tuned (QLoRA)** | **49/51 (96%)** | **[87%, 99%]** | **0.7295** | **[0.666, 0.788]** | **+0.323** |

**How to read this honestly:**
- The validity story is real and large: **0% → 12% → 96%**. Even the lower CI bound (87%) clears
  both baselines' upper bounds.
- The `overall_f1` story is the important one, *once framed against the floor*: the baselines sit
  **at the floor** (0.407 / 0.414 vs. a 0.407 floor) — i.e. they contribute almost no real field
  accuracy; their nominal 0.41 is the free-null effect. The fine-tune scores **0.73, a genuine
  +0.32 over the floor.** That gap — not the bare 0.73 — is the fine-tune's real contribution.

## 6. Hallucination & safety (the legal-relevant number)

For a legal tool, *inventing* a value is worse than admitting "not present" — a lawyer might trust
the fabricated answer. We measure, per field, how often the model commits to a value when the gold
is null (`evaluation.analysis.null_confusion`). For the fine-tuned model, aggregated over all 12
fields × 51 contracts:

- **Hallucination rate: 10.4%** — of the truly-empty fields, the model invents a value 10.4% of the
  time (26 cases).
- **Present-precision 0.92 / recall 0.87** — when it commits to a value, the gold agrees one should
  exist 92% of the time; it catches 87% of the values that are actually present.

(This measures *commitment* — present vs. null — not whether the extracted text is verbatim
correct; value accuracy is the per-field match in §5.)

### Why the invalid outputs failed (failure modes)

Categorizing every *invalid* output by its primary failure reason
(`evaluation.analysis.failure_mode_breakdown`) reframes the "0% / 12%" baseline numbers:

| Model | Invalid | Breakdown |
|-------|---------|-----------|
| Naive | 51/51 | **100% markdown fence** — the model produced JSON but wrapped it in ```` ``` ```` |
| Strong-prompt | 45/51 | 19 malformed JSON · 17 prose around the JSON · 9 markdown fence |
| Fine-tuned | 2/51 | 2 truncated (ran out of the 2048-token budget mid-JSON) |

The takeaway: the base model's failures are overwhelmingly **formatting**, not comprehension — the
naive baseline *always* emitted JSON and merely fenced it. This is exactly the kind of error that
grammar/constrained decoding fixes **without fine-tuning**, which is why a constrained-decoding
baseline (see §8) is the right next comparison. The fine-tuned model's only 2 failures are
token-budget truncation — addressable with a larger budget or a serve-time JSON guarantee, not a
comprehension gap.

## 7. Is the improvement real, or luck?

With n=51, we test the *difference*, not just each score, using **McNemar's paired test** on
per-contract validity:

- Fine-tuned vs. naive: 49 contracts flipped right, 0 regressed → **p ≈ 0**.
- Fine-tuned vs. strong-prompt: 43 flipped right, 0 regressed → **p ≈ 0**.

The lift is overwhelmingly significant, not a sampling fluke.

### Hand-audit of the worst predictions

Reading the 5 lowest-scoring fine-tuned predictions field-by-field (pred vs. gold) separates real
model errors from scoring artifacts:

- **Roughly half the "misses" are exact-match artifacts or CUAD label noise, not model mistakes.**
  The model's `parties` lists are frequently *cleaner* than the gold — CUAD golds often include span
  fragments such as *"Hereinafter individually referred to as the Party..."* or an entity with a
  trailing comma, which the model sensibly omits, yet exact/set match scores it wrong. Long clause
  fields (`governing_law`, `cap_on_liability`) are often a faithful **paraphrase** scored 0.
- **Genuine model errors do occur** and are worth fixing: wrong `governing_law` *jurisdiction* on 2
  contracts (e.g. predicting one country's law where the gold names another), and a wrong
  `document_name`/document-type on 1.
- **Hallucination is visible** on the sparse clause fields: inventing `renewal_term`, `non_compete`,
  or `exclusivity` text where the gold is null — consistent with the 10.4% rate in §6.
- **Truncation** shows up directly: the single worst case was an essentially empty generation, and
  one `governing_law` was cut off mid-sentence — both the 2048-token-budget failure from §5.

**Implication:** the exact-match `overall_f1` is a **lower bound** on true quality — a semantic
(LLM-judge) metric is expected to raise the fine-tuned score, because many "wrong" answers are
correct paraphrases. The genuine jurisdiction/document-type errors and the ~10% hallucination rate
remain real, separate targets.

## 8. Known limitations (honest)

- **Small test set (n=51).** The CIs above are wide by construction (validity 96% spans 87–99%).
  A k-fold cross-validation over all 510 contracts is planned to tighten this.
- **Exact-match is harsh on free-text fields.** `governing_law`, `non_compete`, `cap_on_liability`,
  etc. are long spans; a semantically-correct paraphrase scores 0 under exact match, so §5 likely
  *understates* quality on those fields. A semantic (LLM-judge) metric is planned.
- **Truncation ceiling.** Long contracts are head+tail truncated to 8000 tokens; a clause living in
  the dropped middle is unrecoverable by construction, capping achievable recall on some fields.
- **Pretraining contamination (disclosed).** CUAD has been publicly available since 2021, and Llama
  3.1 was trained on large web crawls — so the base model **may already have seen these contracts**
  during its original pretraining. We cannot rule this out. It mainly affects the *baseline*
  numbers; the fine-tune's *relative* lift over the strong baseline is the more contamination-robust
  signal.
- **The fair-baseline question.** The base model's 0% validity is largely a *formatting* failure
  (markdown fences, prose), which constrained/grammar-guided decoding can fix without fine-tuning.
  A constrained-decoding baseline is planned so the accuracy lift can be attributed cleanly to
  fine-tuning rather than to "learned to emit JSON."

## 9. Reproducing these numbers

```bash
# Aggregate analysis (CIs, floor, hallucination, significance) — CPU, no network:
python -m evaluation.analysis            # writes data/results/analysis_summary.json

# Three-way comparison table:
python evaluation/compare.py             # writes data/results/comparison_summary.json
```

The raw per-contract predictions embed CUAD-derived text and are gitignored; only the aggregate,
text-free `*_summary.json` files are committed.
</content>
