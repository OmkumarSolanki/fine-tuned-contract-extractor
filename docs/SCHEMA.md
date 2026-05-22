# The Extraction Schema

This document describes `ContractExtraction` — the 12-field Pydantic model that defines what we extract from every contract. For each field, we explain what it means in commercial-law context, why it matters in M&A diligence, and how CUAD annotates it.

The schema is implemented in [`extractor/schemas.py`](../extractor/schemas.py). Field declaration order is **canonical and load-bearing** — it determines key order in the assistant's training target and the iteration order of `evaluation/metrics.overall_f1`. Don't reorder.

---

## 1. Why these 12?

CUAD has 41 categories. Most are useful in some contexts; only a subset are universally critical. We picked 12 by three filters:

1. **Commercial impact.** Would a missed extraction here change a deal's economics or risk profile? Yes for uncapped liability carve-outs, no for "Affiliate License-Licensee".
2. **Annotation density.** Is the field present in a meaningful percentage of contracts? Yes for parties (every contract has them), no for "Audit Rights" (rare).
3. **Inferential difficulty.** Is the field hard enough to be a meaningful test of an extractor — neither trivially solved by a regex nor so rare that the metric is dominated by absent values? Yes for narrative clauses (renewal terms, exclusivity), less so for the document title.

The mix of "easy" fields (document name, parties, dates) and "hard" fields (renewal term, uncapped liability) gives any model trained on this dataset a richer accuracy story than 12 easy fields would.

---

## 2. The Schema

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class ContractExtraction(BaseModel):
    """The structured output schema for contract extraction."""

    # Identity
    document_name: Optional[str] = Field(None, ...)
    parties: List[str] = Field(default_factory=list, ...)

    # Dates
    agreement_date: Optional[str] = Field(None, ...)
    effective_date: Optional[str] = Field(None, ...)
    expiration_date: Optional[str] = Field(None, ...)

    # Legal framework
    governing_law: Optional[str] = Field(None, ...)

    # Term and renewal
    renewal_term: Optional[str] = Field(None, ...)
    notice_period_to_terminate_renewal: Optional[str] = Field(None, ...)

    # Commercial terms
    exclusivity: Optional[str] = Field(None, ...)
    non_compete: Optional[str] = Field(None, ...)

    # Risk allocation
    cap_on_liability: Optional[str] = Field(None, ...)
    uncapped_liability: Optional[str] = Field(None, ...)
```

All non-list fields are `Optional[str]` and default to `None`. `parties` is `List[str]` and defaults to `[]`. There are no other constraints — Pydantic v2's default `extra='ignore'` means future-compatible extra keys are silently dropped.

---

## 3. Field-by-field

### 3.1 `document_name`

**What it is:** The title of the contract — what a lawyer would call it on first reference. Examples: `"License Agreement"`, `"Distributor Agreement"`, `"Co-Branding and Advertising Agreement"`.

**Why it matters:** The document type tells you which law and which playbook applies. A "Master Services Agreement" implies one set of expectations (typically work orders, SLAs, warranties); a "License Agreement" implies another (territory, exclusivity, royalty mechanics). You can't reason about the rest of the contract without it.

**CUAD source:** Category `"Document Name"`. Usually one annotation per contract; appears verbatim in the contract's caption or recital.

**Mapping rule:** Longest non-empty span, or `None`. Most CUAD annotations are short (1–10 words) so longest vs. first usually agree.

**Difficulty:** Easy. Almost always in the first 100 tokens of the contract.

---

### 3.2 `parties`

**What it is:** All named parties to the contract. Stored as a `List[str]`. Examples: `["Acme Corp.", "Beta Inc."]`, `["Lime Energy Corp.", "Lime Energy of Illinois LLC"]`, `["2TheMart.com, Inc.", "i-Escrow, Inc."]`.

**Why it matters:** Identifying parties is step zero of any contract analysis. Counterparties drive concentration risk in M&A diligence; party identification is also the entry point for KYC, sanctions screening, and corporate-relationship mapping.

**CUAD source:** Category `"Parties"`. Often returns multiple text spans — the full legal name, the short form ("Acme"), and sometimes the role indicator ("the 'Distributor'"). CUAD's documentation explicitly notes that Parties may include 4–10 separate text strings per contract.

**Mapping rule:** Collect every non-empty span; deduplicate case-insensitively while preserving the original casing of the first occurrence. So `["Acme Corp", "ACME CORP", "Beta", "  beta  "]` becomes `["Acme Corp", "Beta"]`.

**Difficulty:** Medium. The names themselves are easy, but knowing where the named entities end and where role indicators ("the 'Buyer'") begin requires legal context.

**Metric:** Set-based F1 (`evaluation/metrics.parties_f1`) — case-insensitive set equality.

---

### 3.3–3.5 The three dates: `agreement_date`, `effective_date`, `expiration_date`

These three are the temporal anchors of the contract.

#### `agreement_date`

**What it is:** The date the agreement was *signed* (i.e., when the parties bound themselves). Sometimes called the execution date.

**Why it matters:** This is the legal moment of contract formation. It's the reference point for statute-of-limitations calculations, change-of-control clauses, and any obligation phrased as "from the date of this Agreement."

#### `effective_date`

**What it is:** The date the agreement *takes effect* — sometimes the same as the agreement date, sometimes deliberately later (e.g., closing of an acquisition, regulatory approval, or the start of a fiscal quarter).

**Why it matters:** Most operational provisions key off the effective date, not the agreement date. Term length, renewal windows, and SLA commitments all start counting from here.

#### `expiration_date`

**What it is:** The date the agreement expires, if specified. Many contracts auto-renew or are perpetual, in which case this field is `None`.

**Why it matters:** It tells you when the deal needs to be renegotiated, terminated, or renewed. Combined with `notice_period_to_terminate_renewal`, you can compute the latest date at which a counterparty must be notified to prevent automatic renewal.

#### Mapping rule (all three)

Take the longest non-empty span; pass through `dateutil.parser.parse(s, fuzzy=True)`; emit ISO `YYYY-MM-DD` if a real date was extracted, otherwise return the raw stripped span as a fallback. See [`DATA_PIPELINE.md`](./DATA_PIPELINE.md) §2.3 for the exact algorithm and edge cases.

**Difficulty:** Easy-to-medium. The hardest case is when the document uses prose like "the effective date of the Distribution Agreement, dated as of January 1, 2024" — fuzzy parsing usually catches this. Cases like "as of the Effective Date" (referring to a previous document's date that isn't in this contract) fall back to the raw string and are correctly None'd out by the user downstream.

---

### 3.6 `governing_law`

**What it is:** The legal jurisdiction whose law governs disputes under the contract. Examples: `"Delaware"`, `"the laws of the State of New York, without regard to conflicts of laws principles"`, `"England and Wales"`.

**Why it matters:** Choice of law determines which body of contract law (and which procedural rules) applies. It affects contract interpretation, available remedies, and forum-selection enforceability. In tech contracts, Delaware and New York dominate US choice-of-law clauses; Illinois and California are common too.

**CUAD source:** Category `"Governing Law"`. Annotations are typically full-sentence spans like `"This Agreement will be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflicts of laws principles."`

**Mapping rule:** Longest non-empty span. We do *not* attempt to extract just the jurisdiction name — keeping the full clause preserves carve-outs ("without regard to conflicts of laws principles", "subject to mandatory laws of the consumer's home jurisdiction") that matter for downstream review.

**Difficulty:** Medium. The clause is structurally formulaic but appears at the end of the contract (~80–95% of the way through), where head-only truncation would have lost it. Our head + tail truncation is specifically designed to keep this in view.

---

### 3.7 `renewal_term`

**What it is:** How (if at all) the agreement renews after its initial term. Examples: `"Auto-renews for successive 1-year terms unless either party gives 60 days' written notice"`, `"This Agreement shall renew only by mutual written agreement"`, or `None` if no renewal mechanism is specified.

**Why it matters:** Auto-renewal clauses are a frequent source of unintended contract continuation — companies forget to send timely termination notice and end up bound for another year. Renewal terms are also often *asymmetric* — one party has unilateral renewal rights, the other doesn't.

**CUAD source:** Category `"Renewal Term"`. Often a multi-sentence span explaining the mechanism.

**Mapping rule:** Longest non-empty span.

**Difficulty:** Hard. The clause often spans multiple sentences and references defined terms that appear elsewhere in the contract.

---

### 3.8 `notice_period_to_terminate_renewal`

**What it is:** The notice period a party must give to *prevent* automatic renewal. Examples: `"60 days written notice prior to the end of the then-current term"`, `"at least 90 days' notice"`, or `None`.

**Why it matters:** Pure operational risk. Most automatic-renewal disputes come down to whether timely notice was given. Knowing the exact required notice (and the form, e.g., written + certified mail) is the difference between a clean exit and another year of obligations.

**CUAD source:** Category `"Notice Period To Terminate Renewal"`. Closely related to `renewal_term`; sometimes the same clause carries both annotations.

**Mapping rule:** Longest non-empty span.

**Difficulty:** Hard, similar to `renewal_term`. The two are highly correlated — getting one right typically means getting the other right too.

---

### 3.9 `exclusivity`

**What it is:** Any exclusivity restrictions — territorial, customer-based, product-based, or otherwise. Examples: `"Exclusive distributor in the United States and Canada"`, `"Customer agrees not to engage other vendors for the same scope of services for the term of this Agreement"`, or `None`.

**Why it matters:** Exclusivity is one of the most economically significant clauses in commercial contracts. Exclusive distribution agreements affect downstream pricing power; exclusive customer arrangements affect competitive dynamics; failure to honor or recognize exclusivity is a common source of breach claims.

**CUAD source:** Category `"Exclusivity"`. Typically a single clause spanning 1–4 sentences.

**Mapping rule:** Longest non-empty span.

**Difficulty:** Medium-hard. The clause is often phrased indirectly ("during the term, neither party shall ... with any third party engaged in ..."); a model needs to recognize the exclusivity *concept* without an explicit "exclusivity" header.

---

### 3.10 `non_compete`

**What it is:** Restrictions on competing activities — typically by one party against the other (or against the deal economy as a whole). Examples: `"For 12 months following termination, Distributor shall not market or sell competing products in the Territory"`, `"Employee shall not, during employment and for one year thereafter, engage in any business that competes with the Company"`, or `None`.

**Why it matters:** Non-competes shape post-termination markets and (especially in employment contexts) are subject to heavy regulatory scrutiny — California voids most employee non-competes, the FTC moved to ban many of them in 2024. In M&A, non-competes against the seller's principals are standard but increasingly contested.

**CUAD source:** Category `"Non-Compete"`. The CUAD label combines several variants — competition restrictions on a party, on its principals, and on its affiliates — into one category.

**Mapping rule:** Longest non-empty span.

**Difficulty:** Medium-hard. Similar to exclusivity; the model has to recognize the *behavior being restricted* even when the clause doesn't use the word "non-compete".

---

### 3.11 `cap_on_liability`

**What it is:** The contractual cap on a party's monetary liability, if any. Examples: `"Each party's aggregate liability shall not exceed the fees paid by Customer in the prior 12 months"`, `"$5,000,000"`, `"the greater of (a) $1,000,000 or (b) the fees paid in the 24 months preceding the claim"`, or `None`.

**Why it matters:** Liability caps are *the* core risk-allocation lever in commercial contracts. Most enterprise software, services, and licensing contracts cap liability at "fees paid in the prior X months" with carve-outs for IP infringement, confidentiality breaches, and gross negligence/willful misconduct. The cap (and its carve-outs) directly determines a counterparty's downside exposure.

**CUAD source:** Category `"Cap On Liability"`. Typically a multi-sentence span with carve-out language.

**Mapping rule:** Longest non-empty span.

**Difficulty:** Hard. Liability caps are some of the most syntactically complex clauses in commercial contracts — heavy use of defined terms, exception lists, and capitalized boilerplate.

---

### 3.12 `uncapped_liability`

**What it is:** The complement to `cap_on_liability` — `cap_on_liability` defines the ceiling on monetary damages; `uncapped_liability` defines the carve-outs that punch through it. These are the categories of liability that are explicitly excluded from the contract's overall cap. Typical examples: gross negligence, willful misconduct, infringement of intellectual property rights, breach of confidentiality, and fraud. Sample spans: `"Notwithstanding the foregoing, the limitations in this Section shall not apply to (a) breaches of confidentiality, (b) infringement of intellectual property rights, or (c) gross negligence or willful misconduct"`, `"Indemnification obligations under Section 9 shall not be subject to the cap set forth above"`, or `None`.

**Why it matters:** Carve-outs are the hidden teeth of any liability-cap clause. A $1M cap with five carve-outs (IP indemnity, confidentiality, fraud, gross negligence, breach of license scope) can leave a counterparty exposed to effectively unlimited damages on the categories that matter most. Reading the cap without reading the carve-outs is reading the contract wrong — which is exactly why these two fields (`cap_on_liability` + `uncapped_liability`) are paired in the schema.

**CUAD source:** Category `"Uncapped Liability"`. Often a single multi-clause sentence or a numbered carve-out list directly following the cap-on-liability clause.

**Mapping rule:** Longest non-empty span.

**Difficulty:** Hard. The clause is heavily dependent on cross-references to other sections (e.g., "the indemnification obligations under Section X"), so a model has to decide whether to extract the carve-out language verbatim or resolve the reference. We extract verbatim — resolving cross-references would make the gold annotation pipeline brittle.

---

## 4. Two related models in the same file

`extractor/schemas.py` defines two more Pydantic models that round out the data contract — a request and response shape that any future serving layer can consume:

### 4.1 `ExtractRequest`

```python
class ExtractRequest(BaseModel):
    contract_text: str = Field(..., min_length=50)
```

The request body for `POST /extract`. The `min_length=50` is a defensive validator — anything shorter than that is almost certainly not a real contract and should be rejected with a 422 before we burn GPU cycles.

### 4.2 `ExtractResponse`

```python
class ExtractResponse(BaseModel):
    extraction: ContractExtraction
    inference_time_ms: float
    tokens_generated: int
```

The response body. The timing/token fields are exposed so callers can build SLAs and dashboards without instrumenting the network round-trip themselves.

---

## 5. What's *not* in the schema (and why)

CUAD has 29 categories that we deliberately exclude. The criteria were the three filters in §1; the most notable omissions are:

- **`License Grant` / `License Type`** — present in only ~40% of CUAD contracts (license agreements specifically), so the per-field accuracy across the full test set would be unfair.
- **`Volume Restriction` / `Minimum Commitment`** — economically important but rare in CUAD.
- **`Audit Rights`, `Insurance`, `Source Code Escrow`** — important for some deal types, but their absence is informative — we don't want to train the model to emit verbose nulls for fields most contracts don't have.
- **`Anti-Assignment`, `Change of Control`** — these are critical in M&A but their CUAD annotations are inconsistent and would hurt training stability.

If you're fine-tuning for a specific deal type (e.g., licensing-only), it would be reasonable to drop some of our 12 and add license-specific categories. The schema is small and easy to swap.

---

## 6. Extending the schema

If you add or remove fields:

1. Update `ContractExtraction` in `extractor/schemas.py`. Keep the order semantic (group by Identity / Dates / Legal / Commercial / Risk).
2. Update `EXPECTED_FIELDS` in `tests/test_schemas.py::test_field_count_and_order` so the canonical-order test continues to lock the order.
3. Update `TARGET_CATEGORIES` in `training/ingest_cuad.py` to include the new field's CUAD category string.
4. If the new field is a list, add it to `LIST_FIELDS`. If it's a date, add it to `DATE_FIELDS`. Otherwise it's automatically treated as a singular string.
5. Re-run the data pipeline. The 80/10/10 split is deterministic so the same contracts go to train/val/test as before.

The 12 fields are the schema we're going to commit to for the public release. Future changes should be a versioned schema (e.g., `ContractExtractionV2`) to keep evaluation runs comparable.
