# Problem Understanding — Business Entity Resolution (ML Challenge 2026)

Living document. Each section is backed by a reproducible script under `analysis/`
and is updated as the understanding changes. Numbers are from the **training** split
only; the test split is never used for analysis or tuning.

Status legend: ✅ verified from data · 📝 from README · 🔎 open question

---

## 0. Task framing 📝

- Three sources of business records (`entity_id`, `business_name`, `business_address`, `country`).
- **Source 1 (S1)** is the deduplicated reference. For every S1 entity, return the list of
  S2/S3 records that refer to the same real-world business (zero, one, or many).
- Output: one row per S1 entity in `matching_results.tsv` (+ `candidate_pairs.tsv` from
  the blocking stage, which must be a superset of the matches).
- Metric: **F0.5, macro-averaged per S1 entity, singletons included**. An S1 with no true
  matches scores 1.0 for an empty prediction and 0.0 for any prediction.
- Test adds a country (**France**) that is absent from training → nothing in the pipeline
  may branch on, filter by, or one-hot the country value.
- No external lookups (geocoding, registries, APIs). Final model: MIT/Apache-2.0, ≤ 8B params.

## 1. Ground truth (`train_ground_truth.tsv`)

Script: `analysis/01_ground_truth.py` → `analysis/outputs/01_ground_truth.json`

### 1.1 Structural rules the labels obey ✅
| Check | Result | Consequence |
|---|---|---|
| S1 rows vs GT rows | 2,206,821 = 2,206,821, same ID set | Every S1 entity is labelled (singletons have an empty list). |
| Duplicate IDs inside one list | 0 | — |
| **S2/S3 record claimed by >1 S1** | **0** of 7,638,365 links | **A target belongs to at most one S1.** Matching is a one-to-many *assignment*, not independent pair classification: when two S1s compete for a target, at most one can win. |
| Link targets missing from S2/S3 files | 0 | Labels are complete w.r.t. the files. |
| Links whose S1/target countries differ | **0** | Country equality is a lossless blocking filter (it is a generic equality test, so it also works for unseen countries). |
| corr(S1 id, target id), corr(S1 row, target row) | 0.0001, 0.0001 | IDs and file order carry no signal (no leakage to exploit or to accidentally depend on). |

### 1.2 Shape of the links ✅
- 7,638,365 links: S2 3,693,619 (48%), S3 3,944,746 (52%).
- Matches per S1: 0 → 5.6%, 1 → 5.4%, 2 → 17%, **3 → 24%**, 4 → 22%, 5 → 15%, 6 → 7.5%, 7+ → 4%; max 11. Mean 3.46.
- Per source: **≤ 5 from S2, ≤ 6 from S3** for any S1. 51% of S1s have ≥2 S2 records and 55% have ≥2 S3 records
  → **S2 and S3 are not deduplicated**; several records of one source describe the same business.
- Composition: none 5.6%, only S2 6.5%, only S3 7.5%, both 80.5%.
- **India and US are statistically identical** (singleton rate 5.59% vs 5.58%, mean links 3.46 vs 3.46, S2/S3 means equal)
  → the data comes from one synthetic generator applied per country. Expect France to follow the same process with
  French-specific surface noise.

### 1.3 Unlinked records (distractors) ✅
- ~26.6% of S2 and ~25.4% of S3 records are linked to **no** S1 entity, identical across countries.
- So a target without a good S1 partner is normal; "best S1 for this target" must still clear a bar. 🔎 What these
  distractors look like (random businesses vs. near-duplicates of S1 entities) is analysed in section 3.

### 1.4 What macro F0.5 rewards ✅
| Strategy (oracle) | Macro F0.5 |
|---|---:|
| Predict empty for everyone | 0.056 |
| Exactly 1 correct match per non-singleton, nothing else | 0.696 |
| Exactly 2 correct (or all if fewer) | 0.873 |
| All but one correct | 0.925 |
| All correct **plus one false positive** each | 0.752 |

Per entity with n true matches: missing one gives 0.833 (n=2) / 0.909 (n=3) / 0.938 (n=4), while one extra false
positive gives 0.714 / 0.789 / 0.833. **Missing a match is always cheaper than adding a wrong one (n ≥ 2)**, and a
false positive on a singleton costs the full 1.0.

→ Implications: rank candidates per S1, emit only confident ones; a couple of sure matches already earn most of the
score. The decision threshold must be tuned on the macro metric itself (with singletons), not on pair-level F1/AUC.

## 2. Source 1 (reference source)

Script: `analysis/02_source1.py` → `analysis/outputs/02_source1.json`

### 2.1 Hygiene ✅
- 2,206,821 rows (US 1,323,633 · India 883,188), unique `S1-<digits>` IDs, no empty name/address/country.
- **Clean, canonical formatting**: names are Title Case, pure ASCII (0 non-ASCII), no all-caps; addresses 0.025%
  non-ASCII, 968 with stray whitespace. S1 is the "clean" side; the noise lives in S2/S3.
- Names are business-style (`Orelee's Barbershop`), person/firm-style (`Lacy M. Lemmon, DDS`, `Machado, Gonzalez & Rosas`),
  and occasionally address-like (`1280 Madison Blvd Realty Center`, `62/97 Shreya`); 1.6% contain digits.

### 2.2 Names ✅
- Raw length mostly 3–4 words. After dropping legal forms the **core name is short: 1 word 3%, 2 words 54%, 3 words 34%**.
- Legal form is always the **last** word in S1 (never leading). Top endings — India: Limited 59%, Ltd/Ltd. 17%, LLP 4%;
  US: LLC 27%, Inc/Inc. 18%, Corp, PC/P.C., L.L.C., PLLC, LP.
- (A "null" bucket in the JSON's last-word table is an analysis artifact: names ending in a non-letter such as `)`.)

### 2.3 Addresses ✅
| | US | India |
|---|---|---|
| Comma-separated parts | 3 (84%) or 4 (16%, with a unit) | 2–19, mean 5.7 |
| Starts with house number | 86% | 26% |
| Components reordered (e.g. `MN, 506 9th Street, Duluth`) | ~14% (state first 6.4%) | common (`West Bengal, Kolkata, …`) |
| Last part | 2-letter state code (86%) | full state name (Maharashtra 18%, Delhi 12%, … "Orissa") |
| Unit designator | 13.6% | 3% |
| Landmarks ("Near …") | ~0% | 11% |
| No digit at all | 0% | 8.7% |
| **Postal codes (ZIP / PIN)** | **none** | **none** |

→ Address comparison must be **order-insensitive** (token sets, not sequence alignment), and there is **no postal code
to block on**; house numbers + street/locality words carry the address signal.

### 2.4 Ambiguity inside S1 — the main false-merge risk ✅
- No two S1 rows share both core name and normalized address (consistent with "deduplicated").
- **50% of S1 entities share their core name with another S1 in the same country** (40% share the full normalized name).
  Largest groups: "Meridian" ×572, "Cedar" ×351, "Summit" ×339 (US). Example: `B+ Retail Inc` exists at 4+ addresses.
- 5.3% share their normalized address with another S1 (several businesses at one address).

→ **Name alone cannot identify an entity**; the address must disambiguate, and vice versa. Because of the
one-S1-per-target rule (§1.1), a target that fits several same-name S1s should go to the one whose address fits best —
the model needs *competition features* (how this S1 ranks among all S1s that want the same target).

### 2.5 Singleton status is not visible from S1 ✅
The singleton rate is flat at **5.6% ± 0.2pp** across country, core-name length, address parts, digit-first,
unit presence, name ambiguity and legal form. The generator picks singletons at random, so a singleton can only be
recognised by the **absence of a convincing candidate** in S2/S3 → per-S1 decisions must be driven by candidate
scores (thresholds), not by S1 attributes.

## 3. Sources 2 and 3 relative to Source 1

Script: `analysis/03_sources23_vs_source1.py` → `analysis/outputs/03_sources23_vs_source1.json`
(noise rates from a 1.5M-link sample; similarity distributions from 300k pairs per group)

### 3.1 Noise on true links ✅
Rates = share of true links showing the pattern (target vs its S1 entity):

| Pattern | S2/US | S2/India | S3/US | S3/India |
|---|---:|---:|---:|---:|
| Raw name identical | 6% | 3% | 6% | 3% |
| Normalized name identical | 33% | 46% | 33% | 39% |
| Core name identical | 59% | 72% | 55% | 64% |
| Core words changed (typos / substitutions) | 28% | 22% | 27% | 25% |
| Core words added / dropped | 3% / 7% | 4% / 1% | 8% / 7% | 8% / 2% |
| Core words reordered | 3% | 1% | 3% | 1% |
| Legal form differs | 31% | 31% | 27% | 34% |
| Legal word moved to front (`LLC Moncada …`) | 2% | 4% | 2% | 4% |
| **Name in Indic script** | 0% | **23%** | 0% | **13%** |
| Domain-style name (`wilfordhancock.com`) | 5% | 4% | 5% | 4% |
| All-caps name / accented Latin (`Stúdios`) | 23% / 7% | 16% / 5% | 3% / 7% | 2% / 6% |
| Address empty / placeholder (`<NULL>`, `None`) | 5% / 4% | 4% / 3% | 5% / 4% | 4% / 3% |
| Address all-caps | 89% | 24% | 0% | 0% |
| Indic script inside address (state names) | 0% | 23% | 0% | 22% |
| Address token multiset identical | 41% | 22% | 48% | 30% |
| House numbers identical / dropped / changed | 61% / 15% / 7% | 73% / 4% / 3% | 70% / 13% / 6% | 69% / 6% / 3% |
| Street key (`<number> <street word>`) identical | 61% | 52% | 63% | 41% |

Source styles: S2 = uppercase USPS-like abbreviations (`ST`, `TRL`) with states as codes (US) or full/Indic names (India);
S3 = mixed case, **full state names** in the US (`Texas`) and **codes** in India (`MH`, `DL`, `KA`, `TG`).

### 3.2 Indic-script names are word-for-word transliterations ✅
- 551k links have an Indic-script target name (Devanagari, Bengali, Gujarati, Oriya, Tamil, Telugu, Kannada,
  Malayalam, Gurmukhi); **99.6% have exactly the S1 token count** — including legal words (`प्रा. लि.` = "Pvt. Ltd.").
- Rule-based transliteration alone (`baseline/er_text.py`) shares ≥1 core token with the S1 name for only **41%** of them
  (`kanstrakshan`, `intaranyasanal`, `praibhet`, `gret`).
- A **token rewrite table learned from training links** (position-aligned, ≥10 occurrences, ≥50% dominant; 540 rules,
  e.g. `praibhet→private`, `kanstrakshan→construction`, `lotas→lotus`, `sury→surya`) raises that to **~100%**, and
  recovers *all* S1 core tokens for 95.6%. Indian names come from a small shared vocabulary, so the table should
  transfer to the test split. (Rules whose token usually maps to itself, e.g. `lakshmi` vs `laxmi`, are excluded.)

### 3.3 Hard negatives are separated by the address, not the name ✅
**50.6% of true links belong to an S1 entity that has a same-core-name twin** in S1 (§2.4). Pairing a target with the
*twin* instead of its own S1:

| Similarity (mean) | True links | Same-name hard negatives | Random same-country |
|---|---:|---:|---:|
| Core-name token Jaccard | 0.77 | 0.79 | 0.003 |
| Core-name trigram Jaccard | 0.82 | 0.83 | 0.009 |
| Address word Jaccard | **0.80** | **0.007** | 0.003 |
| Address trigram Jaccard | 0.73 | 0.02 | 0.02 |
| House-number Jaccard | **0.74** | **0.006** | 0.005 |
| Street key equal | 56% | 0% | 0% |

→ Name similarity says "same kind of business", the **address decides which one**. Same-name twins live at unrelated
addresses, so address agreement is close to a hard requirement when the name is shared.
→ The hard part of true links is their own noise: 10% of true links have core-name token Jaccard ≤ 0.33 and 10% have
address word Jaccard ≤ 0.25 — these need the *other* field (and trigram/character similarity) to be recovered.

### 3.4 Which blocking signals survive the noise (share of true links) ✅
| Signal shared with the S1 entity | Coverage |
|---|---:|
| ≥1 core-name token (after transliteration map) | 92.0% |
| ≥2 core-name tokens | 79.4% |
| Identical concatenated core name | 64.8% |
| ≥1 house number | 78.4% |
| Core-name token **and** house number | 71.9% |
| Street key | 55.8% |
| ≥1 identifying address word | 93.5% |
| None of core token / number / address word | **0.02%** |

→ No single key is enough, but the union of name keys and address keys covers ~99.98% of links. Blocking must use
**several key families** and rank candidates, not rely on one exact key.

### 3.5 What the unlinked (distractor) records are ✅
| | Linked S2/S3 | Unlinked S2/S3 |
|---|---:|---:|
| Some S1 has the same core name | 64–69% | 30% |
| Some S1 has the same street key | 62–63% | 26% |
| **One S1 has both** (a "looks like a match" twin) | 32–38% | **0.9%** |
| Another S2/S3 record has the same core name + street key | 32–37% | 1.2% |
| Domain-style name / empty address | 4.5% / 4.5% | 0.3% / 0.3% |

→ Distractors are **independent single records of businesses absent from S1**, built from the same name and street
vocabulary (so they collide with S1 on name *or* address by chance, rarely both). They rarely come in clusters and carry
less of the domain/empty-address noise. Requiring name **and** address agreement removes almost all of them.

## 4. Design decisions for the training pipeline

Code: `baseline/` (`er_text.py` normalisation, `er_features.py` token/similarity expressions, `er_pipeline.py`
blocking/features/decision/metric, `train_baseline.py`, `predict_baseline.py`).

| # | Decision | Why (evidence) |
|---|---|---|
| D1 | Normalise both fields: Indic→Latin transliteration + **learned rewrite table**, accent stripping, lower-case, `&`→and, placeholders removed, legal forms and street/unit/direction words canonicalised, state names→codes, leading zeros stripped. Same rules for every country. | §3.1–3.2 noise; S2/S3 state-format mismatch; France must reuse the code path. |
| D2 | **Block within country** (equality, not a whitelist). | 0 cross-country links (§1.1). |
| D3 | Blocking = inverted index over six key families — core token (N), core-token pair (P), concatenated core (C), core token × house number (X), street key (S), identifying address word (A) — keeping keys seen in ≤ cap target records; score = Σ IDF of shared keys; keep **top-K per S1**. | No single signal covers >94% of links but their union covers 99.98% (§3.4); IDF favours rare, identifying keys. |
| D4 | Pair features: name token/trigram/containment similarity, legal-form agreement, address word/trigram/number/street-key agreement, emptiness, target flags (Indic, domain, source), blocking scores per family, and **competition features** (rank/ratio/margin of this pair within its S1's candidates and within the target's S1 suitors). | Names tie within same-name groups and the address decides (§2.4, §3.3); one S1 per target (§1.1). No country feature (France unseen). |
| D5 | Model: gradient-boosted trees (`sklearn` HistGradientBoosting; handles missing values natively). **2-fold cross-fitting by S1 entity** → every training pair gets an out-of-fold probability; test uses the average of both fold models. | Honest validation on all 2.2M S1s, including singletons, without a separate holdout. |
| D6 | Decision: (1) each S2/S3 record goes to **its highest-probability S1 only**; (2) keep pairs with p ≥ τ; τ chosen to maximise **macro F0.5 incl. singletons** on OOF predictions. | Exclusivity (§1.1); metric asymmetry favours precision (§1.4); singleton status only visible through candidate quality (§2.5). |
| D7 | Never touch the test split during development; `predict_baseline.py` is provided for the user to run. | User instruction; keeps validation honest. |

## 5. Results log

_Pending._

## 6. Open questions 🔎

_Pending._
