# Pipeline implementation report

Status: **run end to end on the full data on 2026-10-04** (RTX 5060 Laptop GPU, 16 GB RAM).
Both output files were written and passed the local format check
(`src/tools/check_submission.py`, because the official `utils/validate_submission.py` is not in this checkout).

## 0. Measured results
| | |
|---|---|
| Blocking recall, train (K = 32, GPU re-rank) | **0.9713** (7,419,426 / 7,638,365 links); recall@10 0.956, @20 0.967; 91.1% of S1s have every match covered |
| Train candidate pairs | 69.15M (31.3 per S1) |
| Training set | 32.87M rows: 7.42M positives, 25.45M negatives (97% hard) |
| XGBoost (2 folds, CUDA) | 1,500 rounds each, val logloss 0.0116, ~7 min per fold |
| **OOF macro F0.5 (train, cross-fitted, incl. singletons)** | **0.9720** at τ = 0.6975 |
| Singletons / non-singletons | 0.9711 / 0.9721 |
| Micro precision / recall | 0.9927 / 0.9354 (FP 52,190; FN 493,445, of which 218,939 blocking never proposed) |
| Test candidate pairs | 54.72M for 1,732,544 S1s (38 without candidates) |
| Test predictions | 5,730,818 links; 94.2% of S1s matched, 5.78% empty (train singleton rate 5.6%); no target assigned twice |

The OOF score is measured on training entities with cross-fitting. The test split was never used
for tuning.

### Measured wall time and resources
| Stage | Train | Test | Peak RSS* | Peak VRAM |
|---|---|---|---|---|
| 01 normalize | ~10 min (earlier run) | 3 min | 1.4 GB | 0 |
| 02 labels + rewrite map | 0.6 min | — | — | 0 |
| 03 records + index | 14 min | 13 min | 3.2 GB | 0 |
| 04 candidates + GPU re-rank | 40 min | 28 min | 2.1 GB | 1.9 GB |
| 05 features (GPU) | 12 min | 27 min† | 4.9 GB | 0.5 GB |
| 06 training set | 2 min | — | 1.4 GB | 0 |
| 07 XGBoost, 2 folds | 14 min | — | **5.9 GB** | 3.2 GB (device total) |
| 08 OOF / 09 τ / 10 bundle | 4 min | — | 1.0 GB | 0.3 GB |
| 11 predict + write + check | — | 24 min | 1.7 GB | 0.3 GB |

\* RSS includes pages of the memory-mapped record files, which the OS can evict; in 03 and 05
most of the reported RSS is mapped record pages.
† Slowed by sharing the GPU with the neural experiment; train-side speed was 165k pairs/s.

**Over the estimate:** stage 07 held about 5.9 GB RSS, because XGBoost builds the
`QuantileDMatrix` on the host from numpy batches (cupy is not installed, so it cannot be built on the
device) and keeps it alongside the device copy. Available RAM dipped to ~1.7 GB during that stage.
The pre-implementation estimate of 0.5 GB was wrong. If RAM is tighter, set `--entity-frac 0.6` in
stage 06 or lower `MAX_GPU_MEMORY_GB` so 07 subsamples whole entities.

### Changes made during the run
- Blocking recall was 0.927 with IDF-sum ranking. Stage 04 now re-ranks every aggregated candidate
  on the GPU (`utils/rerank.py`), and the N/A key cap was raised from 300 to 1000.
- Stages record an input signature and clear outputs built from different inputs, so a rerun can no
  longer reuse stale candidate shards.
- The runner flushes output line by line and writes UTF-8 to the console. A Hindi example line had
  crashed it under the Windows code page.
- xgboost 3.4.1 (CUDA build) was installed and pinned.

Evidence base: `problem.md`. The `analysis/` and `baseline/` folders it references are not in this
checkout, so every rule implemented here follows the decisions D1–D7 and the numbers written in
problem.md.

## 1. Files created (nothing existing was modified)
```
code/business_entity_resolution/
  README.md, requirements.txt
  docs/RESOURCE_ESTIMATES.md      pre-implementation estimates (per component, with arithmetic)
  docs/PIPELINE_REPORT.md         this file
  src/utils/config.py             defaults / env overrides
  src/utils/io.py                 paths, streaming TSV reader, atomic parquet writer, manifests
  src/utils/gpu.py                RAM/VRAM reports, check_ram, device + VRAM cap, xgboost probe
  src/utils/normalization.py      Indic->Latin, basic + canonical normalisation, state tables
  src/utils/records.py            265-byte fixed-width record dtype + builder
  src/utils/blocking.py           key families, partitioned CSR build, chunked top-K generation
  src/utils/features.py           GPU kernels (sets, trigram Dice, Levenshtein) + competition features
  src/utils/metrics.py            per-S1 F0.5, exact threshold sweep, summary
  src/utils/decision.py           target exclusivity (streaming)
  src/utils/model.py              XGBoost DataIter/QuantileDMatrix trainer, HGB fallback, predict
  src/training/00_resource_check.py … 11_predict_test.py
```

## 2. Pipeline flow
| Stage | Input | Output | Device |
|---|---|---|---|
| 01 normalize | raw TSV (200k-row Arrow batches) | `work/<split>/normalized/*.parquet`, id lookup arrays | CPU |
| 02 labels + rewrite map (train) | GT TSV, 01 output | `owner.npy`, `n_true.npy`, `s1_bucket.npy`; `artifacts/rewrite_map.json` | CPU |
| 03 records + index | 01 output, rewrite map | `records/{s1,tgt}.npy`; CSR index | CPU |
| 04 candidates | index | `candidates/part-*.parquet` (top-K per S1), recall report | CPU |
| 05 features | candidates, records | `features/part-*.parquet` (64 float32 features) | **GPU** |
| 06 training set (train) | features | `trainset/part-*.parquet` | CPU |
| 07 train | trainset | `work/models/fold{0,1}.*` | **GPU** (XGB) / CPU (HGB) |
| 08 OOF (train) | features, models | `oof/part-*.parquet` | **GPU** (XGB) |
| 09 threshold | OOF, labels | `artifacts/threshold.json` | CPU |
| 10 bundle | models, threshold, rewrite map | `artifacts/final/` | — |
| 11 predict (test) | test features, bundle | `output/matching_results.tsv`, `output/candidate_pairs.tsv` | **GPU** (XGB) |

## 3. GPU vs CPU
**GPU**
- Stage 05 pair kernels, in batches of `BATCH_SIZE=65,536` pairs, ≈ 0.6–1.2 GB VRAM:
  - token-set intersections by broadcast equality (core 6×6, name 8×8, address 16×16, house numbers, street keys);
  - trigram Dice on core names (30×30) and order-free address strings (62×62);
  - batched Levenshtein (row recurrence plus `torch.cummin`, 32 or 64 steps).
- Stage 07 XGBoost `hist` on `device=cuda`.
- Stages 08 and 11 XGBoost `inplace_predict` in 1M-row batches.

**CPU, and why**
- TSV parsing and normalisation: regexes, Unicode tables and branchy per-character transliteration.
- Inverted-index sort/unique: about 1 s per 8M keys on CPU, so the GPU would mostly add transfers.
- Candidate expansion: variable-length gathers plus segmented sort. Padding to the worst key frequency
  on a GPU is exactly the memory blow-up to avoid.
- Training-set filtering, competition passes (`ufunc.at` over target-sized arrays) and the
  threshold sweep: each is one pass or one sort, about seconds on CPU.
- Text embeddings: **not used**. The hard negatives are same-name twins separated by the address
  (§3.3), and the cost is about 19 GB of embeddings per split (see RESOURCE_ESTIMATES 2.8).

## 4. Estimated RAM / VRAM per stage
See RESOURCE_ESTIMATES §3. The peak RSS of every stage is ≤ 1 GB except the HGB fallback (≈ 2.5 GB).
Memmapped records and indexes add evictable page cache. Peak VRAM ≈ 1.5 GB (XGBoost training).
Each stage calls `check_ram()` with its own estimate before starting and logs RSS, available RAM
and VRAM allocated/peak on every progress line.

## 5. Batch sizes and limits
| Knob | Default | Where |
|---|---|---|
| `CHUNK_ROWS` | 200,000 records | 01, 02, 03 |
| `INDEX_PARTITIONS` | 16 hash-range partitions | 03 |
| `MAX_KEY_FREQUENCY` | 1000 (families P, C, X, S) | 03 |
| `MAX_KEY_FREQUENCY_COMMON` | 300 (families N, A) | 03 |
| `MAX_CANDIDATES_PER_S1` (K) | 32 (true max is 11 matches) | 04 |
| `MAX_EXPANDED_POSTINGS` | 8M postings per sub-chunk (~0.4 GB) | 04 |
| `S1_PER_SHARD` | 50,000 S1 (≤ 1.6M pairs per shard) | 04–11 |
| `BATCH_SIZE` | 65,536 pairs (halves on CUDA OOM) | 05 |
| `MAX_GPU_MEMORY_GB` | 6.0 of 8 GB (per-process fraction) | 05, 07 |
| `TRAIN_ITER_ROWS` | 500,000 rows per DataIter batch | 07 |
| `PREDICT_BATCH_ROWS` | 1,000,000 | 08, 11 |
| `N_THREADS` | 4 | all |

## 6. Normalisation (D1)
- **Basic (01):** HTML entities; Brahmic → Latin, with one offset table for the 9 scripts, schwa
  deletion at word end, and nukta/chillu/virama handling; NFKD accent stripping; lower-case;
  `&` → `and`; dots removed in names (`l.l.c` → `llc`); domain names reduced to their label; placeholder
  addresses and parts (`<NULL>`, `None`, `null`, …) removed; flags for Indic name/address, domain,
  all-caps, accented, empty and placeholder.
- **Canonical (03):**
  - the learned rewrite table (Indic names only);
  - legal forms canonicalised (`limited` → `ltd`, `private` → `pvt`, … ; `public` only before `ltd`) and
    removed from the core name anywhere (handles `LLC Moncada …`);
  - street, direction and unit words canonicalised, with unit designators dropped;
  - house numbers stripped of leading zeros, with ordinals excluded;
  - street key = number + next word within a comma part;
  - state taken from any comma part, scanning from the end, via the US/India alias tables plus the
    learned Indic state-part map. An unknown country gets no state, and nothing is filtered.
- **Rewrite map (02):** position-aligned token pairs from training links whose target name is Indic
  and has the same token count, kept at count ≥ 10 and dominant share ≥ 0.5, excluding tokens that
  usually map to themselves. The same thresholds learn the Indic address-part → S1-state map.
  Only training data is used. Stage 03 refuses a map whose `learned_from` is not `train`.

## 7. Blocking (D2, D3)
There are six key families: N core token, P core-token pair (from the first 4 core tokens), C concatenated core,
X core token × house number, S street key, and A identifying address word. Each key is hashed as
`crc32 ⊕ crc32'` (64-bit) of `country | family | value`.

Keys with document frequency above the family cap are dropped. Score = Σ IDF of shared keys, with
IDF = log(1 + N_targets / df). The top-K per S1 are kept (ties go to the lower target index).

Stage 04 reports candidate count, candidates per S1 (mean, median, p95, p99, max, zero), recall over
all GT links, recall@{1, 2, 3, 5, 10, 20}, the share of S1s with every match covered, positives per
family and the expansion histogram. It **exits non-zero below `--min-recall 0.97`**.

## 8. Features (D4): 64 columns, all float32, NaN = not computable
- **Name (14):** core-token Jaccard, containment and intersection; full-name-token Jaccard; core
  trigram Dice; core Levenshtein similarity; concatenated-core equality; first-core-token equality;
  core counts for each side and their absolute difference; legal equal, both present, missing on one side.
- **Address (18):** token Jaccard, containment and intersection; order-free trigram Dice and
  Levenshtein; house-number Jaccard, intersection, conflict, and missing on either side; street-key
  match and conflict; state equal, conflict, missing on one side; token counts; target address empty.
- **Flags (10):** source S3; target Indic name, Indic address, domain, all-caps, accented, legal-in-front,
  placeholder address, landmark; S1 landmark.
- **Blocking (9):** score, number of shared keys, rank, and one membership bit per family.
- **Heuristic (3):** `name_best = max(core-token Jaccard, core trigram Dice)`, `addr_best` likewise,
  `h_score = (name_best + addr_best) / 2`. NaN counts as 0 here.
- **Competition (10):** computed from base features only, never labels, and identically for train and test.
  - `s1_ncand`: number of candidates of this S1.
  - `s1_rank_h`: rank of h within the S1 (ties broken by blocking score, then target index).
  - `s1_h_margin` / `s1_blk_margin`: x minus the max of x over the *other* candidates of the same S1
    (equals x when there is no other candidate; 0 for tied bests).
  - `s1_h_ratio` / `s1_blk_ratio`: x divided by the S1's max.
  - `t_nsuitors`: number of S1s that have this target as a candidate.
  - `t_h_margin` / `t_blk_margin`: x minus the max over the target's *other* S1 suitors, from streaming
    best, best multiplicity and strict second-best arrays.
  - `t_is_best_h`: 1 when this S1 has the target's highest h.

  **Leakage note:** target-side features span folds, because a target's suitors can be S1s in the
  other fold. They use only feature values, so no label crosses folds.

## 9. Training-set strategy (06)
- Keep every blocked positive.
- Keep hard negatives: blocking rank < 6, h-rank < 6, h ≥ 0.5, or the S1 is the target's best h-suitor.
- Keep easy negatives at a 2% rate selected by a **deterministic hash of (s1, tgt)**, weighted 50.
- Optionally keep a fraction of whole S1 entities (`--entity-frac`).
- Reported: positive and negative counts, ratio, hard share, the hard categories (same-name
  competitor, address-conflict, high-rank), discarded rows and disk size.
- Expected size ≈ 21M rows: 7.5M positives and 13M negatives, ≈ 93% of them hard.

## 10. Model configuration (07)
- **XGBoost:** `hist`, `device=cuda`, `max_bin=256`, `max_depth=8`, `eta=0.08`, `subsample=0.8`,
  `colsample_bytree=0.8`, `min_child_weight=5`, `lambda=1`, up to 1500 rounds with early stopping at
  100 on the fold's es_val entities, `nthread=4`. Data flows through a streaming `DataIter` into a
  `QuantileDMatrix`. If the estimated VRAM exceeds 80% of `MAX_GPU_MEMORY_GB`, whole S1 entities are
  subsampled.
- **Fallback:** sklearn HistGradientBoosting (`max_iter=600`, `lr=0.1`, 63 leaves, 255 bins,
  `l2=1`, early stopping on the es_val entities), with an entity-level row cap of 3M.

## 11. Cross-fitting
fold = crc32(S1 entity_id) % 1000 % 2, and es_val = (bucket // 2) % 20 == 0 (about 5% of each fold,
for early stopping only). All pairs of an S1 share its fold. Model f trains on the other fold. Stage 08
scores **every** candidate pair (not just the sampled rows) with the model that did not see its S1.
The test score is the mean of the two fold models.

## 12. Exclusivity and threshold (D6)
- Each target goes only to its highest-probability S1 (ties go to the lowest S1 row), and that pair is
  then kept if p ≥ τ. The same code runs in 09 and 11, and 11 asserts that no target is assigned twice.
- τ maximises **macro F0.5 per S1 including singletons** over OOF decision candidates, using an exact
  O(W log W) sweep over every distinct probability. n_true counts every GT link, including those blocking
  missed. Among tied optima the highest τ is chosen, and the floor is τ ≥ 0.05.
- Reported at τ: macro F0.5 overall, for singletons and for non-singletons; micro precision and recall;
  predicted links; TP, FP and FN; zero-match S1s predicted empty versus falsely merged; the
  predictions-per-S1 histogram; and a 0.05-step sweep grid.
- Saved: `artifacts/threshold.json` and `threshold_sweep_full.npz`.

## 13. Checkpoint / resume
- 01 and 04–11 write one file per part or shard, atomically (`.tmp` → rename), and skip finished parts.
- 03 writes `_records_done.json` after the record and key pass, so a failed index build restarts at the CSR step.
- Each stage ends with a `_MANIFEST.json` containing its statistics. Downstream stages refuse to start
  without the upstream manifest. `--force` rebuilds a stage.

## 14. Known limitations / open points
1. **`analysis/` and `baseline/` are missing**, so numbers such as the 540-rule rewrite table or
   100% token recovery are problem.md's figures for *their* transliterator. This transliterator is new.
   Stage 02 prints the rules it learns, and stage 03 prints the rewrite coverage, so the user can compare.
2. **xgboost is not installed.** Without it, training falls back to CPU HGB on ≤ 3M rows (weaker),
   and scikit-learn is BSD-3, while the rules ask for MIT/Apache-2.0. Install xgboost
   (Apache-2.0) for the final submission and confirm the GPU with `00_resource_check.py --probe-xgboost`.
   Whether a given wheel ships sm_120 (RTX 50-series) kernels is not verified here.
3. Test-time key document frequencies and competition features are computed on the test candidates
   themselves. This is unsupervised inference with no labels and no tuning, but it does mean IDF reflects
   the test pool.
4. τ is tuned on single-model OOF scores and applied to the mean of two models. Averaging slightly
   compresses scores. `--model-set full` (after `10_train_final.py --refit-full`) is available but
   not tuned.
5. French addresses get no state extraction (there's no table and no training data). Legal-form lists
   come from training-era (US/India) forms only, and no test-derived rule was added.
6. Key caps and K are defaults from estimates, not measurements. Check the stage 04 recall@k and
   expansion statistics before training. If recall is below 0.97, the stage stops.
7. Runtime is not measured. The Python normalisation in 01 and 03 is expected to take 10–20 min per split.
8. I looked at the first 5 lines of each test file only to confirm the schema. No design decision uses them.

## 15. Commands
One command: `python run_pipeline.py` (each stage in its own process, logged, stops on the first
failure, resumable; `--dry-run` lists the commands). Per-stage commands are in README.md. Run order: 00, then on train 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, then on test
01, 03, 04, 05 with `--split test`, then 11, then the official validator.
