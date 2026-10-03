# Resource estimation and design (written before implementation)

Target machine: RTX 5060 Laptop GPU (8 GB, 8151 MiB reported by `nvidia-smi`), 16 GB RAM of which
**~5 GB is treated as the working budget** (at inspection time `psutil` reported only 3.0 GB free, so
every default below aims at a **≤ 3.5 GB peak process RSS**), 16 logical CPUs, Windows, Python 3.12.10.

Installed and relevant: numpy 2.3.5, pyarrow 24.0.0, scikit-learn 1.9.0, torch 2.11.0+cu128
(`torch.cuda.is_available() == True`, CUDA 12.8, which supports the Blackwell sm_120 GPU), psutil 7.2.2.
**Not installed:** xgboost, lightgbm, rapidfuzz, unidecode, indic-transliteration, numba, cupy.
Nothing is installed by this work. GPU XGBoost is used if the user installs it and
`00_resource_check.py` confirms it was built with CUDA. Otherwise sklearn HistGradientBoosting runs
on the CPU with a row cap.

Evidence source: `problem.md` (sections 1–4, decisions D1–D7). The `analysis/` and `baseline/`
scripts it cites are **not present in this checkout**, so every number below that is quoted from
the analysis comes from `problem.md`. Sizes come from file metadata (`wc -l`, `ls -l`).

## 0. Input sizes (file metadata)

| File | Bytes | Data rows |
|---|---:|---:|
| train_source1.tsv | 210,069,713 | 2,206,821 |
| train_source2.tsv | 489,301,488 | 5,034,616 |
| train_source3.tsv | 503,705,637 | 5,285,603 |
| train_ground_truth.tsv | 127,015,583 | 2,206,821 (7,638,365 links) |
| test_source1.tsv | 175,022,086 | 1,732,544 |
| test_source2.tsv | 509,456,422 | 4,887,273 |
| test_source3.tsv | 506,002,772 | 5,082,316 |

Targets (S2+S3): train **10,320,219**, test **9,969,589**. Average row ≈ 97 bytes.

## 1. Architecture

```
raw TSV (streamed, 200k-row Arrow batches)
 └─01 basic normalisation (CPU, Python per record)        → work/<split>/normalized/*.parquet
     Indic→Latin transliteration, accents, case, &→and, placeholders, domain/caps/legal-front flags,
     numeric ids, per-source row index
 └─02 labels + learned rewrite map (train only, CPU)      → artifacts/rewrite_map.json,
     owner[target] = S1 index or −1, n_true[S1], fold[S1]   work/train/labels/*.npy
 └─03 canonicalisation + record matrix + blocking index   → work/<split>/records/{s1,tgt}.npy (memmap)
     rewrite map, legal forms, street/unit/direction,       work/<split>/index/* (CSR, memmap)
     states, house numbers, street keys; 6 key families
     hashed with the country (equality, no whitelist)
 └─04 candidate generation (CPU, numpy, S1 chunks)        → work/<split>/candidates/part-*.parquet
     CSR lookup, Σ IDF score, family bits, top-K per S1;
     train: recall report and hard failure below a floor
 └─05 pair features (GPU, torch batches) + competition    → work/<split>/features/part-*.parquet
     features (streaming per-target best/second arrays)
 └─06 training set (hard negatives + hashed easy sample)  → work/train/trainset/part-*.parquet
 └─07 fold models (entity-level 2-fold, GPU XGBoost or CPU HGB) → work/models/fold*.{ubj,joblib}
 └─08 OOF probabilities on ALL train candidates           → work/train/oof/part-*.parquet
 └─09 target exclusivity + exact macro-F0.5 sweep         → artifacts/threshold.json
 └─10 final bundle (fold ensemble, optional full refit)   → artifacts/final/
 └─11 test: predict (fold average) → exclusivity → τ      → output/matching_results.tsv,
                                                            output/candidate_pairs.tsv
```

All per-record data lives in fixed-width numpy record arrays (memory-mapped). Pairs are only ever
`(s1_index:int32, target_index:int32)`; strings are never joined per pair.

## 2. Per-component estimates

Notation: N1 = S1 rows, NT = target rows, K = `MAX_CANDIDATES_PER_S1` (default 32), F = feature
count (64), P = candidate pairs. Upper-bound pair count P ≤ N1·K: **train 70.6M, test 55.4M**.

### 2.1 Data loading
| | |
|---|---|
| Input | 1.33 GB (train), 1.19 GB (test) |
| Records | 12.53M train, 11.70M test |
| Pairs | — |
| RAM | Arrow batch of 200k rows × ~97 B ≈ 19 MB raw; Arrow + Python strings ≈ 3–4× → **≤ 80 MB** |
| VRAM | 0 |
| Disk | none (feeds 01) |
| Complexity | O(bytes), single pass |
| GPU? | No. Parsing is I/O and branch-bound. |
| Batch | `CHUNK_ROWS=200000` |
| Over budget | Lower `--chunk-rows`; memory scales linearly. |

Only one source file is open at a time. pandas is not used.

### 2.2 Normalisation (01)
| | |
|---|---|
| Input | the stream above |
| Records | 12.53M (train) |
| RAM | per chunk: 200k × (2 output strings ≈ 110 B + flags) ≈ 25 MB plus Python objects ≈ **≤ 150 MB** |
| VRAM | 0 |
| Disk | ≈ 120 B/row uncompressed → 1.5 GB; zstd parquet ≈ **0.5 GB** per split |
| Complexity | O(characters); ~20–40 µs per record in Python → **~5–9 min per split** |
| GPU? | No. Regexes, Unicode tables and per-character transliteration don't map to tensors. |
| Batch | 200k rows per parquet part (resumable per part) |
| Over budget | Smaller chunks. Never needs more than one chunk in memory. |

### 2.3 Labels + learned Indic rewrite map (02, train only)
| | |
|---|---|
| Input | GT 127 MB streamed, normalized parquet (only `name_basic`, ids, flags columns) |
| Records | 7,638,365 links; ~551k with an Indic-script target name (problem.md §3.2) |
| RAM | id lookup arrays: N1 × 8 B + NT × 8 B = 17.7 + 82.6 MB, plus sort permutations ≈ **0.2 GB**; owner array NT × 4 B = 41 MB; S1 `name_basic` column ≈ 2.2M × ~35 B ≈ 80 MB; Indic target names ≈ 0.75M × ~40 B = 30 MB; aligned-token counter ≈ 2M pairs → ≤ 100k distinct ≈ 20 MB. **Peak ≈ 0.5 GB** |
| VRAM | 0 |
| Disk | owner.npy 41 MB, n_true.npy 2.2 MB, s1_fold.npy 2.2 MB, rewrite_map.json < 1 MB |
| Complexity | O(links · log NT) for `searchsorted` id resolution |
| GPU? | No. Counting string pairs is dictionary work. |
| Batch | GT read in 200k-row Arrow batches |
| Over budget | Cannot realistically exceed. |

### 2.4 Canonicalisation, record matrix, blocking-index construction (03)
Record row = 265 bytes (32 core-name chars, 64 sorted-address chars, 6 core-token + 8 name-token +
16 address-token + 4 house-number + 3 street-key int32 hashes, 3 scalar hashes, 7 counts, 16-bit
flags).

| | |
|---|---|
| Records | N1 2.21M → **585 MB**; NT 10.32M → **2.73 GB** (both on disk, written chunk-wise into `open_memmap`) |
| Keys | estimated ≈ 13 keys/record (N 2.3, P 1.8, C 1, X 2.8, S 1, A 4); hard cap 27/record. Targets: 134M expected, 279M upper |
| RAM, key spill | keys written per chunk to 16 hash-range partition files: per chunk 200k × 27 × 12 B = **65 MB** |
| RAM, index build | per partition 134M/16 = 8.4M postings (upper 17.4M) × 32 B (key, idx, argsort, sorted copies) = **270 MB (upper 560 MB)** |
| VRAM | 0 |
| Disk | temp key files 134M × 12 B = 1.6 GB (upper 3.3 GB, deleted after build). Final CSR: unique keys × 21 B + postings × 4 B ≈ 0.8–1.2 GB. S1 keys: 2.2M × 13 × 12 B = 0.34 GB |
| Complexity | O(records · keys) generation; O(postings log postings) sort per partition |
| GPU? | No. Inverted-index construction is sort/unique on 64-bit keys, and argsort over 8M keys takes about 1 s on CPU. Moving it to the GPU would add transfers and VRAM pressure for little gain. |
| Batch | 200k records, 16 partitions (`--partitions`) |
| Over budget | Double `--partitions`; peak halves. |

Hash ranges are taken from the top bits of the 64-bit key, so concatenating partitions in order gives
one globally sorted key array without a merge step.

### 2.5 Candidate generation (04)
| | |
|---|---|
| Lookups | 2.2M S1 × ~13 keys = 29M `searchsorted` calls on memory-mapped sorted keys |
| Expanded postings | per S1 Σ df over its keys. Caps: `MAX_KEY_FREQUENCY=1000` (P, C, X, S) and `MAX_KEY_FREQUENCY_COMMON=300` (N, A). Expected ≈ 700/S1 → **≈1.5B per split** (realistic upper 8.9k/S1) |
| RAM | chunks are cut by exact expansion size ≤ `MAX_EXPANDED_POSTINGS=8M`: 8M × (int64 pair key 8 + float32 w 4 + uint8 fam 1) = 104 MB, ×3 for argsort and reductions ≈ **0.35 GB**, plus memmapped index pages (OS-managed and evictable) |
| VRAM | 0 |
| Disk | output P × 14 B (s1, tgt, score, nkeys, fam, rank) = **≤ 1.0 GB train / 0.8 GB test** |
| Complexity | O(E log E) per chunk, ≈ 190 chunks × ~2–3 s ≈ **8–10 min** at the expected E |
| GPU? | No. The work is variable-length gather plus segmented sort. A GPU version would need padding to the worst key frequency, which is the memory blow-up we want to avoid. |
| Batch | `S1_PER_SHARD=50000` per output shard (≤ 1.6M pairs), split internally by expansion size |
| Over budget | Lower caps or `MAX_EXPANDED_POSTINGS`. One S1 can never exceed 27 × 1000 = 27k postings. |

Train-only checks: blocking recall over all 7.64M GT links, recall@k for k ∈ {1, 2, 5, 10, 20, K},
share of S1s with every match captured, and candidates-per-S1 median/p95/p99/max. The stage **exits
non-zero if recall < `--min-recall` (default 0.97)**.

### 2.6 Candidate-pair storage
Parquet shards, zstd, about 44 shards for train. `candidate_pairs.tsv` for test is written by
streaming shards in S1 order: 55.4M ids × ~13 chars ≈ 0.75 GB on disk, written in ≤ 50k-S1 slices
(≤ 40 MB RAM each).

### 2.7 Pair features (05, GPU)
Per batch of B pairs, gather both records from the memmaps on CPU (B × 2 × 265 B; 35 MB at B = 65,536),
copy pinned → CUDA, then compute:

| Tensor | Shape | Bytes at B = 65,536 |
|---|---|---:|
| address-trigram equality (bool) | B × 62 × 62 | 252 MB |
| core-trigram equality | B × 30 × 30 | 59 MB |
| address-token equality | B × 16 × 16 | 17 MB |
| Levenshtein DP rows (int16/32) | B × 65 × 2 | 34 MB |
| inputs, intermediates, outputs (B × 64 float32 = 17 MB) | | ≈ 150 MB |
| **Peak** | | **≈ 0.6 GB** (≈ 1.2 GB with PyTorch allocator slack) |

| | |
|---|---|
| Pairs | ≤ 70.6M train, ≤ 55.4M test |
| RAM | batch gather 35 MB + output 16 MB + pyarrow writer row group; page cache for the 2.73 GB target memmap is OS-managed. **Process RSS ≈ 0.5–0.8 GB** |
| VRAM | ≈ 0.6–1.2 GB at B = 65,536; capped by `torch.cuda.set_per_process_memory_fraction(MAX_GPU_MEMORY_GB / total)`; B halves automatically on CUDA OOM |
| Disk | P × 64 × 4 B = **18.1 GB raw train (≈ 7–9 GB zstd)**, 14.2 GB raw test. Base shards are deleted after the competition pass unless `--keep-base` (544 GB free on C:) |
| Complexity | O(P · (L_a·L_b)) elementwise, fully parallel |
| GPU? | **Yes.** Set intersections by broadcast equality, trigram Dice and banded Levenshtein (32 or 64 DP steps with `torch.cummin`) are dense, uniform, elementwise work on fixed-width arrays. Doing this in Python on CPU would cost ~50–100 µs per pair, about 1.5–2 h for 70M pairs, versus minutes on the GPU. |
| Batch | `BATCH_SIZE=65536` |
| Over budget | OOM handler halves B and retries. `--device cpu` runs the same torch code on CPU (slow but correct). |

Competition features (label-free, same code for train and test):
- S1 side: S1 groups are contiguous in a shard, so rank, best and second-best are computed in-shard with numpy.
- Target side: two streaming passes keep arrays of size NT (best h, second h, best blocking score,
  second blocking score, suitor count): 10.32M × 20 B = **206 MB**.

### 2.8 Optional text embeddings: **not implemented (decision)**
Cost if added (e.g. a 22M-parameter MiniLM-class Apache-2.0 encoder, 384-d, fp16):
- Storage: 12.53M records × 384 × 2 B = **9.6 GB per text field per split** (19 GB for name + address).
- VRAM: weights 44 MB + activations at batch 512 × 32 tokens ≈ 0.3 GB, so it fits.
- Time: ~3–5k records/s on a laptop GPU, about 45–70 min per field per split.
- It also needs `sentence-transformers` (not installed) and a model download.

Value: problem.md §3.3 shows the hard negatives are same-name twins that a name already matches
(core-name Jaccard 0.79 on hard negatives vs 0.77 on true links). They are separated by the address
(word Jaccard 0.80 vs 0.007; house numbers 0.74 vs 0.006). A semantic name embedding mostly encodes
"same kind of business", which is the signal that does *not* separate twins. The noise that breaks
lexical matching (typos, abbreviations, transliteration) is handled by trigram, Levenshtein and the
learned rewrite map. The resource cost is not justified now. If OOF error analysis later shows
semantic misses, the hook is a per-record fp16 memmap plus a dot product inside the same 05 batch.

### 2.9 GPU similarity computation
This is 2.7 above. There is no all-pairs comparison anywhere: similarity is computed only for the
≤ K blocked candidates per S1.

### 2.10 Training-set construction (06)
Rows kept per S1 (expected): positives found by blocking ≈ 3.4; hard negatives (blocking rank < 6,
heuristic rank < 6, h ≥ 0.5, or the pair is the target's best suitor) ≈ 5–6; easy negatives at a
hashed rate of 2% with weight 50 ≈ 0.5.

| | |
|---|---|
| Rows | ≈ 2.2M × 9.5 ≈ **21M** (≈ 7.5M positives, ≈ 13M negatives, of which ≈ 93% hard) |
| RAM | streaming per shard: ≤ 1.6M rows × 70 columns × 4 B = 0.45 GB, filtered in place |
| VRAM | 0 |
| Disk | 21M × (64 × 4 + 16 meta) B = 5.7 GB raw, ≈ 2.5 GB zstd |
| Complexity | O(P) single pass |
| GPU? | No. This is a filter. |
| Over budget | `--entity-frac` (deterministic hash on the S1 id), lower `--hard-rank`, lower `--easy-rate` |

### 2.11 Model training (07)
**GPU XGBoost path** (`tree_method=hist`, `device=cuda`, `max_bin=256`, `max_depth=8`, `eta=0.08`,
`subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, up to 1500 rounds with early stopping
at 100, `nthread=N_THREADS=4`). The data is fed through `xgboost.DataIter` into a **`QuantileDMatrix`**,
so the float matrix is never materialised: shards stream in 500k-row batches and are quantised to
≤ 1 byte per value on the device.
- Per fold ≈ 10.5M rows × 64 features × 1 B (ELLPACK) = **0.67 GB**
- Gradients, hessians, labels, weights and prediction cache ≈ 10.5M × 24 B = 0.25 GB
- Histograms: 2^8 nodes × 64 × 256 bins × 16 B ≈ 67 MB
- **VRAM ≈ 1.0–1.5 GB. Host RSS ≈ 0.5 GB** (one batch of 120 MB plus sketches)

If the estimate exceeds `MAX_GPU_MEMORY_GB`, the trainer subsamples whole entities until it fits.
ExtMemQuantileDMatrix is the next step if the user ever needs every row; it isn't required at this scale.

**CPU fallback** (sklearn HistGradientBoosting): it converts X to float64, so rows are capped at
`--max-train-rows=3,000,000` (entity-level subsample): 3M × 64 × 8 B = **1.54 GB** + binned uint8
0.19 GB + the float32 source 0.77 GB, freed after conversion. **Peak ≈ 2.5 GB**, `max_iter=600`
with early stopping on 10% of entities.

### 2.12 OOF prediction (08)
Every train candidate (≤ 70.6M) is predicted by the fold model that did not see its S1.
- RAM: 1M-row batch × 64 × 4 B = 256 MB
- VRAM: the same 256 MB plus the model
- Disk: oof P × 13 B = 0.9 GB
- GPU? Yes for XGBoost (`inplace_predict` on CUDA); the CPU fallback runs HGB `predict_proba` in batches.

### 2.13 Threshold tuning (09)
Exclusivity is streaming: pass 1 `np.maximum.at(best_p, tgt, p)` over NT-sized arrays (41 MB);
pass 2 keeps winners (ties go to the lowest S1 index via `np.minimum.at`). Winners ≤ NT = 10.3M ×
(p 4 B, s1 4 B, y 1 B) = 93 MB.

The **exact sweep** sorts winners by p descending, computes each pair's change to its S1's F0.5
(tp, pred counts within the S1), and takes a cumulative sum. Macro F0.5 at every distinct threshold
costs O(W log W) with **≈ 0.4 GB peak**. The base is S1s with n_true = 0, which score 1.0 with
empty output. S1s whose true links were missed by blocking keep n_true in the denominator, so the
score is honest.

GPU? No. This is a single sort over ≤ 10M rows, about 1 s on CPU.

### 2.14 Final test inference (11)
Upstream stages 01, 03, 04 and 05 with `--split test` have the same per-stage costs scaled by 0.93.
Prediction is the average of the 2 fold models → 2 × 55.4M row-predictions in 1M-row batches
(256 MB RAM and VRAM). Exclusivity uses the same NT arrays (40 MB). Writers stream per shard.
**Peak ≈ 1 GB.**

## 3. Summary table

| Stage | Peak RAM (est.) | Peak VRAM | Temp disk | Device |
|---|---:|---:|---:|---|
| 01 normalise | 0.15 GB | 0 | 0.5 GB | CPU |
| 02 labels + rewrite map | 0.5 GB | 0 | 0.05 GB | CPU |
| 03 records + index | 0.6 GB (0.9 upper) | 0 | 3.3 GB records + 1.6–3.3 GB temp keys | CPU |
| 04 candidates | 0.5 GB + page cache | 0 | 1.0 GB | CPU |
| 05 features | 0.8 GB + page cache | 0.6–1.2 GB | 7–9 GB (×2 transient) | **GPU** |
| 06 training set | 0.6 GB | 0 | 2.5 GB | CPU |
| 07 train (XGB) | 0.5 GB | 1.0–1.5 GB | < 50 MB | **GPU** |
| 07 train (HGB fallback) | 2.5 GB | 0 | < 50 MB | CPU |
| 08 OOF | 0.5 GB | 0.3 GB | 0.9 GB | **GPU** (XGB) |
| 09 threshold | 0.4 GB | 0 | — | CPU |
| 11 test predict + write | 1.0 GB | 0.3 GB | 0.8 GB outputs | **GPU** (XGB) |

## 4. Bottlenecks
1. **Random gather from the 2.73 GB target memmap in 05.** If page cache can't hold it (less than about
   3 GB free), each pair is one 4 KB random read. Worst case on NVMe at ~100k IOPS is 70M / 100k ≈
   12 min, which is slow but safe because memmap pages are evictable and never cause an OOM. The row
   layout keeps it to one page touch per pair.
2. **Python per-record normalisation** in 01/03 (≈ 10–20 min per split). Single process on purpose:
   multiprocessing would duplicate the interpreter and buffers for each worker.
3. **Expansion volume in 04** is driven by the key-frequency caps. The printed expansion statistics
   tell the user whether to tighten them.

## 5. Memory-safety strategy (what the code enforces)
- No full-table pandas loads. All reads are Arrow streaming batches or column-pruned parquet row groups.
- Per-record state lives only in memmaps; per-pair state lives only in parquet shards.
- Fixed-width arrays (uint8/int32/float32), with no Python lists of records beyond one chunk.
- No Cartesian products. The largest transient pair structure is bounded by `MAX_EXPANDED_POSTINGS`.
- `check_ram()` runs before every stage with a per-stage estimate and fails with a clear message
  unless `--force`. Each stage logs RSS, available RAM and allocated/reserved VRAM.
- `torch.cuda.set_per_process_memory_fraction` caps VRAM at `MAX_GPU_MEMORY_GB` (default 6.0 of 8 GB).
  `torch.cuda.empty_cache()` runs between stages, and the batch halves on CUDA OOM.
- Single process. `N_THREADS=4` bounds BLAS, XGBoost and Arrow threads.
- Atomic writes (`.tmp` → `os.replace`) and per-part resume. A `_MANIFEST.json` marks a finished stage.
