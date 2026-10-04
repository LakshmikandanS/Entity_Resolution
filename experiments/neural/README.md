# Neural / hybrid experiment (bi-encoder features + baseline XGBoost)

Experimental branch next to the classical baseline in `code/business_entity_resolution/`. It asks one
question: **do bi-encoder similarity features improve out-of-fold macro-F0.5 when added to the baseline's
64 handcrafted features, with everything else held fixed?**

The baseline is imported read-only through `baseline_api.py`. Normalisation, rewrite map, blocking,
candidate pairs, handcrafted features, training-row sampling, folds, XGBoost/HGB, target exclusivity, the
exact macro-F0.5 sweep and the TSV writer are all the baseline's own code. Nothing here edits baseline files.

Design and sizing rationale: [`PLAN.md`](PLAN.md). Differences from the plan are listed at the end.

## Status

| Stage | Implemented | Synthetic smoke test (600 S1) | Real-data sample (20k train S1, real e5) | Full real data |
|---|---|---|---|---|
| `baseline_api.py` adapter | yes | yes | yes | no |
| `n01_build_encoder_data.py` | yes | yes | yes | **no** |
| `n02_train_encoder.py` (multilingual-e5-small) | yes | yes, incl. interrupt + resume | yes | **no** |
| `n03_embed.py` | yes | yes, incl. resume | yes | **no** |
| `n04_pair_similarity.py` | yes | yes, incl. rerun skip | yes | **no** |
| `n05_train_compare.py` (A/B) | yes | yes (HGB backend) | yes (HGB backend) | **no** |
| `n06_predict_test.py` | yes | yes | yes (on a held-out slice of train, not test) | **no** |

`tests/smoke_test.py` runs the unmodified baseline stages (train 01-06, test 01/03/04/05) on a tiny
synthetic dataset, then n01-n06 against them (n02 is interrupted and resumed), and checks shapes, unit
norms, no NaN, rank/margin correctness, that both arms use identical training rows, that no encoder-split S1
enters the A/B, and one output row per test S1. Its numbers mean nothing. No real data, no model download,
~2 min:

```bash
python experiments/neural/tests/smoke_test.py --out <scratch dir>
```

## Prerequisites

1. Baseline **train** stages 01-06 finished, with the default `--work-dir work` and `--artifacts-dir artifacts`
   at the repo root (or edit `baseline:` in `config.yaml`). From the repo root:
   `python code/business_entity_resolution/run_pipeline.py --to 06`. For step 6 also the baseline **test**
   stages: `python code/business_entity_resolution/run_pipeline.py --only 01-test 03-test 04-test 05-test`.
   Every neural script checks this first and prints the command to run if a stage is missing.
2. Python packages already present on this machine: torch 2.11+cu128, transformers 5.5, pyarrow, polars,
   psutil, pyyaml, scikit-learn. `xgboost` is optional exactly as in the baseline: if it is missing, both arms
   use the baseline's HistGradientBoosting fallback (CPU, 3M-row entity subsample).
3. `intfloat/multilingual-e5-small` (MIT licence, 118M parameters, ~470 MB) comes from Hugging Face. It is
   already in this machine's Hugging Face cache (downloaded while testing). That is a model download, not an
   entity lookup; no data leaves the machine.

## One command for the whole flow

```bash
python experiments/neural/run_neural.py --with-baseline   # missing baseline stages, then n01 .. n06
python experiments/neural/run_neural.py                   # n01 .. n06 when the baseline is already done
python experiments/neural/run_neural.py --dry-run         # print the commands only
python experiments/neural/run_neural.py --train-only      # n01 .. n05 (A/B report)
python experiments/neural/run_neural.py --test-only       # n03-test, n04-test, n06
python experiments/neural/run_neural.py --from n03-train --to n05
python experiments/neural/run_neural.py --stage-args n02="--max-steps 2000"
```

Each stage runs in its own process; output goes to the screen and to
`experiments/neural/work/logs/neural_<time>.log`, with a per-stage summary in `work/logs/run_<time>.json`.
It stops at the first failing stage, and rerunning the same command resumes. Tested end to end on the
20k-S1 real-data sample (8 stages, 3.1 min with a tiny model).

## Commands step by step (from the repository root; not yet run on the full real data)

```bash
# 1. encoder fine-tuning data from the deterministic 10% encoder split of TRAIN S1
python experiments/neural/n01_build_encoder_data.py

# 2. fine-tune the bi-encoder (GPU); --max-steps N stops early with a checkpoint, rerun to resume
python experiments/neural/n02_train_encoder.py

# 3. embeddings for every train record (GPU)
python experiments/neural/n03_embed.py --split train

# 4. neural pair features for every baseline train candidate pair (GPU)
python experiments/neural/n04_pair_similarity.py --split train

# 5. A/B: baseline features vs baseline + neural features, same rows/folds/model/threshold procedure
python experiments/neural/n05_train_compare.py
#    -> experiments/neural/work/compare/ab_report.md (+ .json)

# 6. test predictions with arm B (only artefacts fitted on train)
python experiments/neural/n03_embed.py --split test
python experiments/neural/n04_pair_similarity.py --split test
python experiments/neural/n06_predict_test.py            # --arm A for the retrained-baseline arm
#    -> experiments/neural/work/output_test/arm_B/{matching_results,candidate_pairs}.tsv
```

Every step is resumable: rerun the same command after an interruption. Progress files live next to the
outputs under `experiments/neural/work/` (gitignored). Per-stage wall time, peak RSS and peak VRAM are
logged to `work/runlog/*.json` and summarised in the A/B report.

## Resource budget per step

Measured on the 20k-S1 real-data sample with the real e5-small (RTX 5060 Laptop, 2026-10-04):
fine-tuning ~180 groups/s at batch 64 groups with peak VRAM 1.8 GB; embedding ~4,750 records/s
(2 sequences each) with peak VRAM 1.0 GB; peak working set 2.2-3.0 GB per process, a large share of which is
the CUDA/torch DLLs Windows counts in the working set. The full-data times below are extrapolated from those
rates; the rest are estimates.

Scale used: train S1 2,206,821, S2+S3 10,320,219; test S1 1,732,544, S2+S3 9,969,589 (streamed line counts).
Baseline `MAX_CANDIDATES_PER_S1 = 32`, so train pairs P <= 70.6M (eval split ~63.6M), test P <= 55.4M.

| Step | Pairs / records | Peak RAM | Peak VRAM | Disk written | Time (RTX 5060 Laptop) |
|---|---|---:|---:|---:|---:|
| 1 n01 | 221k S1, ~763k links, <= 2.2M hard-neg slots, <= 3M target texts | ~1.5 GB | 0 | ~0.5 GB | 10-20 min |
| 2 n02 | ~763k groups x 5 records x 2 fields, batch 64 groups | ~2.5-3.5 GB (measured 3.0 GB on the sample) | 1.8 GB measured | ~1 GB checkpoint, 0.47 GB model | ~70 min (763k / 180 groups/s) |
| 3 n03 train | 12.53M records x 2 sequences | ~2.2 GB measured | 1.0 GB measured | 6.4 GB | ~45 min (at 4,750 records/s) |
| 4 n04 train | P <= 70.6M pairs | < 1.5 GB | ~3 GB | 1.1 GB features + 0.6 GB pair metadata (+1.1 GB temp) | 5-15 min |
| 5 n05 | training rows = baseline trainset on eval S1; 2 arms x 2 folds; 2 OOF passes over P | ~1-2 GB (XGBoost) / up to ~2.6 GB (HGB) | baseline model only (XGBoost <= 0.8 x 6 GB cap) | trainset copy (same order as baseline's) + 0.56 GB OOF | dominated by 4 model fits |
| 6 n03+n04+n06 test | 11.7M records; P <= 55.4M | ~2.2 GB | ~3 GB / model only | 6.0 + 1.3 GB + TSVs | ~1 h (embedding ~41 min) |

Memory techniques: chunked Arrow reads (never pandas), length-sorted fp16 inference with per-chunk string
dedup, fp16 memmaps for embeddings and features, S1 embeddings resident on GPU with targets streamed in
1M-row blocks, gradient checkpointing + frozen word embeddings for training, a VRAM cap
(`torch.cuda.set_per_process_memory_fraction(0.85)`) and OOM back-off (batch halving) in n02.

Disk total for a full train + test run of this branch: roughly 20-25 GB on top of the baseline's ~35 GB.

## What the A/B compares

* Population: train S1 outside the 10% encoder split (~1.99M S1, singletons included). Only their pairs are
  scored, so encoder-split S1s neither compete for targets nor count in the metric. Absolute scores therefore
  differ slightly from the baseline's own stage 09 (all S1), which the report quotes separately for reference.
* Arm A: the baseline's 64 features. Arm B: the same 64 + `nn_cos_name`, `nn_cos_addr`, `nn_cos_comb`,
  `nn_cos_name_if_addr_empty`, `nn_s1_rank`, `nn_s1_margin`, `nn_t_rank`, `nn_t_margin`.
* Identical for both arms: training rows (baseline stage-06 sample restricted to eval S1, fingerprinted in the
  report), weights, folds, early-stopping subset (`es_val`), backend, hyper-parameters, XGBoost VRAM
  `keep_frac` (computed once for the wider arm), target exclusivity, exact threshold sweep with the 09 rule
  (tau >= 0.05, highest tau among ties).
* Report: macro-F0.5, tau, singleton / non-singleton macro-F0.5, micro precision/recall, FP/FN; per-segment
  F0.5/P/R (singletons, link count, Indic-name links, empty-address links, country); neural feature
  distributions and single-feature AUC; a leakage check (`neg_enc_target`); runtime, peak RAM/VRAM, disk.

## Leakage controls

* Test is read only by n03/n04/n06 with `--split test`, after every choice is fixed. tau comes from train OOF.
* The encoder never sees an evaluation-split S1 or its links. The remaining channel (targets linked to
  encoder-split S1s appear as negatives for eval S1s) is measured in the report as `neg_enc_target`.
* Hard negatives and masking keys use the baseline's rewrite map, which is learned from train links only.

## Assumptions checked against the data

* Schema: 4 string columns, tab-separated, no quoting; S1 names pure ASCII; S2/S3 ~12-15% Indic names in a
  200k-row sample. Countries in train: US, India. Row counts as above.
* Address placeholders are mostly single components inside an address (`, NULL,`, `, N/A,`, `<NULL>`), and
  fully empty addresses are plain empty strings (~3.3% of S2/S3 in a 400k-row sample). The light clean drops
  placeholder components by exact match only (`Thiruvananthapuram` stays intact). Architecture unchanged.
* Baseline target index = S2 rows then S3 rows, `s1` = file row: identical to this branch's row ids
  (asserted at runtime in n04).

## Differences from PLAN.md

* The baseline did not exist when the plan was written; the adapter is now wired to it. Training rows are the
  baseline's stage-06 sample (positives + hard negatives + 2% weighted easy negatives) rather than the
  sampling rule sketched in the plan.
* K = 32 (baseline default), so P is 70.6M rather than the 66M/110M bracket.
* n02 makes one pass over every (S1, positive) group (~763k) instead of 2 epochs over one positive per S1;
  the time estimate is 35-70 min accordingly.

## Future extension (not implemented)

Bi-encoder retrieval -> top-K -> small cross-encoder reranker only if arm B beats arm A by more than fold
noise and the remaining errors concentrate among a few top candidates per S1. Sizing is in PLAN.md section 6.

## Baseline issue found while testing

Stage 03 kept the target-record memmap open while renaming it, which failed on Windows (WinError 32). It was
reported to the baseline owner and fixed there (commit 61d5e3a); this branch did not modify it.
