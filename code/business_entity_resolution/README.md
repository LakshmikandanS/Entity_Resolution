# Business entity resolution: training and inference pipeline

Built for an RTX 5060 Laptop GPU (8 GB) and a ~5 GB practical RAM budget on Windows. Design evidence
is in `../../problem.md`. Resource estimates are in `docs/RESOURCE_ESTIMATES.md`, and the
implementation report is in `docs/PIPELINE_REPORT.md`.

```
src/
  training/00_resource_check.py … 11_predict_test.py   one stage per script, resumable
  utils/  config io gpu normalization records blocking features metrics decision model
docs/   RESOURCE_ESTIMATES.md  PIPELINE_REPORT.md
```

Paths default to the repository root: `dataset/` (input), `work/` (bulky intermediates, ~35 GB
peak for train + test), `artifacts/` (small reviewable outputs and the model bundle), and `output/`
(submission files). Override them with `--data-dir --work-dir --artifacts-dir`.

## Setup
```bash
pip install -r requirements.txt            # see the torch / xgboost notes inside
python src/training/00_resource_check.py --probe-xgboost
```

## Run (from `code/business_entity_resolution/`)
```bash
# training split
python src/training/01_normalize.py            --split train
python src/training/02_build_rewrite_map.py                     # labels + learned Indic rewrite map
python src/training/03_build_blocking_indexes.py --split train
python src/training/04_generate_candidates.py  --split train   # prints recall, fails below --min-recall
python src/training/05_build_features.py       --split train   # GPU
python src/training/06_build_training_set.py
python src/training/07_train.py                                 # GPU XGBoost, or CPU HGB fallback
python src/training/08_generate_oof.py
python src/training/09_tune_threshold.py                        # macro F0.5 incl. singletons
python src/training/10_train_final.py

# test split (uses only artifacts learned from train)
python src/training/01_normalize.py            --split test
python src/training/03_build_blocking_indexes.py --split test
python src/training/04_generate_candidates.py  --split test
python src/training/05_build_features.py       --split test
python src/training/11_predict_test.py          # writes output/matching_results.tsv + candidate_pairs.tsv

python ../../utils/validate_submission.py --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test
```

Every stage skips work that is already finished (per-part files plus a `_MANIFEST.json`). Pass
`--force` to rebuild a stage. Tuning knobs are environment variables or CLI flags: `BATCH_SIZE`,
`MAX_GPU_MEMORY_GB`, `MAX_CANDIDATES_PER_S1`, `MAX_KEY_FREQUENCY`, `MAX_KEY_FREQUENCY_COMMON`,
`N_THREADS`, `CHUNK_ROWS`, `MAX_EXPANDED_POSTINGS`, `RAM_BUDGET_GB` (see `src/utils/config.py`).

The pipeline uses no external lookups and no pretrained text model. The only model is a gradient-boosted
tree ensemble (XGBoost, Apache-2.0, or scikit-learn, BSD-3) trained from scratch on the training split.
