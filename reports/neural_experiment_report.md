# Neural bi-encoder experiment: status report

Date: 2026-10-04. Machine: RTX 5060 Laptop (8 GB), 16 GB RAM, Windows.
Code: `experiments/neural/` (run with `python experiments/neural/run_neural.py`). Baseline numbers are in the
[baseline report](https://claude.ai/code/artifact/e64d7526-3b76-4901-8321-07c87d2aa7ce) and
`reports/baseline_pipeline_report.md`; this report covers only the neural experiment.

## Summary

* The train side of the experiment is complete on the full real data: encoder training, embeddings for all
  12.5M train records, and 8 neural features for all 69.2M baseline candidate pairs.
* **The A/B comparison (does adding the neural features to the baseline model raise macro-F0.5?) has not run
  yet.** The run was stopped on request at 17:34 IST, 2 minutes into the first of its four XGBoost fits.
  Test-side steps have not run either. No macro-F0.5 number exists for the hybrid model yet.
* Early signal, from a 4.5M-pair sample of entities the encoder never saw: the combined embedding similarity
  alone separates true from false candidate pairs better than any single baseline score I checked (AUC 0.9988
  vs 0.9717 for the baseline's candidate rank), and it puts a true match first for 99.8% of entities vs 97.8%
  for the baseline's candidate rank. This is a single-feature comparison, not a model result: the baseline's
  XGBoost already combines 64 features and reaches OOF macro-F0.5 0.9720, so the A/B is what decides whether
  the neural features add anything.

## What ran

| Step | What it did | Wall time | Peak RAM* | Peak VRAM (process / device) |
|---|---|---:|---:|---:|
| n01 encoder data | 10% of train S1 (220,545 entities, 763,000 true links, 2.40M target texts, ~10 hard negatives per entity from baseline candidates) | 2.0 min | 3.0 GB | 0 |
| n02 encoder training | multilingual-e5-small (MIT, 118M params, 21.7M trained), 384→128 projection, InfoNCE with hard + in-batch negatives, 11,678 steps of 64 groups | 28.2 min after resume (+ 66 min before the pause, slowed by GPU sharing) | 3.9 GB | 1.8 / 4.2 GB** |
| n03 train embeddings | name + address embeddings for 12,526,821 records, fp16 memmaps | 45.5 min | 5.3 GB | 1.0 / 2.3 GB |
| n04 pair similarities | 8 features for 69,150,434 candidate pairs, 11 target blocks on GPU | 0.7 min | 5.4 GB | 3.1 / 3.5 GB |
| n05 A/B (partial) | eval-split training rows built (29.6M rows, 6.68M positives); stopped during the first XGBoost fit | 1.1 min done | 4.6 GB at stop | n/a |

\* Windows working set, which includes memory-mapped embedding files the OS can release; the private
footprint is lower. \*\* Device-wide, while the baseline was also using the GPU.

Encoder training was paused once (15:53 to 16:16 IST) so the baseline could finish its submission without
GPU contention; it resumed from its step-7,000 checkpoint.

## Encoder quality

Recall@1 on 4,505 held-out encoder-validation entities: how often the true match scores above every one of
that entity's ~10 hardest wrong candidates from the baseline blocking.

| Step | Name + address | Address only | Name only |
|---:|---:|---:|---:|
| 2,000 | 0.956 | 0.838 | 0.599 |
| 6,000 | 0.976 | 0.846 | 0.618 |
| 11,678 (final) | **0.979** | **0.853** | **0.623** |

Name alone is weak by design: same-name "twins" at other addresses are hard negatives, and the name loss masks
them, so the name embedding is not pushed to separate them; the address decides, as `problem.md` §3.3 found.

## Neural features on unseen entities (sample check)

Sample: the first 5M candidate pairs, restricted to evaluation-split entities (not used to train the encoder):
4,504,464 pairs, 143,712 entities, 483,227 true pairs. Single-feature AUC, true vs false candidate pairs:

| Neural feature | AUC | Baseline feature | AUC |
|---|---:|---|---:|
| nn_cos_comb | **0.9988** | blk_rank (candidate rank) | 0.9717 |
| nn_s1_margin | 0.9932 | h_score (heuristic score) | 0.9698 |
| nn_s1_rank | 0.9841 | blk_score | 0.9376 |
| nn_cos_addr | 0.9807 | addr_best | 0.9359 |
| nn_t_margin | 0.9778 | name_best | 0.7313 |
| nn_t_rank | 0.9751 | | |
| nn_cos_name | 0.9010 | | |
| nn_cos_name_if_addr_empty | 0.8310 (only defined when an address is empty) | | |

Top-1 precision per entity (135,550 entities with at least one true candidate): is the best-scored candidate
a true match? nn_cos_comb 0.9977, baseline blk_rank 0.9775, baseline h_score 0.9329.

Caveats: this compares single features, not models, on a sample of the training split. The baseline model's
real strength is its OOF macro-F0.5 of 0.9720 with all features combined. The leakage check the A/B report runs
(negatives owned by encoder-split vs evaluation-split entities) passed on the 20k-entity test sample but has
not run on the full data yet.

## What is left, and how to resume

Everything finished is saved under `experiments/neural/work/` (9.9 GB: embeddings 6.1 GB, A/B training rows
1.4 GB, features 1.1 GB, pair metadata 0.6 GB, model 0.5 GB, encoder data 0.2 GB). To continue:

```
python experiments/neural/run_neural.py --from n05
```

| Remaining step | Estimate | Notes |
|---|---:|---|
| n05 A/B: 4 GPU XGBoost fits | ~25 min | ~4.6-5 GB RAM; both arms train on the same 83.6% entity subsample (RAM cap) |
| n05: OOF predictions for both arms + report | ~20-25 min | writes `experiments/neural/work/compare/ab_report.md` |
| n03 test embeddings (11.7M records) | ~40 min | ~1 GB VRAM |
| n04 test similarities + n06 test predictions | ~20 min | writes `experiments/neural/work/output_test/arm_B/` |

Total about 1 h 45 min, best run with the machine otherwise idle (n05 needs up to ~5 GB RAM).

## Run log

* 14:45 n01 started (after baseline stage 07). 14:47 n02 started.
* 15:41-15:52 both jobs slowed sharply while sharing RAM and GPU with the baseline's test features.
* 15:53 n02 paused at step 7,000 at the coordinator's request; 16:16 resumed from the checkpoint.
* 16:44 n02 done; 17:30 n03 and n04 done; 17:31 n05 started.
* 17:34 stopped on request (machine needed for other work).

Fixes made along the way (all in `experiments/neural/`, committed on `training-pipeline`): clear
prerequisite messages instead of tracebacks, a corrected like-for-like leakage check, lighter checkpoints,
UTF-8-safe output for Indic text on Windows consoles, device-wide GPU memory in run logs, and a host-RAM cap
for the A/B's XGBoost fits.
