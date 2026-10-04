# Neural bi-encoder experiment: results report

Date: 2026-10-04. Machine: RTX 5060 Laptop (8 GB), 16 GB RAM, Windows.
Code: `experiments/neural/` (run with `python experiments/neural/run_neural.py`). Baseline numbers are in the
[baseline report](https://claude.ai/code/artifact/e64d7526-3b76-4901-8321-07c87d2aa7ce) and
`reports/baseline_pipeline_report.md`; this report covers only the neural experiment.

## Summary

**Adding the 8 neural features to the baseline's 64 features raises out-of-fold macro-F0.5 from 0.97167 to
0.97957 (+0.0079).** Both arms used identical training rows, folds, XGBoost settings, target exclusivity and
threshold search; only the feature list differed. They were scored on the 1,986,276 train entities the encoder
never saw (singletons included). The full run, train and test, is complete, and the hybrid's test submission
files pass the format check.

| | A: baseline features (64) | B: + neural features (72) |
|---|---:|---:|
| OOF macro-F0.5 | 0.97167 | **0.97957** |
| threshold tau | 0.7023 | 0.7147 |
| singletons F0.5 | 0.9707 | **0.9850** |
| non-singletons F0.5 | 0.9717 | **0.9792** |
| micro precision / recall | 0.9926 / 0.9349 | **0.9957 / 0.9490** |
| false matches | 47,786 | **28,238** (-41%) |
| missed matches | 447,538 | **350,809** (-22%) |

Reference: the baseline's own stage 09 reports 0.97201 on all train entities (encoder split included, full
training rows). Arm A is close to it; the small gap comes from the different population and from both arms
training on the same 83.6% entity subsample (RAM cap). Absolute baseline details are in the baseline report.

Per segment, B beats A everywhere:

| Segment | Entities | F0.5 A | F0.5 B | B - A |
|---|---:|---:|---:|---:|
| all | 1,986,276 | 0.9717 | 0.9796 | +0.0079 |
| singletons | 110,940 | 0.9707 | 0.9850 | +0.0143 |
| 1 link | 107,291 | 0.9159 | 0.9336 | +0.0177 |
| 2-3 links | 815,204 | 0.9709 | 0.9792 | +0.0083 |
| 4+ links | 952,841 | 0.9787 | 0.9844 | +0.0057 |
| has an Indic-script name link | 245,430 | 0.9791 | 0.9873 | +0.0082 |
| has an empty-address link | 220,202 | 0.9440 | 0.9531 | +0.0091 |
| India | 794,992 | 0.9684 | 0.9817 | +0.0133 |
| US | 1,191,284 | 0.9738 | 0.9781 | +0.0043 |

Leakage check: negatives whose target belongs to an encoder-training entity and negatives whose target belongs
to another evaluation entity have identical medians on every neural feature (e.g. nn_cos_comb 0.350 vs
0.350), so the encoder did not memorise anything that flatters the evaluation.

Test submission (arm B, tau 0.7147 from train OOF, mean of the two fold models):
`experiments/neural/work/output_test/arm_B/matching_results.tsv` and `candidate_pairs.tsv`. 1,732,544 S1 rows,
5,791,229 links (baseline: 5,730,818), 98,725 entities left empty (5.7%; train singleton rate 5.6%), same
54,724,105 candidate pairs as the baseline. Local format check (baseline `src/tools/check_submission.py`): PASS.
It has not been copied to `output/`, which still holds the baseline submission.

## What ran

| Step | What it did | Wall time | Peak RAM* | Peak VRAM (process / device) |
|---|---|---:|---:|---:|
| n01 encoder data | 10% of train S1 (220,545 entities, 763,000 true links, 2.40M target texts, ~10 hard negatives per entity from baseline candidates) | 2.0 min | 3.0 GB | 0 |
| n02 encoder training | multilingual-e5-small (MIT, 118M params, 21.7M trained), 384→128 projection, InfoNCE with hard + in-batch negatives, 11,678 steps of 64 groups | 28.2 min after resume (+ 66 min before the pause, slowed by GPU sharing) | 3.9 GB | 1.8 / 4.2 GB** |
| n03 train embeddings | name + address embeddings for 12,526,821 records, fp16 memmaps | 45.5 min | 5.3 GB | 1.0 / 2.3 GB |
| n04 pair similarities | 8 features for 69,150,434 candidate pairs, 11 target blocks on GPU | 0.7 min | 5.4 GB | 3.1 / 3.5 GB |
| n05 A/B | eval-split training rows (29.6M rows, 6.68M positives), 4 GPU XGBoost fits, OOF for both arms, report | 1.1 + 25.4 min | 9.2 GB in arm B's fits | 5.9 GB device |
| n03 test embeddings | 11,702,133 test records | 42.3 min | 3.6 GB | 1.0 / 2.3 GB |
| n04 + n06 test | similarities for 54.7M pairs, predictions, exclusivity, TSVs | 5.1 min | 3.7 GB | 1.1 GB device |

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

This compares single features on a sample; the model-level answer is the A/B in the summary. On all
62.2M evaluation pairs the A/B report gives nn_cos_comb an AUC of 0.9987, consistent with this sample.

## Reproduce

Everything is saved under `experiments/neural/work/` (gitignored, ~12 GB). From the repository root, after
the baseline train and test stages: `python experiments/neural/run_neural.py`. Finished steps are skipped.

Resource notes: the arm-B XGBoost fits peaked at 9.2 GB working set, above the 5 GB estimate in `config.yaml`
(`compare.max_train_ram_gb`); they need the machine to themselves.

## Run log

* 14:45 n01 started (after baseline stage 07). 14:47 n02 started.
* 15:41-15:52 both jobs slowed sharply while sharing RAM and GPU with the baseline's test features.
* 15:53 n02 paused at step 7,000 at the coordinator's request; 16:16 resumed from the checkpoint.
* 16:44 n02 done; 17:30 n03 and n04 done; 17:31 n05 started.
* 17:34 stopped on request (machine needed for other work).
* 18:56 resumed from n05. 19:21 A/B done (+0.0079).
* 20:04 test embeddings saved, then the step failed writing its timing log (Windows file lock, WinError 5);
  fixed with retries and a non-fatal log write, resumed at 20:05.
* 20:10 test predictions and submission files written; format check PASS.

Fixes made along the way (all in `experiments/neural/`, committed on `training-pipeline`): clear
prerequisite messages instead of tracebacks, a corrected like-for-like leakage check, lighter checkpoints,
UTF-8-safe output for Indic text on Windows consoles, retrying/non-fatal run-log writes, device-wide GPU memory in run logs, and a host-RAM cap
for the A/B's XGBoost fits.
