# Neural / hybrid experiment — design and resource plan

Status: **implemented and smoke-tested on synthetic data; not run on real data.** See README.md for the
run commands, the current resource table and the differences from this plan (the baseline now exists under
`code/business_entity_resolution/` and is imported read-only; K = 32). The baseline code (normalisation,
blocking, features, folds, XGBoost/HGB, threshold, metric) is imported, never copied.

## 1. What I checked (2026-10-04)

| Item | Finding |
|---|---|
| Repo | Only `README.md`, `problem.md`, `dataset/`. The `analysis/` and `baseline/` folders that `problem.md` cites **are not in the repo**; the baseline thread is writing the pipeline now. |
| GPU | RTX 5060 Laptop, 8,151 MiB, compute 12.0 (sm_120); 7.3 GB free at idle. |
| PyTorch | 2.11.0+cu128, sm_120 in arch list, FP16 matmul on CUDA tested OK. |
| Installed | transformers 5.5.0, sentence-transformers 5.5.1, peft, accelerate, polars 1.44, pyarrow 24, faiss-cpu, onnxruntime-gpu, scikit-learn 1.9. |
| Missing | **xgboost** (not installed), rapidfuzz. No HF models cached locally. |
| RAM | 15.4 GB total, **2.4 GB free** at the moment of checking. Plan for ≤ 4 GB peak. |
| Disk | 544 GB free on C:. |

Row counts (streamed line count, no parsing):

| Split | S1 | S2 | S3 | S2+S3 targets |
|---|---:|---:|---:|---:|
| train | 2,206,821 | 5,034,616 | 5,285,603 | 10,320,219 |
| test | 1,732,544 | 4,887,273 | 5,082,316 | 9,969,589 |

Field lengths (first 200k train rows per source): names mean 24–25 chars / p95 37–42 / max 104, 3.5 words;
addresses mean 46–52 chars / p95 91–103 / max 222. 15% of S2 and 12% of S3 names in the sample are non-ASCII
(Indic script).

## 2. Where the branch fits

```
baseline: raw → normalise → rewrite map → blocking → candidate pairs → handcrafted feats ─┐
                                                                                         ├→ XGBoost → exclusivity → τ → output
neural:   raw text (light clean) → bi-encoder (GPU) → name/addr embeddings → pair sims ──┘
```

The neural branch consumes the baseline's **candidate pair table** and adds columns to its **feature table**. It does
not change blocking, folds, the decision rule or the metric, so both arms are scored by the same code.

### Data split (train only, test never touched)
- **Encoder split:** 10% of train S1 entities, chosen by a deterministic hash of `entity_id` (~221k S1, ~763k
  positive links). Used only to fine-tune the bi-encoder.
- **Evaluation split:** the other 90% (~1.99M S1). Both arms (baseline features vs baseline + neural features) are
  trained and scored here with the baseline's own 2-fold cross-fitting by S1 and its OOF macro-F0.5 threshold search.
- Why not reuse all of train: if the encoder sees an S1's links, its cosine on that S1's pairs is memorised and
  XGBoost would over-trust it, making the OOF score dishonest. Cross-fitting the encoder (2 encoders) would double
  target embedding cost; a 10% holdout is plenty for fine-tuning a small model.
- Known minor bias: targets linked to encoder-split S1s still appear as negatives for eval S1s. I will log the cosine
  distribution of those negatives vs the rest to check that it is negligible.

## 3. Model

**Recommended: `intfloat/multilingual-e5-small` (MIT)** — 118M params, of which ~96M is the 250k-token vocabulary
embedding and ~21M is the 12-layer, 384-wide transformer. XLM-R tokenizer covers all nine Indic scripts in the data
and French. Licence file will be checked on download.

| Option | Licence | Params | Why / why not |
|---|---|---:|---|
| multilingual-e5-small | MIT | 118M | Retrieval-pretrained, multilingual, small. **Pick.** |
| paraphrase-multilingual-MiniLM-L12-v2 | Apache-2.0 | 118M | Same size; fallback if e5 underperforms. |
| char-CNN from scratch | own code | ~2M | Fastest and typo-robust, but no Indic/French pretraining. Ablation only. |
| e5-base / bge-m3 | MIT | 278M / 568M | 2.5–5× embedding time. Not justified until small model shows gain. |

Architecture: one shared encoder, mean pooling, separate inputs `"name: …"` and `"address: …"`, plus a trained linear
projection 384 → **128** (keeps disk and similarity cost down). Word-embedding matrix **frozen** (saves 96M × 12 B of
optimiser state and keeps unseen French tokens aligned with pretraining); all transformer layers trained.

Input text: raw strings with only placeholder (`<NULL>`, `None`, `--`) and whitespace cleanup, **not** the baseline's
normalised text. The point is a signal that is complementary to the rule-based/learned rewrites, and native Indic
script handling that also applies to France.

### Training objective
Group = 1 S1 + 1 positive target + 3 hard negatives. Hard negatives come from the baseline candidate set of that S1
(same blocking/name family, not a true link), sampled from its top-ranked non-links.
- Combined loss: InfoNCE on `normalize(name_emb ⊕ addr_emb)` with hard negatives + in-batch negatives.
- Address loss: InfoNCE on address embeddings, hard negatives preferred from same-core-name twins (§3.3 of
  `problem.md`: the address decides between twins).
- Name loss: InfoNCE on name embeddings with in-batch negatives only, **masking** negatives whose normalised core
  name equals the anchor's (a twin's name is not a wrong name).
- In-batch masking of other positives of the same S1 (S2/S3 are not deduplicated). Temperature 0.05.

## 4. Resource estimates

Notation: P = candidate pairs. The baseline's top-K is not fixed yet; I use **K = 30 (typical)** and **K = 50 (upper
bound)**: train eval split P ≈ 60M / 100M, test P ≈ 52M / 87M.

Token lengths (XLM-R ≈ 3–4 Latin chars/token, Indic denser): names mean ~10, cap 32; addresses mean ~17, cap 64.
Per-batch padding with length bucketing gives ≈ 12 / 20 average padded tokens.

Forward cost of the 21M-param body ≈ 2 × 21M = 43 MFLOP/token.

### 4.1 Fine-tuning (encoder split)
| Item | Calculation | Estimate |
|---|---|---:|
| Weights fp32 | 118M × 4 B | 0.47 GB |
| Grads + AdamW (trainable 21.7M) | 21.7M × (4 + 8) B | 0.26 GB |
| Activations, bf16, no checkpointing | per token per layer s·h·(34 + 5·a·s/h) B → names 4.2 MB/seq, addr 9.2 MB/seq at worst-case padding 24/48; × 320 records/batch | 4.3 GB |
| Activations with gradient checkpointing | layer inputs 12·s·h·2 B + one layer recompute | ~0.6 GB |
| CUDA context + allocator slack | | ~0.8 GB |
| **Peak VRAM** (batch 64 groups = 320 records = 640 sequences, checkpointing on) | | **≈ 2.2–2.8 GB** |
| Peak RAM | groups streamed from parquet; tokenisation per batch | < 1.5 GB |
| Time | 221k groups × 5 records × ~32 tok ≈ 35M tokens/epoch × 43 MFLOP × 4 (fwd+bwd+recompute) ≈ 6 PFLOP; at 5–10 TFLOPS effective | **10–20 min/epoch**, 2 epochs |

Safety: `set_per_process_memory_fraction(0.85)`, log `max_memory_allocated` every 100 steps, on OOM halve the batch
and resume from the last checkpoint (checkpoint every 1,000 steps).

### 4.2 Embedding (inference)
| Item | Calculation | Estimate |
|---|---|---:|
| Records to embed | train: all S1 + targets = 12.5M; test 11.7M; × 2 fields | 25.0M / 23.4M sequences |
| Weights fp16 | 118M × 2 B | 0.24 GB |
| Activations (no grad), batch 1024 seq × 48 tok | 1024·48·384·~20 B + attention 1024·12·48²·2 B | ~0.45 GB |
| **Peak VRAM** | batch 1024 (addresses) / 2048 (names) | **≈ 1.2 GB** |
| Peak RAM | chunk of 200k rows: strings ~80 MB, token ids per batch, output 200k × 2 × 128 × 2 B = 102 MB | **< 1 GB** |
| Disk | 12.5M × 2 × 128 × 2 B (fp16 memmap) | 6.4 GB train, 6.0 GB test |
| Time | train ≈ 12.5M × 32 tok = 400M tokens × 43 MFLOP = 17 PFLOP → 30–60 min GPU; tokenisation ~25M seq at ~50k seq/s ≈ 8 min CPU | **≈ 40–70 min per split** |

Embeddings are written once and reused by every later stage (and by a future reranker).

### 4.3 Pair similarity on GPU
All target embeddings (10.3M × 512 B = 5.3 GB) plus S1 (1.0 GB) do not fit safely in 8 GB, so:
- Eval-split S1 embeddings resident on GPU: 1.99M × 512 B = **1.0 GB**.
- Targets processed in blocks of 1M rows (0.5 GB on GPU); for each block, scan the pair table for pairs whose target
  is in the block (unsorted parquet scan, ~10 reads of P × 8 B ≈ 0.8 GB each, cheap) and process them in chunks of
  1M pairs: gather 1M × 2 sides × 256 dims × 2 B = **1.0 GB** transient.
- Output written by **pair index** into a float16 memmap `[P, F_neural]` so it lines up with the baseline feature
  table without a 100M-row join: 100M × 8 × 2 B = 1.6 GB disk.
- **Peak VRAM ≈ 2.8 GB, peak RAM < 1.5 GB.** Dot products are 100M × 256 = 26 GFLOP, so this stage is I/O bound;
  GPU is used because the data is already there, not because it is needed. ~5–10 min.

### 4.4 Neural features (8 columns)
`cos_name`, `cos_addr`, `cos_comb`, `cos_name_if_addr_empty`, and competition features on `cos_comb`: rank within the
S1's candidates, margin to the S1's best, rank among the target's S1 suitors, margin to the target's best suitor.
Target-side features are computed inside the target block (all of a target's pairs are in one block); S1-side in a
second pass over S1 blocks of 200k entities. Peak RAM < 1 GB.

Note: `cos_comb` of unit-norm concatenated halves equals `(cos_name + cos_addr)/2`, which trees can already derive;
it is kept only because it is the trained objective. The competition features are where the extra value should be.

### 4.5 XGBoost (needs `xgboost` installed)
Training rows per fold, eval split, K = 30: ~30M pairs. Raw fp32 would be 30M × ~60 feats × 4 B = 7.2 GB — **does not
fit RAM**. Redesign:
- Train on all positives (~3.4M per fold) + negatives sampled 100% from the top-5 of each S1 and 10% of the rest
  (~10M rows per fold). The threshold is still tuned on full, unsampled OOF predictions, so sampling does not
  bias the decision.
- Build `QuantileDMatrix` from a `DataIter` over parquet shards of 1M rows (232 MB RAM per shard).
- GPU ELLPACK at `max_bin=256`: ~10M × 60 × ~1 B ≈ **0.6 GB VRAM**, + gradients 80 MB; trees `max_depth=8`,
  `lr=0.05`, ≤ 2,000 rounds, early stopping, `subsample=0.8`, `colsample_bytree=0.8`, `device=cuda`, `nthread=4`.
  **Peak VRAM ≈ 1.5–2 GB, peak RAM ≈ 1.5 GB.**
- OOF / test prediction in chunks of 2M pairs; probabilities to memmap (P × 4 B = 0.4 GB).
- Fallback if the CUDA build fails on sm_120: CPU `hist` XGBoost or the baseline's HistGradientBoosting, same
  sampling and folds.

The same routine trains both arms with identical rows, folds, hyper-parameters and threshold search; only the feature
list differs. This is the A/B that answers "does the neural branch improve macro-F0.5".

### 4.6 Test inference
Encoder (trained on the 10% train split) embeds test S1 + targets (§4.2, ~1 h), similarities on the baseline's test
candidates (§4.3), hybrid model = average of the two fold models, baseline exclusivity rule and the τ chosen on train
OOF. No test statistic is used anywhere.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Gain is small. The analysis shows address features already separate same-name twins (address Jaccard 0.80 vs 0.007). | Report per-segment deltas (Indic names, low name-Jaccard tail, empty addresses) so a small total gain is still interpretable. Stop before the cross-encoder if the delta is within noise. |
| France is unseen; neither the rewrite map nor the fine-tune covers French. | Frozen word embeddings + multilingual pretraining is the main argument for the neural branch here. Cannot be validated on train; stated as such. |
| Embedding time is 2× my estimate (laptop thermals, tokenizer). | Unique-string dedup per source before encoding, length bucketing, resumable chunked output; worst case ~2 h per split. |
| RAM is already tight (2.4 GB free at check time). | Single process, chunked reads, no pandas over full files, memmaps; each stage logs peak RSS. |
| Baseline pair/feature table format not defined yet. | Code written against a small adapter (`baseline_api.py`) that is mapped to the baseline modules once they exist. |

## 6. Future: retrieval + cross-encoder (not implemented)
Only if the hybrid beats the baseline and error analysis shows remaining errors concentrated among a few top
candidates per S1:
1. Bi-encoder retrieval per country with faiss (already installed) to add candidates the blocking caps dropped.
2. Cross-encoder (same e5-small body, pair input ~64 tokens) on top-5 per S1: train ~10M pairs × 64 tok = 640M tokens
   ≈ 27 PFLOP ≈ 45–90 min per split for inference; ~2–3 GB VRAM at batch 256.

## 7. Planned files (all under `experiments/neural/`)
`config.yaml`, `baseline_api.py` (adapter), `n01_build_encoder_data.py`, `n02_train_encoder.py`,
`n03_embed.py`, `n04_pair_similarity.py`, `n05_train_compare.py` (A/B on the eval split), `n06_predict_test.py`,
`README.md` with run commands and expected time/memory per step.
