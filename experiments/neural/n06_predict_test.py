"""n06: hybrid (arm B by default) test predictions -> matching_results.tsv + candidate_pairs.tsv.

Uses only artefacts fitted on TRAIN: the encoder (n02), the arm's fold models and its tau (n05). Test pairs and
the 64 handcrafted features come from the baseline's test stages (01, 03, 04, 05 with --split test); neural
features from n03/n04 with --split test. Probability = mean of the fold models; decision = baseline target
exclusivity then p >= tau; lines are formatted with the baseline's own 11_predict_test helpers. Output goes to
predict.out_dir/arm_<arm>/ and never overwrites the baseline's output/. No test statistic is used for any choice.

Estimated cost (test: 1.73M S1, 9.97M targets, P_test ~ 55M at K=32):
  RAM   one parquet batch (1M x 72 x 4 B = 0.29 GB) + exclusivity arrays (~80 MB) + id strings for writing
        (~10M short strings, ~0.8 GB peak while writing; same approach as the baseline's stage 11)
  VRAM  model inference only (XGBoost ~1 GB; HGB on CPU)
  disk  probabilities P x 4 B = 0.22 GB + the two TSVs (~0.5 GB)
  time  n_folds prediction passes over P pairs + writer; prerequisites: n03 (~1 h) and n04 (~10 min) on test
Resumable per feature shard (work/test_pred/progress_<arm>.json).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import baseline_api
from common import Monitor, add_config_arg, atomic_write_json, load_config, open_memmap, read_json
from n04_pair_similarity import FEATURES as NN


def main():
    ap = add_config_arg(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    ap.add_argument("--arm", default="B", choices=["A", "B"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    cc = cfg["compare"]
    api = baseline_api.load(cfg)
    api.require("01-test", "03-test", "04-test", "05-test")
    cmp_dir = cfg.work / "compare"
    results = read_json(cmp_dir / "arm_results.json", {})
    if args.arm not in results:
        raise SystemExit(f"arm {args.arm} has no train result; run n05_train_compare.py first")
    res = results[args.arm]
    tau = res["tau"]
    base = api.feature_names()
    feats = base + (NN if args.arm == "B" else [])
    if feats != res["features"]:
        raise SystemExit("feature list differs from the one the models were trained on")
    P = api.num_pairs("test")
    add = args.arm == "B"
    if read_json(cfg.work / "feats" / "test_progress.json", {}).get("pass2_rows") != P:
        raise SystemExit("neural features / pair metadata for test incomplete; run n03 and n04 with --split test")
    neural = np.load(cfg.work / "feats" / "test_neural.npy", mmap_mode="r") if add else None
    pdir = cfg.work / "pairs"
    s1_row = np.load(pdir / "test_s1_row.npy", mmap_mode="r")
    t_gid = np.load(pdir / "test_t_gid.npy", mmap_mode="r")
    mon = Monitor(cfg, f"n06_predict_test_{args.arm}")
    out = cfg.wpath("test_pred", "x").parent
    prog_path = out / f"progress_{args.arm}.json"
    prog = read_json(prog_path, {"P": P, "shards_done": []})
    probs = open_memmap(out / f"probs_{args.arm}.npy", (P,), np.float32, fill=np.nan)
    shards, offs = api.shard_offsets("test")

    with mon.stage("predict"):
        models = [api.load_model(res["models"][str(f)]["model_path"], cc["n_threads"])
                  for f in range(len(res["models"]))]
        for k, sp in enumerate(shards):
            if k in prog["shards_done"]:
                continue
            g0 = offs[k]
            for b in pq.ParquetFile(sp).iter_batches(batch_size=cfg["predict"]["chunk"], columns=base):
                X = api.batch_matrix(b, base)
                if add:
                    X = np.hstack([X, np.asarray(neural[g0:g0 + b.num_rows], np.float32)])
                probs[g0:g0 + b.num_rows] = np.mean([api.predict(m, X) for m in models], axis=0)
                g0 += b.num_rows
            probs.flush()
            prog["shards_done"].append(k)
            atomic_write_json(prog_path, prog)
        del models
        if np.isnan(probs).any():
            raise RuntimeError("some test pairs have no probability")

    with mon.stage("decide_and_write"):
        n1, n2, n3 = api.sizes("test")

        def chunks():
            for s in range(0, P, 5_000_000):
                yield (np.asarray(s1_row[s:s + 5_000_000]), np.asarray(t_gid[s:s + 5_000_000]),
                       np.asarray(probs[s:s + 5_000_000]))
        ex = api.exclusivity_winners(n2 + n3, chunks)
        entity_ids, grouped_lines = api.predict_script_helpers()
        norm = api.paths("test").normalized
        s1_ids = entity_ids(norm, "s1").to_pylist()
        tgt_ids = pa.concat_arrays([entity_ids(norm, "s2"), entity_ids(norm, "s3")])
        per = api.s1_per_shard("test")
        out_dir = cfg.root / cfg["predict"]["out_dir"] / f"arm_{args.arm}"
        out_dir.mkdir(parents=True, exist_ok=True)
        m_path, c_path = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
        assigned, n_links = [], 0
        with open(str(m_path) + ".tmp", "w", encoding="utf-8", newline="\n") as fm, \
                open(str(c_path) + ".tmp", "w", encoding="utf-8", newline="\n") as fc:
            fm.write("source1_entity_id\tmatched_entity_ids\n")
            fc.write("source1_entity_id\tcandidate_entity_ids\n")
            for k in range(len(shards)):
                lo, hi = k * per, min(n1, (k + 1) * per)
                s1 = np.asarray(s1_row[offs[k]:offs[k + 1]])
                tg = np.asarray(t_gid[offs[k]:offs[k + 1]])
                pr = np.asarray(probs[offs[k]:offs[k + 1]])
                ids = tgt_ids.take(pa.array(tg)).to_pylist()
                fc.write("\n".join(grouped_lines(s1_ids, lo, hi, s1, ids)) + "\n")
                keep = ex.winners(s1, tg, pr) & (pr >= tau)
                fm.write("\n".join(grouped_lines(s1_ids, lo, hi, s1[keep],
                                                 [ids[i] for i in np.flatnonzero(keep)])) + "\n")
                assigned.append(tg[keep])
                n_links += int(keep.sum())
            if len(shards) * per < n1:
                raise RuntimeError("candidate shards do not cover every S1 row")
        ex.check(np.concatenate(assigned) if assigned else np.zeros(0, np.int32))
        os.replace(str(m_path) + ".tmp", m_path)
        os.replace(str(c_path) + ".tmp", c_path)
        rep = {"arm": args.arm, "tau_from_train_oof": tau, "s1": n1, "candidate_pairs": P, "predicted_links": n_links,
               "matching_results": str(m_path), "candidate_pairs_tsv": str(c_path)}
        atomic_write_json(out / f"report_{args.arm}.json", rep)
        mon.note("outputs", rep)
        print(rep)
    mon.close()


if __name__ == "__main__":
    main()
