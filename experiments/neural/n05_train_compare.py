"""n05: A/B experiment on the TRAIN evaluation split (the 90% of S1 not used to fine-tune the encoder).

  A: the baseline's 64 features                      â” same training rows (baseline stage 06 sample,
  B: the baseline's 64 features + 8 neural features  â”˜ eval-split S1 only), same folds, same model code,
                                                       hyper-parameters, keep_frac, exclusivity and
                                                       exact macro-F0.5 threshold sweep.
Only the feature list differs. Everything model/decision/metric related is the baseline's code
(utils.model, utils.decision, utils.metrics) called through baseline_api.py.

Out-of-fold: the baseline's entity-level folds; model f never sees an S1 of fold f. Only eval-split pairs are
scored, so encoder-split S1s neither compete for targets nor count in the metric (identical for A and B).
Test data is never read.

Estimated cost (full train; eval split ~1.99M S1, P_eval ~ 0.9 x baseline candidate pairs):
  disk   augmented training shards ~0.9 x baseline trainset (+8 float32 columns) -> a few GB, zstd parquet
         OOF probabilities 2 x P x 4 B (P = 70.6M at K=32 -> 0.56 GB); models ~4 x <100 MB
  RAM    one parquet batch (1M x 72 x 4 B = 0.29 GB) + exclusivity arrays (n_targets x 8 B = 83 MB)
         + decision candidates (<= n_targets x 13 B = 134 MB); HGB fallback adds max_train_rows x 72 x 12 B
  VRAM   baseline model only: GPU XGBoost QuantileDMatrix ~rows x (F + 24) B (07's estimate), capped by
         max_gpu_gb with the same entity subsample for both arms
  time   4 model fits (2 arms x 2 folds) + 2 OOF passes over P pairs; dominated by the model backend
Resumable: per training shard, per arm/fold model and per arm/shard OOF (work/compare/progress.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import baseline_api
from common import (Monitor, add_config_arg, atomic_write_json, dir_size_bytes, encoder_split_mask, gb,
                    load_config, open_memmap, read_json)
from n04_pair_similarity import FEATURES as NN


def hist_quantiles(h, edges, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    c = np.cumsum(h)
    if c[-1] == 0:
        return {f"p{int(q * 100):02d}": None for q in qs}
    return {f"p{int(q * 100):02d}": float(edges[np.searchsorted(c, q * c[-1]) + 1]) for q in qs}


def hist_auc(hpos, hneg):
    """Single-feature AUC from binned counts (ties within a bin count 1/2)."""
    if hpos.sum() == 0 or hneg.sum() == 0:
        return None
    neg_below = np.cumsum(hneg) - hneg
    return float((hpos * (neg_below + 0.5 * hneg)).sum() / (hpos.sum() * hneg.sum()))


def build_eval_trainset(api, eval_mask, neural, out_dir):
    """Baseline stage-06 training rows of eval-split S1s, plus the nn_* columns looked up by pair_idx."""
    shards, offs = api.shard_offsets("train")
    k_of = {os.path.basename(p): k for k, p in enumerate(shards)}
    stats = read_json(out_dir / "_stats.json", {"rows": 0, "positives": 0, "shards": {}})
    for p in api.trainset_shards():
        name = os.path.basename(p)
        dest = out_dir / name
        if dest.exists():
            continue
        t = pq.read_table(p)
        t = t.filter(pa.array(eval_mask[t.column("s1").to_numpy()]))
        k = k_of[name]
        f = pq.read_table(shards[k], columns=["s1", "tgt"])
        fkey = f.column("s1").to_numpy().astype(np.int64) << 32 | f.column("tgt").to_numpy()
        order = np.argsort(fkey)
        tkey = t.column("s1").to_numpy().astype(np.int64) << 32 | t.column("tgt").to_numpy()
        pos = np.searchsorted(fkey[order], tkey)
        pidx = offs[k] + order[pos]
        if not np.array_equal(fkey[order][pos], tkey):
            raise RuntimeError(f"{name}: training rows not found in the feature shard")
        nn = np.asarray(neural[np.sort(pidx)], dtype=np.float32)[np.argsort(np.argsort(pidx))]
        for j, c in enumerate(NN):
            t = t.append_column(c, pa.array(nn[:, j]))
        tmp = dest.with_suffix(".tmp")
        pq.write_table(t, tmp, compression="zstd")
        os.replace(tmp, dest)
        stats["shards"][name] = t.num_rows
        stats["rows"] += t.num_rows
        stats["positives"] += int(t.column("label").to_numpy().sum())
        atomic_write_json(out_dir / "_stats.json", stats)
    return sorted(str(p) for p in out_dir.glob("part-*.parquet")), stats


def main():
    ap = add_config_arg(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    ap.add_argument("--arms", default=None, help="comma list, default from config (A,B)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cc = cfg["compare"]
    arms = (args.arms or ",".join(cc["arms"])).split(",")
    api = baseline_api.load(cfg)
    api.require("01-train", "02", "03-train", "04-train", "05-train", "06")
    mon = Monitor(cfg, "n05_train_compare")
    out = cfg.wpath("compare", "x").parent
    pdir = cfg.work / "pairs"
    P = api.num_pairs("train")
    if read_json(cfg.work / "feats" / "train_progress.json", {}).get("pass2_rows") != P:
        raise SystemExit("neural features for train incomplete; run n04_pair_similarity.py --split train")

    with mon.stage("setup"):
        n1, n2, n3 = api.sizes("train")
        n_t = n2 + n3
        s1_row = np.load(pdir / "train_s1_row.npy", mmap_mode="r")
        t_gid = np.load(pdir / "train_t_gid.npy", mmap_mode="r")
        labels = np.load(pdir / "train_label.npy", mmap_mode="r")
        neural = np.load(cfg.work / "feats" / "train_neural.npy", mmap_mode="r")
        ids = pq.read_table(cfg.work / "emb" / "train" / "s1_ids.parquet")
        enc = encoder_split_mask(ids.column("entity_id").to_pylist(), cfg)
        eval_mask = ~enc
        local = np.cumsum(eval_mask) - 1                      # S1 row -> index among eval S1s
        folds = api.s1_folds()
        n_folds = api.n_folds()
        n_true = api.n_true()
        n_true_eval = n_true[eval_mask]
        base = api.feature_names()
        shards, offs = api.shard_offsets("train")
        pair_eval = np.empty(P, dtype=bool)
        for s in range(0, P, 10_000_000):
            pair_eval[s:s + 10_000_000] = eval_mask[np.asarray(s1_row[s:s + 10_000_000])]
        setup = {"P": P, "n_s1": n1, "n_eval_s1": int(eval_mask.sum()), "n_encoder_s1": int(enc.sum()),
                 "P_eval": int(pair_eval.sum()), "n_folds": n_folds, "n_baseline_features": len(base),
                 "neural_features": NN, "eval_true_links": int(n_true_eval.sum()),
                 "eval_singletons": int((n_true_eval == 0).sum())}
        print(json.dumps(setup, indent=1))

    with mon.stage("eval_trainset"):
        tshards, tstats = build_eval_trainset(api, eval_mask, neural, cfg.wpath("compare", "trainset", "x").parent)
        h = hashlib.sha256(json.dumps(tstats["shards"], sort_keys=True).encode()).hexdigest()[:16]
        setup.update({"training_rows": tstats["rows"], "training_positives": tstats["positives"],
                      "training_rows_fingerprint": h})

    backend, device = api.choose_backend(cc["backend"])
    keep_frac, est_ram = 1.0, None
    if backend == "xgboost":
        # One entity-level subsample fraction, computed for the wider arm (B) and used by both arms, so the
        # training rows stay identical. Two budgets: the baseline's VRAM rule, and host RAM, which is the
        # binding one on this laptop: XGBoost's QuantileDMatrix holds ~xgb_host_bytes_per_value bytes per
        # training value in host RAM while it is built.
        rows_fold = tstats["rows"] * (n_folds - 1) / n_folds * 0.95          # minus the 5% es_val rows
        n_feat = len(base) + len(NN)
        est_ram = 0.6 + rows_fold * n_feat * cc["xgb_host_bytes_per_value"] / 1e9
        ram_keep = min(1.0, (cc["max_train_ram_gb"] - 0.6) / max(est_ram - 0.6, 1e-9))
        vram_keep = api.xgb_keep_frac(rows_fold, n_feat, cc["max_gpu_gb"]) if device == "cuda" else 1.0
        keep_frac = min(ram_keep, vram_keep)
        print(f"XGBoost training rows per fold ~{rows_fold:,.0f} x {n_feat} features: estimated host RAM "
              f"{est_ram:.1f} GB at full size; keep_frac {keep_frac:.3f} (RAM budget {cc['max_train_ram_gb']} GB, "
              f"VRAM budget {cc['max_gpu_gb']} GB), same for both arms", flush=True)
    setup.update({"backend": backend, "device": device, "keep_frac": keep_frac,
                  "est_train_ram_gb_full": est_ram})
    mon.note("setup", setup)

    prog_path = out / "progress.json"
    prog = read_json(prog_path, {})
    results = read_json(out / "arm_results.json", {})
    mdir = cfg.wpath("compare", "models", "x").parent
    for arm in arms:
        feats = base + (NN if arm == "B" else [])
        infos = {}
        for f in range(n_folds):
            key = f"fit_{arm}_{f}"
            if prog.get(key) != "done":
                with mon.stage(f"fit_{arm}_fold{f}"):
                    model, info = api.fit_fold(tshards, f, feats, backend, device, cc["n_threads"], cc["iter_rows"],
                                               keep_frac, cc["max_train_rows"])
                    path = api.save_model(model, backend, mdir / f"{arm}_fold{f}")
                    info["model_path"] = path
                    atomic_write_json(mdir / f"{arm}_fold{f}.json", info)
                    del model
                prog[key] = "done"
                atomic_write_json(prog_path, prog)
            infos[f] = read_json(mdir / f"{arm}_fold{f}.json")
        oof = open_memmap(out / f"oof_{arm}.npy", (P,), np.float32, fill=np.nan)
        with mon.stage(f"oof_{arm}"):
            models = [api.load_model(infos[f]["model_path"], cc["n_threads"]) for f in range(n_folds)]
            for k, sp in enumerate(shards):
                key = f"oof_{arm}_{k}"
                if prog.get(key) == "done":
                    continue
                g0 = offs[k]
                for b in pq.ParquetFile(sp).iter_batches(batch_size=cc["predict_chunk"], columns=base + ["s1", "fold"]):
                    gidx = np.arange(g0, g0 + b.num_rows)
                    g0 += b.num_rows
                    m = eval_mask[b.column("s1").to_numpy()]
                    if not m.any():
                        continue
                    X = api.batch_matrix(b, base)[m]
                    if arm == "B":
                        X = np.hstack([X, np.asarray(neural[gidx[0]:gidx[-1] + 1], np.float32)[m]])
                    fold = b.column("fold").to_numpy()[m]
                    pr = np.full(int(m.sum()), np.nan, np.float32)
                    for f in range(n_folds):
                        mf = fold == f
                        if mf.any():
                            pr[mf] = api.predict(models[f], X[mf])
                    oof[gidx[m]] = pr
                oof.flush()
                prog[key] = "done"
                atomic_write_json(prog_path, prog)
            del models

        with mon.stage(f"decide_{arm}"):
            def chunks():
                for s in range(0, P, 5_000_000):
                    m = pair_eval[s:s + 5_000_000]
                    yield (np.asarray(s1_row[s:s + 5_000_000])[m], np.asarray(t_gid[s:s + 5_000_000])[m],
                           np.asarray(oof[s:s + 5_000_000])[m])
            ex = api.exclusivity_winners(n_t, chunks)
            W = {"s1": [], "tgt": [], "p": [], "y": []}
            for s in range(0, P, 5_000_000):
                m = pair_eval[s:s + 5_000_000]
                s1, tg, pr = (np.asarray(a[s:s + 5_000_000])[m] for a in (s1_row, t_gid, oof))
                y = np.asarray(labels[s:s + 5_000_000])[m]
                w = ex.winners(s1, tg, pr)
                for c, v in (("s1", s1), ("tgt", tg), ("p", pr), ("y", y)):
                    W[c].append(v[w])
            W = {c: np.concatenate(v) for c, v in W.items()}
            ex.check(W["tgt"])
            sl = local[W["s1"]]
            tau, rep = api.tune_threshold(sl, W["p"], W["y"], n_true_eval, cc["min_threshold"])
            keep = W["p"] >= tau
            pred = np.bincount(sl[keep], minlength=n_true_eval.size)
            tp = np.bincount(sl[keep], weights=W["y"][keep].astype(np.float64), minlength=n_true_eval.size)
            f05 = api.f05(tp, pred, n_true_eval)
            np.savez_compressed(out / f"entity_scores_{arm}.npz", pred=pred, tp=tp, f05=f05)
            results[arm] = {"features": feats, "tau": tau, **rep,
                            "macro_f05_check": float(f05.mean()),
                            "models": {f: {k: v for k, v in infos[f].items() if k != "params"} for f in infos}}
            atomic_write_json(out / "arm_results.json", results)
            print(arm, f"tau={tau:.5f} macro_f05={rep['macro_f05']:.5f}", flush=True)

    # ---- segments (eval S1 only) ---------------------------------------------------------------------
    with mon.stage("segments"):
        tflags = np.concatenate([np.load(cfg.work / "emb" / "train" / f"s{s}_flags.npy") for s in (2, 3)])
        indic_s1, empty_s1 = np.zeros(n1, bool), np.zeros(n1, bool)
        owned_by_enc = np.zeros(n_t, bool)    # target is a true link of an encoder-split S1
        owned_by_eval = np.zeros(n_t, bool)   # target is a true link of an evaluation-split S1
        for s in range(0, P, 10_000_000):
            lab = np.asarray(labels[s:s + 10_000_000]) == 1
            r = np.asarray(s1_row[s:s + 10_000_000])[lab]
            g = np.asarray(t_gid[s:s + 10_000_000])[lab]
            indic_s1[r[(tflags[g] & 2) > 0]] = True
            empty_s1[r[(tflags[g] & 1) > 0]] = True
            owned_by_enc[g[enc[r]]] = True
            owned_by_eval[g[eval_mask[r]]] = True
        country = np.asarray(ids.column("country").to_pylist(), dtype=object)[eval_mask]
        nt = n_true_eval
        segs = {"all": np.ones(nt.size, bool), "singleton": nt == 0, "1_link": nt == 1,
                "2-3_links": (nt >= 2) & (nt <= 3), "4+_links": nt >= 4,
                "has_indic_name_link": indic_s1[eval_mask], "has_empty_address_link": empty_s1[eval_mask]}
        for c in sorted(set(country)):
            segs[f"country={c}"] = country == c
        ents = {arm: np.load(out / f"entity_scores_{arm}.npz") for arm in results}
        seg_rows = {}
        for name, msk in segs.items():
            row = {"n_s1": int(msk.sum())}
            for arm, e in ents.items():
                tp_, pr_, ntr = e["tp"][msk].sum(), e["pred"][msk].sum(), nt[msk].sum()
                row[f"f05_{arm}"] = float(e["f05"][msk].mean()) if msk.any() else None
                row[f"precision_{arm}"] = float(tp_ / pr_) if pr_ else None
                row[f"recall_{arm}"] = float(tp_ / ntr) if ntr else None
            if {"A", "B"} <= set(ents) and msk.any():
                row["delta_f05_B_minus_A"] = row["f05_B"] - row["f05_A"]
            seg_rows[name] = row

    # ---- neural feature distributions on eval pairs ----------------------------------------------------
    with mon.stage("feature_distributions"):
        rng_of = {"nn_s1_rank": (1, 33), "nn_t_rank": (1, 33), "nn_s1_margin": (0, 2), "nn_t_margin": (0, 2)}
        bins = cc["hist_bins"]
        edges = {f: np.linspace(*rng_of.get(f, (-1, 1)), bins + 1) for f in NN}
        groups = ("pos", "neg", "neg_owned_eval", "neg_enc_target")
        H = {(f, k): np.zeros(bins, np.int64) for f in NN for k in groups}
        nan_ct = dict.fromkeys(NN, 0)
        for s in range(0, P, 5_000_000):
            ev = pair_eval[s:s + 5_000_000]
            X = np.asarray(neural[s:s + 5_000_000], dtype=np.float32)[ev]
            lab = np.asarray(labels[s:s + 5_000_000])[ev] == 1
            tg = np.asarray(t_gid[s:s + 5_000_000])[ev]
            encn = owned_by_enc[tg] & ~lab
            evn = owned_by_eval[tg] & ~lab
            for j, f in enumerate(NN):
                v = X[:, j]
                ok = ~np.isnan(v)
                nan_ct[f] += int((~ok).sum())
                vc = np.clip(v, edges[f][0], edges[f][-1])
                for k, m in (("pos", lab & ok), ("neg", ~lab & ok & ~encn), ("neg_owned_eval", evn & ok),
                             ("neg_enc_target", encn & ok)):
                    H[(f, k)] += np.histogram(vc[m], bins=edges[f])[0]
        dist = {}
        for f in NN:
            dist[f] = {k: {"n": int(H[(f, k)].sum()), **hist_quantiles(H[(f, k)], edges[f])} for k in groups}
            dist[f]["auc_pos_vs_neg"] = hist_auc(H[(f, "pos")], H[(f, "neg")])
            dist[f]["nan_rate"] = nan_ct[f] / max(setup["P_eval"], 1)

    # ---- report ---------------------------------------------------------------------------------------
    official = None
    thr_path = api.paths("train").artifact("threshold.json")
    if os.path.exists(thr_path):
        o = read_json(Path(thr_path))
        official = {"threshold": o.get("threshold"), "macro_f05": o.get("report", {}).get("macro_f05"),
                    "note": "baseline stage 09 on ALL train S1 (incl. encoder split); not the A/B population"}
    runlogs = {p.stem: json.loads(p.read_text()) for p in sorted((cfg.work / "runlog").glob("*.json"))}
    disk = {d.name: gb(dir_size_bytes(d)) for d in sorted(cfg.work.iterdir()) if d.is_dir()}
    report = {"setup": setup, "arms": results, "segments": seg_rows, "neural_feature_distributions": dist,
              "baseline_official_reference": official,
              "runtime_and_peaks": {k: {"total_seconds": v.get("total_seconds"), "stages": v.get("stages")}
                                    for k, v in runlogs.items()},
              "disk_gb": disk}
    atomic_write_json(out / "ab_report.json", report)
    (out / "ab_report.md").write_text(render_md(report), encoding="utf-8")
    print(f"report: {out / 'ab_report.md'}")
    mon.close()


def render_md(r) -> str:
    fmt = lambda x, d=4: "" if x is None else f"{x:.{d}f}"
    s = r["setup"]
    L = ["# Neural A/B report (train evaluation split, out-of-fold)", "",
         f"Eval S1 {s['n_eval_s1']:,} (encoder split {s['n_encoder_s1']:,} excluded), eval pairs {s['P_eval']:,}, "
         f"training rows {s['training_rows']:,} (fingerprint {s['training_rows_fingerprint']}, identical for both "
         f"arms), backend {s['backend']}/{s['device']}, keep_frac {s['keep_frac']:.3f}, folds {s['n_folds']}.", "",
         "| Arm | features | macro-F0.5 | tau | singletons | non-singletons | micro P | micro R | FP | FN |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, a in r["arms"].items():
        L.append(f"| {arm} | {len(a['features'])} | {a['macro_f05']:.5f} | {a['tau']:.4f} | "
                 f"{fmt(a['macro_f05_singletons'])} | {fmt(a['macro_f05_non_singletons'])} | "
                 f"{fmt(a['micro_precision'])} | {fmt(a['micro_recall'])} | {a['false_positives']:,} | "
                 f"{a['false_negatives']:,} |")
    if {"A", "B"} <= set(r["arms"]):
        L += ["", f"**B - A macro-F0.5: {r['arms']['B']['macro_f05'] - r['arms']['A']['macro_f05']:+.5f}**"]
    if r.get("baseline_official_reference"):
        o = r["baseline_official_reference"]
        L += ["", f"Reference: baseline stage 09 macro-F0.5 {fmt(o['macro_f05'], 5)} at tau {fmt(o['threshold'])} "
                  f"({o['note']})."]
    L += ["", "## Segments (eval S1)", "",
          "| Segment | S1 | F0.5 A | F0.5 B | B - A | P A | P B | R A | R B |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in r["segments"].items():
        d = row.get("delta_f05_B_minus_A")
        L.append(f"| {name} | {row['n_s1']:,} | {fmt(row.get('f05_A'))} | {fmt(row.get('f05_B'))} | "
                 f"{'' if d is None else format(d, '+.4f')} | {fmt(row.get('precision_A'))} | "
                 f"{fmt(row.get('precision_B'))} | {fmt(row.get('recall_A'))} | {fmt(row.get('recall_B'))} |")
    L += ["", "## Neural feature distributions (eval pairs)", "",
          "Leakage check: `enc-owned` = negatives whose target is a true link of an encoder-split S1, `eval-owned` = "
          "negatives whose target is a true link of another evaluation-split S1. If the encoder leaked, enc-owned "
          "negatives would look easier (lower cosine, larger margin) than eval-owned ones. AUC is pos vs "
          "negatives excluding enc-owned; AUC < 0.5 means lower values indicate a match (ranks, margins).", "",
          "| Feature | AUC | NaN rate | pos p25/p50/p75 | neg p25/p50/p75 | eval-owned p50 | enc-owned p50 |",
          "|---|---:|---:|---|---|---:|---:|"]
    q = lambda d: f"{fmt(d['p25'], 3)} / {fmt(d['p50'], 3)} / {fmt(d['p75'], 3)}"
    for f, d in r["neural_feature_distributions"].items():
        L.append(f"| {f} | {fmt(d['auc_pos_vs_neg'])} | {d['nan_rate']:.4f} | {q(d['pos'])} | {q(d['neg'])} | "
                 f"{fmt(d['neg_owned_eval']['p50'], 3)} | {fmt(d['neg_enc_target']['p50'], 3)} |")
    L += ["", "## Runtime and peaks", "",
          "torch VRAM = this process's torch allocations; GPU in use = device-wide (includes XGBoost and any other "
          "process on the GPU).", "",
          "| Script | Stage | s | peak RSS GB | torch VRAM GB | GPU in use GB |", "|---|---|---:|---:|---:|---:|"]
    for k, v in r["runtime_and_peaks"].items():
        for st, x in (v.get("stages") or {}).items():
            L.append(f"| {k} | {st} | {x['seconds']} | {x['peak_rss_gb']} | {x['peak_vram_gb']} | "
                     f"{x.get('peak_gpu_device_gb', '')} |")
    L += ["", "## Disk (experiments/neural/work)", ""] + [f"- {k}: {v} GB" for k, v in r["disk_gb"].items()]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
