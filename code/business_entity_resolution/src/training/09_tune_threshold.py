"""Stage 09: choose tau by maximising MACRO F0.5 PER S1 (singletons included) on OOF predictions,
after the same target-exclusivity rule used at test time.

Decision candidates = for each target, its highest-probability S1 (decision.py). The sweep is
exact over every distinct probability (metrics.exact_sweep). S1s whose true links were never
proposed by blocking keep them in n_true, so blocking misses lower the score honestly.
Pair-level accuracy, F1, ROC-AUC and PR-AUC are NOT used to pick tau.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from utils.decision import Exclusivity  # noqa: E402
from utils.gpu import check_ram  # noqa: E402
from utils.io import (add_common_args, list_shards, log, paths_from_args, read_manifest, write_json)  # noqa: E402
from utils.metrics import exact_sweep, summarize  # noqa: E402


def read_cols(p, cols):
    t = pq.read_table(p, columns=cols)
    return [t.column(c).to_numpy() for c in cols]


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__), split=False)
    ap.add_argument("--min-threshold", type=float, default=0.05,
                    help="never choose tau below this (guards against degenerate optima)")
    ap.add_argument("--compare-no-exclusivity", action="store_true",
                    help="also sweep without exclusivity (loads all OOF pairs, ~2 GB RAM)")
    args = ap.parse_args()
    args.split = "train"
    paths = paths_from_args(args)
    oman = read_manifest(paths.oof, "stage 08")
    iman = read_manifest(paths.index, "stage 03 (train)")
    n_true = np.load(os.path.join(paths.labels, "n_true.npy")).astype(np.int64)
    n_t = iman["n_targets"]
    check_ram(0.3 + n_t * 8 / 2**30 + n_t * 30 / 2**30, "09 threshold", args.force)
    shards = list_shards(paths.oof)

    ex = Exclusivity(n_t)
    for p in shards:
        s1, tgt, pr = read_cols(p, ["s1", "tgt", "p"])
        ex.pass1(tgt, pr)
    for p in shards:
        s1, tgt, pr = read_cols(p, ["s1", "tgt", "p"])
        ex.pass1b(s1, tgt, pr)
    W = {"s1": [], "tgt": [], "p": [], "y": []}
    for p in shards:
        s1, tgt, pr, y = read_cols(p, ["s1", "tgt", "p", "label"])
        m = ex.winners(s1, tgt, pr)
        W["s1"].append(s1[m]); W["tgt"].append(tgt[m]); W["p"].append(pr[m]); W["y"].append(y[m])
    W = {k: np.concatenate(v) for k, v in W.items()}
    ex.check(W["tgt"])
    log(f"exclusivity: {oman['predictions']:,} OOF pairs -> {len(W['p']):,} decision candidates "
        f"(positives among them {int(W['y'].sum()):,} of {int(n_true.sum()):,} true links)")

    thr, macro, n_pred, n_tp = exact_sweep(W["s1"], W["p"], W["y"], n_true)
    ok = thr >= args.min_threshold
    best = int(np.flatnonzero(ok)[np.argmax(macro[ok])])    # highest tau among ties (precision)
    tau = float(thr[best])
    rep = summarize(W["s1"], W["p"], W["y"], n_true, tau)
    grid = []
    for t in np.round(np.arange(0.05, 1.0, 0.05), 2):
        i = np.searchsorted(-thr, -t, side="right") - 1       # last threshold >= t
        if i >= 0:
            grid.append({"threshold": float(t), "macro_f05": float(macro[i]), "n_pred": int(n_pred[i]),
                         "precision": float(n_tp[i] / n_pred[i]), "recall": float(n_tp[i] / n_true.sum())})
    for g in grid:
        log(f"  tau {g['threshold']:.2f}: macro F0.5 {g['macro_f05']:.5f} | links {g['n_pred']:,} | "
            f"P {g['precision']:.4f} R {g['recall']:.4f}", mem=False)
    log(f"SELECTED tau = {tau:.6f}: macro F0.5 {rep['macro_f05']:.5f} | singletons {rep['macro_f05_singletons']:.4f} | "
        f"non-singletons {rep['macro_f05_non_singletons']:.4f} | P {rep['micro_precision']:.4f} R {rep['micro_recall']:.4f} | "
        f"FP {rep['false_positives']:,} FN {rep['false_negatives']:,} | zero-match S1 predicted empty "
        f"{rep['zero_match_predicted_empty']:,}/{rep['s1_zero_match']:,}")
    out = {"threshold": tau, "selection": "argmax macro F0.5 per S1 incl. singletons, OOF, after exclusivity",
           "report": rep, "sweep_grid": grid, "oof_backend": oman.get("backend"),
           "note": "Test scores are the mean of the fold models; tau was tuned on single-model OOF scores."}
    if args.compare_no_exclusivity:
        A = {k: [] for k in ("s1", "p", "y")}
        for p in shards:
            s1, pr, y = read_cols(p, ["s1", "p", "label"])
            A["s1"].append(s1); A["p"].append(pr); A["y"].append(y)
        A = {k: np.concatenate(v) for k, v in A.items()}
        t2, m2, _, _ = exact_sweep(A["s1"], A["p"], A["y"], n_true)
        out["no_exclusivity_best"] = {"threshold": float(t2[np.argmax(m2)]), "macro_f05": float(m2.max())}
        log(f"without exclusivity: best macro F0.5 {m2.max():.5f} at tau {t2[np.argmax(m2)]:.4f}")
    write_json(paths.artifact("threshold.json"), out)
    np.savez_compressed(paths.artifact("threshold_sweep_full.npz"), thresholds=thr, macro_f05=macro,
                        n_pred=n_pred, n_tp=n_tp)


if __name__ == "__main__":
    main()
