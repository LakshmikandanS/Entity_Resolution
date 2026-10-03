"""Macro F0.5 per S1 entity (singletons included) and the exact threshold sweep.

Per S1 with n_true true links, pred predicted links and tp correct ones:
    F0.5 = 1.25 * tp / (0.25 * n_true + pred)      (= (1+b^2)PR / (b^2 P + R))
    special case n_true == 0 and pred == 0 -> 1.0  (correct singleton)
n_true always counts ALL ground-truth links, including those blocking never proposed, so recall
lost in blocking is charged to the score.
"""
import numpy as np

BETA2 = 0.25


def f05(tp, pred, n_true):
    tp, pred, n_true = (np.asarray(x, dtype=np.float64) for x in (tp, pred, n_true))
    denom = BETA2 * n_true + pred
    out = np.where(denom > 0, (1 + BETA2) * tp / np.maximum(denom, 1e-12), 0.0)
    return np.where((n_true == 0) & (pred == 0), 1.0, out)


def exact_sweep(s1, p, y, n_true):
    """Macro F0.5 at every distinct threshold over the decision candidates (one per target after
    exclusivity). Returns (thresholds desc, macro_f05, n_pred, n_tp) for 'keep pairs with p >= t'.

    Each pair, taken in descending p, raises its S1's (tp, pred) by (y, 1); the change in that S1's
    F0.5 is exact, so the macro score at every cut is a cumulative sum. O(W log W)."""
    n_s1 = len(n_true)
    base = float(np.sum(n_true == 0))                 # everyone empty: only singletons score
    if len(p) == 0:
        return np.array([1.0]), np.array([base / n_s1]), np.array([0]), np.array([0])
    order = np.lexsort((s1, -p))                       # p desc, then s1
    s1o, po, yo = s1[order], p[order], y[order].astype(np.int64)
    # position of each pair within its S1 (in descending p) and cumulative tp at that position
    o2 = np.argsort(s1o, kind="stable")                # stable keeps descending-p order inside S1
    s1s, ys = s1o[o2], yo[o2]
    starts = np.flatnonzero(np.r_[True, s1s[1:] != s1s[:-1]])
    sizes = np.diff(np.r_[starts, len(s1s)])
    k_in = np.arange(len(s1s)) - np.repeat(starts, sizes) + 1           # pred after adding
    cs = np.cumsum(ys)
    tp_in = cs - np.repeat(cs[starts] - ys[starts], sizes)              # tp after adding
    nt = n_true[s1s]
    delta_sorted = f05(tp_in, k_in, nt) - f05(tp_in - ys, k_in - 1, nt)
    delta = np.empty_like(delta_sorted)
    delta[o2] = delta_sorted
    macro = (base + np.cumsum(delta)) / n_s1
    n_pred = np.arange(1, len(po) + 1)
    n_tp = np.cumsum(yo)
    # one entry per distinct threshold: last row of each run of equal p
    last = np.r_[po[1:] != po[:-1], True]
    return po[last], macro[last], n_pred[last], n_tp[last]


def summarize(s1, p, y, n_true, tau):
    """Detailed report at threshold tau over decision candidates."""
    keep = p >= tau
    n_s1 = len(n_true)
    pred = np.bincount(s1[keep], minlength=n_s1)
    tp = np.bincount(s1[keep], weights=y[keep].astype(np.float64), minlength=n_s1)
    f = f05(tp, pred, n_true)
    single = n_true == 0
    total_true = int(n_true.sum())
    n_pred, n_tp = int(pred.sum()), int(tp.sum())
    return {
        "threshold": float(tau),
        "macro_f05": float(f.mean()),
        "macro_f05_singletons": float(f[single].mean()) if single.any() else None,
        "macro_f05_non_singletons": float(f[~single].mean()) if (~single).any() else None,
        "micro_precision": n_tp / n_pred if n_pred else None,
        "micro_recall": n_tp / total_true if total_true else None,
        "predicted_links": n_pred,
        "true_positives": n_tp,
        "false_positives": n_pred - n_tp,
        "false_negatives": total_true - n_tp,
        "s1_total": n_s1,
        "s1_zero_match": int(single.sum()),
        "zero_match_predicted_empty": int((single & (pred == 0)).sum()),
        "zero_match_with_false_merge": int((single & (pred > 0)).sum()),
        "s1_with_predictions": int((pred > 0).sum()),
        "predictions_per_s1_hist": np.bincount(np.minimum(pred, 12), minlength=13).tolist(),
    }
