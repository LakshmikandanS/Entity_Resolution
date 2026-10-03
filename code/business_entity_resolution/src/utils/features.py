"""Pair features (problem.md decision D4).

Base features are computed on the GPU in fixed-size batches from the 265-byte record arrays.
Competition features are computed afterwards from base features alone (no labels):
  S1 side  : over the candidates of the same S1 (contiguous rows in a shard).
  target side: over all S1s whose candidate list contains the same target (streaming per-target
             best / second-best arrays).
No country feature exists (France is unseen in training).
"""
import numpy as np
import torch

from .normalization import (F_ACCENTED, F_ADDR_EMPTY, F_ADDR_PLACEHOLDER, F_ALLCAPS, F_DOMAIN,
                            F_INDIC_ADDR, F_INDIC_NAME, F_LANDMARK, F_LEGAL_FRONT)
from .blocking import FAMILIES

NAME_FEATURES = [
    "core_tok_jacc", "core_tok_contain", "core_tok_inter", "name_tok_jacc", "core_tri_dice",
    "core_lev_sim", "core_concat_eq", "first_core_eq", "n_core_s1", "n_core_t", "n_core_absdiff",
    "legal_eq", "legal_both", "legal_missing_one",
]
ADDR_FEATURES = [
    "addr_tok_jacc", "addr_tok_contain", "addr_tok_inter", "addr_tri_dice", "addr_lev_sim",
    "hnum_jacc", "hnum_inter", "hnum_conflict", "hnum_missing_t", "hnum_missing_s1",
    "skey_match", "skey_conflict", "state_eq", "state_conflict", "state_missing_one",
    "n_addr_s1", "n_addr_t", "addr_empty_t",
]
FLAG_FEATURES = [
    "src_s3", "t_indic_name", "t_indic_addr", "t_domain", "t_allcaps", "t_accented",
    "t_legal_front", "t_addr_placeholder", "t_landmark", "s1_landmark",
]
BLOCK_FEATURES = ["blk_score", "blk_nkeys", "blk_rank"] + [f"fam_{f}" for f in FAMILIES]
HEUR_FEATURES = ["h_score", "name_best", "addr_best"]
BASE_FEATURES = NAME_FEATURES + ADDR_FEATURES + FLAG_FEATURES + BLOCK_FEATURES + HEUR_FEATURES
COMP_FEATURES = [
    "s1_ncand", "s1_rank_h", "s1_h_margin", "s1_h_ratio", "s1_blk_margin", "s1_blk_ratio",
    "t_nsuitors", "t_h_margin", "t_blk_margin", "t_is_best_h",
]
FEATURE_COLUMNS = BASE_FEATURES + COMP_FEATURES
META_COLUMNS = ["s1", "tgt"]

GPU_FEATURES = NAME_FEATURES + ADDR_FEATURES + FLAG_FEATURES + HEUR_FEATURES


def estimate_batch_vram_bytes(batch):
    """Upper bound of live tensors for one feature batch (see docs/RESOURCE_ESTIMATES.md 2.7)."""
    per_pair = (62 * 62 + 30 * 30 + 16 * 16 + 8 * 8 + 6 * 6) * 1  # bool equality cubes
    per_pair += 2 * 265 + 65 * 4 * 3 + 33 * 4 * 3                  # inputs, Levenshtein rows
    per_pair += (62 + 30) * 4 * 2                                 # trigram ids
    per_pair += len(GPU_FEATURES) * 4 * 2                          # outputs and temporaries
    return int(batch * per_pair * 1.5)                            # allocator slack


# --------------------------------------------------------------------------- GPU kernels
def _nan_div(a, b):
    a, b = a.float(), b.float()
    return torch.where(b > 0, a / b.clamp(min=1e-9), torch.full_like(a, float("nan")))


def _set_stats(a, b):
    """a [B,La], b [B,Lb] int32 hashed sets (0 = pad). -> (inter, na, nb)."""
    va, vb = a != 0, b != 0
    eq = (a.unsqueeze(2) == b.unsqueeze(1)) & va.unsqueeze(2) & vb.unsqueeze(1)
    inter = eq.any(2).sum(1)
    return inter, va.sum(1), vb.sum(1)


def _trigrams(c):
    c = c.int()
    t = c[:, :-2] * 65536 + c[:, 1:-1] * 256 + c[:, 2:]
    return torch.where(c[:, 2:] != 0, t, torch.zeros_like(t))


def _dice(ta, tb):
    va, vb = ta != 0, tb != 0
    eq = (ta.unsqueeze(2) == tb.unsqueeze(1)) & va.unsqueeze(2) & vb.unsqueeze(1)
    ia, ib = eq.any(2).sum(1), eq.any(1).sum(1)
    del eq
    return _nan_div(ia + ib, va.sum(1) + vb.sum(1))


def _lev_sim(a, la, b, lb):
    """Batched Levenshtein similarity 1 - d / max(len). Row recurrence with a cummin for inserts."""
    B, La = a.shape
    Lb = b.shape[1]
    dev = a.device
    la, lb = la.long(), lb.long()
    ar = torch.arange(Lb + 1, device=dev, dtype=torch.int32)
    prev = ar.unsqueeze(0).expand(B, Lb + 1).contiguous()
    max_la = int(la.max().item()) if B else 0
    for i in range(1, min(La, max_la) + 1):
        cost = (a[:, i - 1:i] != b).int()
        tmp = torch.minimum(prev[:, 1:] + 1, prev[:, :-1] + cost)
        v = torch.cat([torch.full((B, 1), i, device=dev, dtype=torch.int32), tmp], 1)
        row = torch.cummin(v - ar, dim=1).values + ar
        prev = torch.where((la >= i).unsqueeze(1), row, prev)
    d = prev.gather(1, lb.unsqueeze(1)).squeeze(1).float()
    m = torch.maximum(la, lb).float()
    return torch.where(m > 0, 1.0 - d / m.clamp(min=1), torch.full_like(d, float("nan")))


def _t(block, field, device):
    return torch.from_numpy(np.ascontiguousarray(block[field])).to(device, non_blocking=True)


def gpu_pair_features(rs, rt, src_s3, device):
    """rs, rt: structured record arrays (S1, target) aligned by pair. Returns float32 [B, len(GPU_FEATURES)]."""
    f = {}
    # ---- names
    a, b = _t(rs, "core_tok", device), _t(rt, "core_tok", device)
    inter, na, nb = _set_stats(a, b)
    f["core_tok_jacc"] = _nan_div(inter, na + nb - inter)
    f["core_tok_contain"] = _nan_div(inter, torch.minimum(na, nb))
    f["core_tok_inter"] = inter.float()
    f["first_core_eq"] = ((a[:, 0] == b[:, 0]) & (a[:, 0] != 0)).float()
    f["n_core_s1"], f["n_core_t"] = na.float(), nb.float()
    f["n_core_absdiff"] = (na - nb).abs().float()
    del a, b
    i2, n2a, n2b = _set_stats(_t(rs, "name_tok", device), _t(rt, "name_tok", device))
    f["name_tok_jacc"] = _nan_div(i2, n2a + n2b - i2)
    ca, cb = _t(rs, "core_chars", device), _t(rt, "core_chars", device)
    f["core_tri_dice"] = _dice(_trigrams(ca), _trigrams(cb))
    f["core_lev_sim"] = _lev_sim(ca, _t(rs, "core_len", device), cb, _t(rt, "core_len", device))
    del ca, cb
    cc_s, cc_t = _t(rs, "core_concat", device), _t(rt, "core_concat", device)
    f["core_concat_eq"] = ((cc_s == cc_t) & (cc_s != 0)).float()
    lg_s, lg_t = _t(rs, "legal", device), _t(rt, "legal", device)
    both = (lg_s != 0) & (lg_t != 0)
    f["legal_eq"] = ((lg_s == lg_t) & both).float()
    f["legal_both"] = both.float()
    f["legal_missing_one"] = ((lg_s != 0) ^ (lg_t != 0)).float()
    # ---- addresses
    inter, na, nb = _set_stats(_t(rs, "addr_tok", device), _t(rt, "addr_tok", device))
    f["addr_tok_jacc"] = _nan_div(inter, na + nb - inter)
    f["addr_tok_contain"] = _nan_div(inter, torch.minimum(na, nb))
    f["addr_tok_inter"] = inter.float()
    f["n_addr_s1"], f["n_addr_t"] = na.float(), nb.float()
    aa, ab = _t(rs, "addr_chars", device), _t(rt, "addr_chars", device)
    f["addr_tri_dice"] = _dice(_trigrams(aa), _trigrams(ab))
    f["addr_lev_sim"] = _lev_sim(aa, _t(rs, "addr_len", device), ab, _t(rt, "addr_len", device))
    del aa, ab
    hi, hna, hnb = _set_stats(_t(rs, "hnum", device), _t(rt, "hnum", device))
    f["hnum_jacc"] = _nan_div(hi, hna + hnb - hi)
    f["hnum_inter"] = hi.float()
    f["hnum_conflict"] = ((hna > 0) & (hnb > 0) & (hi == 0)).float()
    f["hnum_missing_t"] = (hnb == 0).float()
    f["hnum_missing_s1"] = (hna == 0).float()
    si, sna, snb = _set_stats(_t(rs, "skey", device), _t(rt, "skey", device))
    f["skey_match"] = (si > 0).float()
    f["skey_conflict"] = ((sna > 0) & (snb > 0) & (si == 0)).float()
    st_s, st_t = _t(rs, "state", device), _t(rt, "state", device)
    sboth = (st_s != 0) & (st_t != 0)
    f["state_eq"] = ((st_s == st_t) & sboth).float()
    f["state_conflict"] = ((st_s != st_t) & sboth).float()
    f["state_missing_one"] = ((st_s != 0) ^ (st_t != 0)).float()
    # ---- flags (target side unless prefixed s1_)
    fl_t = _t(rt, "flags", device).int()
    fl_s = _t(rs, "flags", device).int()
    bit = lambda x, m: ((x & m) != 0).float()
    f["addr_empty_t"] = bit(fl_t, F_ADDR_EMPTY)
    f["src_s3"] = torch.from_numpy(src_s3.astype(np.float32)).to(device)
    f["t_indic_name"] = bit(fl_t, F_INDIC_NAME)
    f["t_indic_addr"] = bit(fl_t, F_INDIC_ADDR)
    f["t_domain"] = bit(fl_t, F_DOMAIN)
    f["t_allcaps"] = bit(fl_t, F_ALLCAPS)
    f["t_accented"] = bit(fl_t, F_ACCENTED)
    f["t_legal_front"] = bit(fl_t, F_LEGAL_FRONT)
    f["t_addr_placeholder"] = bit(fl_t, F_ADDR_PLACEHOLDER)
    f["t_landmark"] = bit(fl_t, F_LANDMARK)
    f["s1_landmark"] = bit(fl_s, F_LANDMARK)
    # ---- label-free heuristic used only to build competition features and hard negatives
    z = lambda x: torch.nan_to_num(x, nan=0.0)
    f["name_best"] = torch.maximum(z(f["core_tok_jacc"]), z(f["core_tri_dice"]))
    f["addr_best"] = torch.maximum(z(f["addr_tok_jacc"]), z(f["addr_tri_dice"]))
    f["h_score"] = 0.5 * f["name_best"] + 0.5 * f["addr_best"]
    return torch.stack([f[n] for n in GPU_FEATURES], 1).float()


def block_features(cand):
    """Blocking-derived features straight from the candidate columns (CPU, numpy)."""
    out = {"blk_score": cand["blk_score"].astype(np.float32),
           "blk_nkeys": cand["blk_nkeys"].astype(np.float32),
           "blk_rank": cand["blk_rank"].astype(np.float32)}
    fam = cand["blk_fam"].astype(np.uint8)
    for i, name in enumerate(FAMILIES):
        out[f"fam_{name}"] = ((fam >> i) & 1).astype(np.float32)
    return out


# --------------------------------------------------------------------------- competition (CPU)
def _group_starts(keys):
    return np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]]) if len(keys) else np.zeros(0, np.int64)


def _max_other_grouped(x, starts, gid):
    """For each row: max of x over the OTHER rows of its group (-inf when alone)."""
    gmax = np.maximum.reduceat(x, starts)
    is_max = x == gmax[gid]
    idx = np.flatnonzero(is_max)
    g = gid[idx]
    first = idx[np.r_[True, g[1:] != g[:-1]]]
    x2 = x.astype(np.float64)
    x2[first] = -np.inf
    gsec = np.maximum.reduceat(x2, starts)
    other = gmax[gid].astype(np.float64)
    other[first] = gsec[gid[first]]
    return other, gmax[gid]


def s1_competition(s1, tgt, h, blk):
    """Rows must be grouped by s1 (true for candidate shards). Returns dict of float32 arrays."""
    n = len(s1)
    if n == 0:
        return {k: np.zeros(0, np.float32) for k in COMP_FEATURES[:6]}
    starts = _group_starts(s1)
    sizes = np.diff(np.r_[starts, n])
    gid = np.repeat(np.arange(len(starts)), sizes)
    h_other, h_max = _max_other_grouped(h, starts, gid)
    b_other, b_max = _max_other_grouped(blk, starts, gid)
    order = np.lexsort((tgt, -blk, -h, gid))
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n) - np.repeat(starts, sizes)
    margin = lambda x, o: np.where(np.isfinite(o), x - o, x).astype(np.float32)
    ratio = lambda x, m: np.where(m > 0, x / np.maximum(m, 1e-9), 0.0).astype(np.float32)
    return {
        "s1_ncand": np.repeat(sizes, sizes).astype(np.float32),
        "s1_rank_h": rank.astype(np.float32),
        "s1_h_margin": margin(h, h_other),
        "s1_h_ratio": ratio(h, h_max),
        "s1_blk_margin": margin(blk, b_other),
        "s1_blk_ratio": ratio(blk, b_max),
    }


class TargetCompetition:
    """Two streaming passes over all candidate rows of a split, arrays sized by #targets.

    pass 1: suitor count, best h, best blocking score per target
    pass 2: multiplicity of the best value and the best value strictly below it
    max over OTHER suitors = best if (row is below best or best is shared) else strict second."""

    def __init__(self, n_targets):
        self.cnt = np.zeros(n_targets, np.int32)
        self.best_h = np.full(n_targets, -np.inf, np.float32)
        self.best_b = np.full(n_targets, -np.inf, np.float32)
        self.nbest_h = np.zeros(n_targets, np.int32)
        self.nbest_b = np.zeros(n_targets, np.int32)
        self.sec_h = np.full(n_targets, -np.inf, np.float32)
        self.sec_b = np.full(n_targets, -np.inf, np.float32)

    def pass1(self, tgt, h, blk):
        np.add.at(self.cnt, tgt, 1)
        np.maximum.at(self.best_h, tgt, h.astype(np.float32))
        np.maximum.at(self.best_b, tgt, blk.astype(np.float32))

    def pass2(self, tgt, h, blk):
        h, blk = h.astype(np.float32), blk.astype(np.float32)
        for x, best, nbest, sec in ((h, self.best_h, self.nbest_h, self.sec_h),
                                    (blk, self.best_b, self.nbest_b, self.sec_b)):
            is_best = x == best[tgt]
            np.add.at(nbest, tgt[is_best], 1)
            np.maximum.at(sec, tgt[~is_best], x[~is_best])

    def features(self, tgt, h, blk):
        h, blk = h.astype(np.float32), blk.astype(np.float32)
        out = {"t_nsuitors": self.cnt[tgt].astype(np.float32)}
        for name, x, best, nbest, sec in (("t_h_margin", h, self.best_h, self.nbest_h, self.sec_h),
                                          ("t_blk_margin", blk, self.best_b, self.nbest_b, self.sec_b)):
            bt = best[tgt]
            other = np.where((x < bt) | (nbest[tgt] > 1), bt, sec[tgt])
            out[name] = np.where(np.isfinite(other), x - other, x).astype(np.float32)
        out["t_is_best_h"] = (h >= self.best_h[tgt]).astype(np.float32)
        return out

    def nbytes(self):
        return sum(a.nbytes for a in vars(self).values())
