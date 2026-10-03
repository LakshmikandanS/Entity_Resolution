"""Multi-family inverted-index blocking (problem.md decision D3).

Key families (coverage of true links from problem.md section 3.4):
  N core-name token (92.0% share >=1), P core-token pair (79.4%), C concatenated core (64.8%),
  X core token x house number (71.9%), S street key (55.8%), A identifying address word (93.5%).
Every key is hashed together with the record's country, so blocking is a generic country-equality
test (0 cross-country links in training) that works unchanged for unseen countries.

Index layout (CSR, memory-mapped): keys (sorted uint64), offs (int64, len U+1), post (int32 target
index), fam (uint8), idf (float32). Keys are built in hash-range partitions so peak RAM during the
sort is bounded; concatenating partitions in order gives a globally sorted array.
"""
import os
import zlib

import numpy as np

FAMILIES = ("N", "P", "C", "X", "S", "A")
FAM_ID = {f: i for i, f in enumerate(FAMILIES)}
COMMON_FAMILIES = ("N", "A")
MAX_KEYS_PER_RECORD = 6 + 6 + 1 + 8 + 3 + 8


def hash64(text):
    b = text.encode("ascii", "ignore")
    return (zlib.crc32(b) << 32) | zlib.crc32(b, 0x9E3779B9)


def record_keys(country_key, core, hnums, skeys, idwords):
    """-> list of (uint64 key, family id), unique per record."""
    out = {}
    c = country_key + "\x1f"

    def add(fam, value):
        out[hash64(c + fam + "\x1f" + value)] = FAM_ID[fam]

    core_u = list(dict.fromkeys(t for t in core if len(t) >= 2))
    for t in core_u[:6]:
        add("N", t)
    head = core_u[:4]
    for i in range(len(head)):
        for j in range(i + 1, len(head)):
            a, b = sorted((head[i], head[j]))
            add("P", a + " " + b)
    concat = "".join(core)
    if len(concat) >= 3:
        add("C", concat)
    for t in head:
        for n in hnums[:2]:
            add("X", t + " " + n)
    for s in skeys[:3]:
        add("S", s)
    for w in idwords[:8]:
        add("A", w)
    return list(out.items())


def family_caps(max_key_frequency, max_key_frequency_common):
    return np.array([max_key_frequency_common if f in COMMON_FAMILIES else max_key_frequency
                     for f in FAMILIES], dtype=np.int64)


# --------------------------------------------------------------------------- index build
class PartitionedKeyWriter:
    """Spills (key, target index, family) triples into hash-range partition files."""

    def __init__(self, directory, n_partitions):
        assert n_partitions & (n_partitions - 1) == 0, "partitions must be a power of two"
        self.dir = directory
        self.n = n_partitions
        self.shift = 64 - int(np.log2(n_partitions)) if n_partitions > 1 else 64
        self.files = []
        for p in range(n_partitions):
            base = os.path.join(directory, f"tmpkeys-{p:03d}")
            self.files.append(tuple(open(base + ext, "wb") for ext in (".u64", ".i32", ".u8")))
        self.count = 0

    def write(self, keys, idx, fam):
        part = (keys >> np.uint64(self.shift)).astype(np.int64) if self.n > 1 else np.zeros(len(keys), np.int64)
        order = np.argsort(part, kind="stable")
        part, keys, idx, fam = part[order], keys[order], idx[order], fam[order]
        bounds = np.searchsorted(part, np.arange(self.n + 1))
        for p in range(self.n):
            a, b = bounds[p], bounds[p + 1]
            if b > a:
                fk, fi, ff = self.files[p]
                keys[a:b].tofile(fk)
                idx[a:b].tofile(fi)
                fam[a:b].tofile(ff)
        self.count += len(keys)

    def close(self):
        for fs in self.files:
            for f in fs:
                f.close()

    def partition_paths(self, p):
        base = os.path.join(self.dir, f"tmpkeys-{p:03d}")
        return base + ".u64", base + ".i32", base + ".u8"


def build_partition_csr(paths, caps):
    """Sort one partition and drop keys whose document frequency exceeds the family cap.
    Returns dict(keys, df, fam, post) plus per-family stats."""
    keys = np.fromfile(paths[0], dtype=np.uint64)
    idx = np.fromfile(paths[1], dtype=np.int32)
    fam = np.fromfile(paths[2], dtype=np.uint8)
    if len(keys) == 0:
        empty = {"keys": np.zeros(0, np.uint64), "df": np.zeros(0, np.int64),
                 "fam": np.zeros(0, np.uint8), "post": np.zeros(0, np.int32)}
        return empty, {}
    order = np.argsort(keys, kind="stable")
    keys, idx, fam = keys[order], idx[order], fam[order]
    del order
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    df = np.diff(np.r_[starts, len(keys)])
    ukeys, ufam = keys[starts], fam[starts]
    del keys, fam
    keep = df <= caps[ufam]
    stats = {}
    for f, name in enumerate(FAMILIES):
        m = ufam == f
        stats[name] = {
            "keys": int(m.sum()), "postings": int(df[m].sum()),
            "keys_dropped": int((m & ~keep).sum()), "postings_dropped": int(df[m & ~keep].sum()),
            "df_hist": np.bincount(np.minimum(np.log2(np.maximum(df[m], 1)).astype(np.int64), 20),
                                   minlength=21).tolist(),
        }
    post = idx[np.repeat(keep, df)]
    return {"keys": ukeys[keep], "df": df[keep], "fam": ufam[keep], "post": post}, stats


# --------------------------------------------------------------------------- lookup
class BlockingIndex:
    def __init__(self, directory):
        load = lambda n: np.load(os.path.join(directory, n), mmap_mode="r")
        self.keys = load("keys.npy")
        self.offs = load("offs.npy")
        self.post = load("post.npy")
        self.fam = load("fam.npy")
        self.idf = load("idf.npy")

    def lookup(self, q):
        """Return (pos, found) for query keys; queries are searched in sorted order for locality."""
        if len(q) == 0 or len(self.keys) == 0:
            return np.zeros(len(q), np.int64), np.zeros(len(q), bool)
        order = np.argsort(q, kind="stable")
        pos_sorted = np.searchsorted(self.keys, q[order])
        pos = np.empty_like(pos_sorted)
        pos[order] = pos_sorted
        pos = np.minimum(pos, len(self.keys) - 1)
        found = np.asarray(self.keys[pos]) == q
        return pos, found


class S1Keys:
    def __init__(self, directory):
        self.keys = np.memmap(os.path.join(directory, "s1_keys.u64"), dtype=np.uint64, mode="r")
        self.offs = np.load(os.path.join(directory, "s1_key_offs.npy"), mmap_mode="r")
        self.n = len(self.offs) - 1


def generate_candidates(index, s1keys, lo, hi, k, max_expanded):
    """Top-k candidates for S1 rows [lo, hi). Returns dict of arrays sorted by (s1, rank) and the
    total number of expanded postings (for reporting)."""
    ko = np.asarray(s1keys.offs[lo:hi + 1])
    q = np.asarray(s1keys.keys[ko[0]:ko[-1]])
    q_s1 = np.repeat(np.arange(hi - lo, dtype=np.int32), np.diff(ko))
    pos, found = index.lookup(q)
    pos, q_s1 = pos[found], q_s1[found]
    starts = np.asarray(index.offs[pos])
    df = np.asarray(index.offs[pos + 1]) - starts
    per_s1 = np.bincount(q_s1, weights=df, minlength=hi - lo).astype(np.int64)

    out = {n: [] for n in ("s1", "tgt", "blk_score", "blk_nkeys", "blk_fam", "blk_rank")}
    # sub-chunks of S1 rows so that expanded postings stay below max_expanded
    cum = np.cumsum(per_s1)
    a = 0
    while a < hi - lo:
        base = cum[a - 1] if a > 0 else 0
        b = int(np.searchsorted(cum, base + max_expanded, side="right"))
        b = max(b, a + 1)
        m = (q_s1 >= a) & (q_s1 < b)
        _expand(index, q_s1[m], pos[m], starts[m], df[m], lo, k, out)
        a = b
    res = {n: (np.concatenate(v) if v else np.zeros(0)) for n, v in out.items()}
    dt = {"s1": np.int32, "tgt": np.int32, "blk_score": np.float32, "blk_nkeys": np.uint8,
          "blk_fam": np.uint8, "blk_rank": np.uint8}
    res = {n: res[n].astype(dt[n], copy=False) for n in res}
    return res, per_s1


def _expand(index, q_s1, pos, starts, df, lo, k, out):
    total = int(df.sum())
    if total == 0:
        return
    rep_s1 = np.repeat(q_s1, df)
    run_start = np.cumsum(df) - df
    idx = np.repeat(starts - run_start, df) + np.arange(total, dtype=np.int64)
    tgt = np.asarray(index.post[idx])
    del idx
    w = np.repeat(np.asarray(index.idf[pos]), df)
    fb = np.repeat((np.uint8(1) << np.asarray(index.fam[pos])).astype(np.uint8), df)
    pk = (rep_s1.astype(np.int64) << 32) | tgt.astype(np.int64)
    del rep_s1, tgt
    order = np.argsort(pk, kind="stable")
    pk, w, fb = pk[order], w[order], fb[order]
    del order
    bounds = np.flatnonzero(np.r_[True, pk[1:] != pk[:-1]])
    score = np.add.reduceat(w, bounds).astype(np.float32)
    bits = np.bitwise_or.reduceat(fb, bounds)
    nkeys = np.diff(np.r_[bounds, len(pk)])
    upk = pk[bounds]
    del pk, w, fb
    s1l = (upk >> 32).astype(np.int32)
    tg = (upk & 0xFFFFFFFF).astype(np.int32)
    o2 = np.lexsort((tg, -score, s1l))
    s1l, tg, score, bits, nkeys = s1l[o2], tg[o2], score[o2], bits[o2], nkeys[o2]
    gstart = np.flatnonzero(np.r_[True, s1l[1:] != s1l[:-1]])
    rank = np.arange(len(s1l)) - np.repeat(gstart, np.diff(np.r_[gstart, len(s1l)]))
    keep = rank < k
    out["s1"].append(s1l[keep] + lo)
    out["tgt"].append(tg[keep])
    out["blk_score"].append(score[keep])
    out["blk_nkeys"].append(np.minimum(nkeys[keep], 255))
    out["blk_fam"].append(bits[keep])
    out["blk_rank"].append(rank[keep])
