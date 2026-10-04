"""Stage 06: reproducible training rows from the train feature shards.

Kept (weight 1):
  * every positive found by blocking
  * hard negatives: blocking rank < --hard-rank, OR S1-side heuristic rank < --hard-rank, OR
    h_score >= --hard-h, OR the S1 is this target's best suitor by h (t_is_best_h)
Kept (weight 1 / --easy-rate): remaining "easy" negatives selected by a hash of (s1, tgt) - not by
a random generator - so the sample is identical across runs, chunkings and machines.
Optional --entity-frac keeps whole S1 entities by hash (never splits an S1's candidates).

Hard-negative categories reported (overlapping): same-core-name competitor (core_concat_eq or
core_tok_jacc >= 0.5, with addr_tok_jacc < 0.3), address-conflicting (hnum_conflict, skey_conflict
or state_conflict), high blocking rank.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from utils import config  # noqa: E402
from utils.gpu import check_ram  # noqa: E402
from utils.io import (AtomicParquetWriter, add_common_args, clear_stage, disk_size_gb, limit_threads,  # noqa: E402
                      list_shards, log, paths_from_args, read_manifest, begin_stage, upstream_stamp, write_json,
                      write_manifest)

M = np.uint64(1_000_003)


def hash_unit(s1, tgt, salt):
    """Deterministic pseudo-uniform value in [0, 1) per (s1, tgt)."""
    x = (s1.astype(np.uint64) * np.uint64(0x9E3779B1) + tgt.astype(np.uint64) * np.uint64(0x85EBCA77)
         + np.uint64(salt)) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(15)
    x = (x * np.uint64(0x2C1B3C6D)) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(12)
    return (x % M).astype(np.float64) / float(M)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__), split=False)
    ap.add_argument("--hard-rank", type=int, default=6)
    ap.add_argument("--hard-h", type=float, default=0.5)
    ap.add_argument("--easy-rate", type=float, default=0.02)
    ap.add_argument("--entity-frac", type=float, default=1.0)
    args = ap.parse_args()
    args.split = "train"
    limit_threads(args.n_threads)
    paths = paths_from_args(args)
    out_dir = paths.trainset
    read_manifest(paths.features, "stage 05 (train)")
    sig = {"stage": "06", "features": upstream_stamp(paths.features), "hard_rank": args.hard_rank,
           "hard_h": args.hard_h, "easy_rate": args.easy_rate, "entity_frac": args.entity_frac}
    if begin_stage(out_dir, sig, args.force):
        return
    check_ram(1.2, "06 training set", args.force)
    st = {k: 0 for k in ("candidates", "positives", "negatives_kept", "hard_negatives", "easy_negatives",
                         "discarded", "hn_same_name", "hn_addr_conflict", "hn_high_rank", "entities_dropped_rows")}
    for p in list_shards(paths.features):
        dest = os.path.join(out_dir, os.path.basename(p))
        t = pq.read_table(p)
        g = lambda c: t.column(c).to_numpy()
        s1, tgt, y = g("s1"), g("tgt"), g("label").astype(bool)
        n = len(y)
        if args.entity_frac < 1.0:
            ent = hash_unit(s1, np.zeros_like(s1), 7) < args.entity_frac
        else:
            ent = np.ones(n, bool)
        hard = ((g("blk_rank") < args.hard_rank) | (g("s1_rank_h") < args.hard_rank)
                | (g("h_score") >= args.hard_h) | (g("t_is_best_h") > 0)) & ~y
        easy = ~y & ~hard & (hash_unit(s1, tgt, 11) < args.easy_rate)
        keep = ent & (y | hard | easy)
        weight = np.where(easy, 1.0 / args.easy_rate, 1.0).astype(np.float32)
        same_name = ((g("core_concat_eq") > 0) | (g("core_tok_jacc") >= 0.5)) & (np.nan_to_num(g("addr_tok_jacc")) < 0.3)
        conflict = (g("hnum_conflict") > 0) | (g("skey_conflict") > 0) | (g("state_conflict") > 0)
        st["candidates"] += n
        st["positives"] += int((keep & y).sum())
        st["hard_negatives"] += int((keep & hard).sum())
        st["easy_negatives"] += int((keep & easy).sum())
        st["negatives_kept"] += int((keep & ~y).sum())
        st["discarded"] += int((~keep).sum())
        st["entities_dropped_rows"] += int((~ent).sum())
        st["hn_same_name"] += int((keep & hard & same_name).sum())
        st["hn_addr_conflict"] += int((keep & hard & conflict).sum())
        st["hn_high_rank"] += int((keep & hard & (g("blk_rank") < args.hard_rank)).sum())
        sub = t.filter(pa.array(keep)).append_column("weight", pa.array(weight[keep]))
        w = AtomicParquetWriter(dest, sub.schema)
        w.write_table(sub)
        w.close()
        del t, sub
    rows = st["positives"] + st["negatives_kept"]
    shards = list_shards(out_dir)
    st.update({"rows": rows, "pos_neg_ratio": st["positives"] / max(st["negatives_kept"], 1),
               "hard_negative_share": st["hard_negatives"] / max(st["negatives_kept"], 1),
               "disk_gb": disk_size_gb(shards), "params": vars(args) | {"seed_salts": [7, 11]},
               "memory_bytes_float32": rows * 70 * 4})
    log(f"training set: {rows:,} rows | positives {st['positives']:,} | negatives {st['negatives_kept']:,} "
        f"(hard {st['hard_negatives']:,}, easy {st['easy_negatives']:,}) | ratio {st['pos_neg_ratio']:.3f} | "
        f"hard share {st['hard_negative_share']:.3f} | discarded {st['discarded']:,} | "
        f"{st['disk_gb']:.2f} GB on disk")
    write_json(paths.artifact("trainset_report.json"), st)
    write_manifest(out_dir, st)


if __name__ == "__main__":
    main()
