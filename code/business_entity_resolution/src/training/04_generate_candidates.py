"""Stage 04: candidate generation from the inverted index (bounded, chunked, resumable).

Per S1: look up its keys, expand postings, aggregate per target (score = sum of key IDF, family
bitmask, number of shared keys), rescore every aggregated candidate on the GPU with a cheap
name/address similarity (utils/rerank.py), keep the top-K (ties: lower target index).
Output: work/<split>/candidates/part-XXXXX.parquet, one shard per S1_PER_SHARD S1 rows, rows sorted
by (s1, blk_rank). These shards are exactly the pairs the model scores, so they are also the source
of output/candidate_pairs.tsv.
On the training split it measures blocking recall against the ground truth and exits non-zero when
recall falls below --min-recall.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from utils import config  # noqa: E402
from utils.blocking import FAMILIES, BlockingIndex, S1Keys, generate_candidates  # noqa: E402
from utils.gpu import GB, check_ram, get_device, gpu_report, release_gpu  # noqa: E402
from utils.rerank import GPUReranker, target_key_norm  # noqa: E402
from utils.io import (AtomicParquetWriter, add_common_args, clear_stage, fail, limit_threads, log,  # noqa: E402
                      paths_from_args, read_manifest, shard_name, write_json, write_manifest, begin_stage, upstream_stamp)

SCHEMA = pa.schema([("s1", pa.int32()), ("tgt", pa.int32()), ("blk_score", pa.float32()),
                    ("blk_nkeys", pa.uint8()), ("blk_fam", pa.uint8()), ("blk_rank", pa.uint8())])
RECALL_AT = (1, 2, 3, 5, 10, 20)


def pct(hist, q):
    c = np.cumsum(hist)
    return int(np.searchsorted(c, q * c[-1])) if c[-1] else 0


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--k", type=int, default=config.MAX_CANDIDATES_PER_S1)
    ap.add_argument("--max-expanded", type=int, default=config.MAX_EXPANDED_POSTINGS)
    ap.add_argument("--s1-per-shard", type=int, default=config.S1_PER_SHARD)
    ap.add_argument("--no-rerank", action="store_true", help="rank by summed key IDF only")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--max-gpu-gb", type=float, default=config.MAX_GPU_MEMORY_GB)
    ap.add_argument("--min-recall", type=float, default=0.95,
                    help="train only: fail if blocking recall over all GT links is below this")
    args = ap.parse_args()
    if args.k > 255:
        fail("--k must be <= 255 (rank is stored as uint8)")
    limit_threads(args.n_threads)
    paths = paths_from_args(args)
    out_dir = paths.candidates
    iman = read_manifest(paths.index, f"stage 03 ({args.split})")
    sig = {"stage": "04", "index": upstream_stamp(paths.index), "k": args.k, "rerank": not args.no_rerank,
           "rerank_version": 1, "s1_per_shard": args.s1_per_shard,
           "labels": upstream_stamp(paths.labels) if args.split == "train" else None}
    if begin_stage(out_dir, sig, args.force):
        return
    check_ram(0.3 + args.max_expanded * 50 / 2**30, "04 candidates", args.force)
    index = BlockingIndex(paths.index)
    s1k = S1Keys(paths.index)
    n1 = s1k.n
    reranker = None
    if not args.no_rerank:
        norm_path = os.path.join(paths.index, "tnorm.npy")
        keys_path = os.path.join(paths.index, "keys.npy")
        if not os.path.exists(norm_path) or os.path.getmtime(norm_path) < os.path.getmtime(keys_path):
            np.save(norm_path, target_key_norm(index, iman["n_targets"]))
        n_rec = n1 + iman["n_targets"]
        est = n_rec * 136 / GB + 0.6
        log(f"GPU re-rank: compact records {n_rec * 136 / GB:.2f} GB + batch ~0.5 GB (cap {args.max_gpu_gb} GB)")
        if est > args.max_gpu_gb:
            fail(f"re-rank needs ~{est:.1f} GB VRAM > --max-gpu-gb {args.max_gpu_gb}; use --no-rerank or raise the cap")
        check_ram(0.3 + args.max_expanded * 50 / GB + n_rec * 136 / GB, "04 re-rank setup", args.force)
        device = get_device(args.device, args.max_gpu_gb)
        reranker = GPUReranker(np.load(os.path.join(paths.records, "s1.npy"), mmap_mode="r"),
                               np.load(os.path.join(paths.records, "tgt.npy"), mmap_mode="r"),
                               np.load(norm_path), device)
        log(f"re-ranker ready on {device} ({reranker.vram_gb():.2f} GB on device)")
    is_train = args.split == "train"
    if is_train:
        owner = np.load(os.path.join(paths.labels, "owner.npy"), mmap_mode="r")
        n_true = np.load(os.path.join(paths.labels, "n_true.npy"))
        if len(owner) != iman["n_targets"] or len(n_true) != n1:
            fail("label arrays do not match the index; rerun 02/03")

    cand_hist = np.zeros(args.k + 1, np.int64)
    exp_hist = np.zeros(40, np.int64)          # log2 buckets of expanded postings per S1
    pos_rank_hist = np.zeros(args.k, np.int64)
    found_per_s1_complete = 0
    fam_pos = np.zeros(len(FAMILIES), np.int64)
    total_pairs = total_expanded = 0
    n_shards = (n1 + args.s1_per_shard - 1) // args.s1_per_shard
    t0 = time.time()
    for sh in range(n_shards):
        lo, hi = sh * args.s1_per_shard, min(n1, (sh + 1) * args.s1_per_shard)
        dest = os.path.join(out_dir, shard_name(sh))
        if os.path.exists(dest):
            res = {c: pq.read_table(dest, columns=[c]).column(0).to_numpy() for c in ("s1", "tgt", "blk_rank", "blk_fam")}
            per_s1 = None
        else:
            res, per_s1 = generate_candidates(index, s1k, lo, hi, args.k, args.max_expanded, reranker)
            w = AtomicParquetWriter(dest, SCHEMA)
            w.write_table(pa.table({c: pa.array(res[c]) for c in SCHEMA.names}, schema=SCHEMA))
            w.close()
            total_expanded += int(per_s1.sum())
            exp_hist += np.bincount(np.minimum(np.log2(np.maximum(per_s1, 1)).astype(np.int64), 39), minlength=40)
        counts = np.bincount(res["s1"] - lo, minlength=hi - lo)
        cand_hist += np.bincount(np.minimum(counts, args.k), minlength=args.k + 1)
        total_pairs += len(res["s1"])
        if is_train:
            y = np.asarray(owner[res["tgt"]]) == res["s1"]
            pos_rank_hist += np.bincount(res["blk_rank"][y].astype(np.int64), minlength=args.k)
            found = np.bincount(res["s1"][y] - lo, minlength=hi - lo)
            found_per_s1_complete += int((found == n_true[lo:hi]).sum())
            fam = res["blk_fam"][y]
            fam_pos += np.array([int(((fam >> i) & 1).sum()) for i in range(len(FAMILIES))])
        if sh % 5 == 0 or sh == n_shards - 1:
            msg = f"shard {sh + 1}/{n_shards}: {total_pairs:,} pairs ({time.time() - t0:.0f}s)"
            if is_train:
                msg += f", recall so far {pos_rank_hist.sum() / max(int(n_true[:hi].sum()), 1):.4f}"
            log(msg)

    del reranker
    release_gpu()
    stats = {"split": args.split, "k": args.k, "rerank": not args.no_rerank, "gpu": gpu_report(), "s1": n1, "pairs": total_pairs, "s1_per_shard": args.s1_per_shard,
             "n_shards": n_shards,
             "expanded_postings_total": total_expanded,
             "expanded_per_s1_log2_hist": exp_hist.tolist(),
             "candidates_per_s1": {"mean": total_pairs / max(n1, 1), "median": pct(cand_hist, 0.5),
                                   "p95": pct(cand_hist, 0.95), "p99": pct(cand_hist, 0.99),
                                   "max": int(np.flatnonzero(cand_hist)[-1]) if cand_hist.any() else 0,
                                   "zero_candidates": int(cand_hist[0]), "hist": cand_hist.tolist()}}
    log(f"candidates: {total_pairs:,} pairs, per S1 mean {stats['candidates_per_s1']['mean']:.1f} "
        f"median {stats['candidates_per_s1']['median']} p95 {stats['candidates_per_s1']['p95']} "
        f"p99 {stats['candidates_per_s1']['p99']} max {stats['candidates_per_s1']['max']}, "
        f"S1 without candidates {cand_hist[0]:,}")
    if is_train:
        links = int(n_true.sum())
        found = int(pos_rank_hist.sum())
        recall = found / max(links, 1)
        cum = np.cumsum(pos_rank_hist)
        stats["recall"] = {
            "true_links": links, "covered": found, "missed": links - found, "recall": recall,
            "recall_at": {str(r): float(cum[min(r, args.k) - 1] / links) for r in RECALL_AT if r <= args.k},
            "s1_all_matches_covered": found_per_s1_complete / n1,
            "positives_by_family": dict(zip(FAMILIES, fam_pos.tolist())),
        }
        log(f"BLOCKING RECALL {recall:.5f} ({found:,}/{links:,}, missed {links - found:,}); "
            f"recall@k {stats['recall']['recall_at']}; S1 fully covered {found_per_s1_complete / n1:.4f}")
        write_json(paths.artifact("blocking_report_train.json"), stats)
        if recall < args.min_recall:
            fail(f"blocking recall {recall:.4f} < --min-recall {args.min_recall}. Raise --k or the key "
                 f"frequency caps (03) before continuing; downstream stages would inherit this ceiling.")
    else:
        write_json(paths.artifact(f"blocking_report_{args.split}.json"), stats)
    write_manifest(out_dir, stats)


if __name__ == "__main__":
    main()
