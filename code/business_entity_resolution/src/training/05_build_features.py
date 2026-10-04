"""Stage 05: pair features for every candidate (GPU batches) + label-free competition features.

pass A (GPU): candidate shard -> base feature shard (work/<split>/features_base/)
pass B (CPU): two streaming passes accumulate per-target best / second-best of h_score and blk_score
pass C (CPU): per shard, S1-side competition (rows of one S1 are contiguous) + target-side
              competition -> final shard (work/<split>/features/), base shard deleted unless --keep-base
Train shards additionally carry label (owner[tgt] == s1), fold and es_val (entity-level, from the
S1 id hash). Labels are never used to compute any feature.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

from utils import config  # noqa: E402
from utils.features import (BASE_FEATURES, COMP_FEATURES, FEATURE_COLUMNS, GPU_FEATURES,  # noqa: E402
                            TargetCompetition, block_features, estimate_batch_vram_bytes, gpu_pair_features,
                            s1_competition)
from utils.gpu import GB, check_ram, get_device, gpu_report, release_gpu  # noqa: E402
from utils.io import (AtomicParquetWriter, add_common_args, clear_stage, disk_size_gb, fail,  # noqa: E402
                      limit_threads, list_shards, log, paths_from_args, read_manifest, begin_stage, upstream_stamp,
                      write_manifest)

BASE_SCHEMA = pa.schema([("s1", pa.int32()), ("tgt", pa.int32())] + [(f, pa.float32()) for f in BASE_FEATURES])
TRAIN_EXTRA = [("label", pa.int8()), ("fold", pa.int8()), ("es_val", pa.bool_())]


def final_schema(is_train):
    fields = [("s1", pa.int32()), ("tgt", pa.int32())] + [(f, pa.float32()) for f in FEATURE_COLUMNS]
    return pa.schema(fields + (TRAIN_EXTRA if is_train else []))


def compute_shard_base(cand_path, dest, rec_s1, rec_t, n2, device, batch):
    cand = pq.read_table(cand_path)
    cols = {c: cand.column(c).to_numpy() for c in cand.column_names}
    n = len(cols["s1"])
    writer = AtomicParquetWriter(dest, BASE_SCHEMA)
    i = 0
    while i < n:
        j = min(n, i + batch)
        s1b, tb = cols["s1"][i:j], cols["tgt"][i:j]
        order = np.argsort(tb, kind="stable")           # sorted gather = better memmap locality
        rt = np.empty(j - i, dtype=rec_t.dtype)
        rt[order] = rec_t[tb[order]]
        rs = rec_s1[s1b]
        try:
            with torch.no_grad():
                g = gpu_pair_features(rs, rt, tb >= n2, device).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            release_gpu()
            if batch <= 1024:
                raise
            batch //= 2
            log(f"CUDA OOM: halving feature batch to {batch:,}")
            continue
        out = {"s1": s1b, "tgt": tb}
        out.update({name: g[:, k] for k, name in enumerate(GPU_FEATURES)})
        out.update(block_features({c: v[i:j] for c, v in cols.items()}))
        writer.write_table(pa.table({c: pa.array(out[c]) for c in BASE_SCHEMA.names}, schema=BASE_SCHEMA))
        i = j
    writer.close()
    return n, batch


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--max-gpu-gb", type=float, default=config.MAX_GPU_MEMORY_GB)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--keep-base", action="store_true")
    args = ap.parse_args()
    limit_threads(args.n_threads)
    torch.set_num_threads(args.n_threads)
    paths = paths_from_args(args)
    iman = read_manifest(paths.index, f"stage 03 ({args.split})")
    read_manifest(paths.candidates, f"stage 04 ({args.split})")
    is_train = args.split == "train"
    sig = {"stage": "05", "candidates": upstream_stamp(paths.candidates), "features": FEATURE_COLUMNS,
           "labels": upstream_stamp(paths.labels) if is_train else None}
    if begin_stage(paths.features, sig, args.force):
        return
    begin_stage(paths.features_base, sig, args.force)
    n_t, n2 = iman["n_targets"], iman["n2"]
    est_vram = estimate_batch_vram_bytes(args.batch_size) / GB
    log(f"feature batch {args.batch_size:,} pairs -> estimated VRAM {est_vram:.2f} GB (cap {args.max_gpu_gb} GB)")
    while est_vram > args.max_gpu_gb * 0.8 and args.batch_size > 4096:
        args.batch_size //= 2
        est_vram = estimate_batch_vram_bytes(args.batch_size) / GB
        log(f"  reduced batch to {args.batch_size:,} ({est_vram:.2f} GB)")
    check_ram(0.6 + TargetCompetition(0).nbytes() + n_t * 24 / GB + 1.6e6 * 70 * 4 * 2 / GB,
              "05 features", args.force)
    device = get_device(args.device, args.max_gpu_gb)
    log(f"device {device}: {gpu_report().get('name', 'cpu')}")
    rec_s1 = np.load(os.path.join(paths.records, "s1.npy"), mmap_mode="r")
    rec_t = np.load(os.path.join(paths.records, "tgt.npy"), mmap_mode="r")
    cand_shards = list_shards(paths.candidates)

    # ---- pass A
    t0 = time.time()
    total, batch = 0, args.batch_size
    for k, cp in enumerate(cand_shards):
        name = os.path.basename(cp)
        final = os.path.join(paths.features, name)
        dest = os.path.join(paths.features_base, name)
        if os.path.exists(final) or os.path.exists(dest):
            continue
        n, batch = compute_shard_base(cp, dest, rec_s1, rec_t, n2, device, batch)
        total += n
        log(f"pass A {k + 1}/{len(cand_shards)}: {total:,} pairs, {total / max(time.time() - t0, 1e-6):,.0f} pairs/s")
    release_gpu()

    # ---- pass B (target-side competition, label-free)
    tc = TargetCompetition(n_t)
    pending = [(os.path.join(paths.features_base, os.path.basename(c)), os.path.join(paths.features, os.path.basename(c)))
               for c in cand_shards]
    srcs = [b if os.path.exists(b) else f for b, f in pending]
    for step in (1, 2):
        for p in srcs:
            t = pq.read_table(p, columns=["tgt", "h_score", "blk_score"])
            tg, h, bl = (t.column(i).to_numpy() for i in range(3))
            (tc.pass1 if step == 1 else tc.pass2)(tg, h, bl)
        log(f"pass B step {step} done")

    # ---- pass C
    schema = final_schema(is_train)
    if is_train:
        owner = np.load(os.path.join(paths.labels, "owner.npy"), mmap_mode="r")
        bucket = np.load(os.path.join(paths.labels, "s1_bucket.npy"))
        n_folds = read_manifest(paths.labels, "stage 02")["n_folds"]
    for base, final in pending:
        if os.path.exists(final):
            continue
        t = pq.read_table(base)
        cols = {c: t.column(c).to_numpy() for c in t.column_names}
        del t
        comp = s1_competition(cols["s1"], cols["tgt"], cols["h_score"], cols["blk_score"])
        comp.update(tc.features(cols["tgt"], cols["h_score"], cols["blk_score"]))
        cols.update(comp)
        if is_train:
            cols["label"] = (np.asarray(owner[cols["tgt"]]) == cols["s1"]).astype(np.int8)
            b = bucket[cols["s1"]].astype(np.int64)
            cols["fold"] = (b % n_folds).astype(np.int8)
            cols["es_val"] = ((b // n_folds) % 20) == 0
        missing = [c for c in schema.names if c not in cols]
        if missing:
            fail(f"internal error: missing columns {missing}")
        w = AtomicParquetWriter(final, schema)
        w.write_table(pa.table({c: pa.array(cols[c]) for c in schema.names}, schema=schema))
        w.close()
        if not args.keep_base:
            os.remove(base)
    shards = list_shards(paths.features)
    if len(shards) != len(cand_shards):
        fail(f"expected {len(cand_shards)} feature shards, found {len(shards)}")
    rows = sum(pq.ParquetFile(p).metadata.num_rows for p in shards)
    stats = {"split": args.split, "rows": rows, "features": FEATURE_COLUMNS, "n_features": len(FEATURE_COLUMNS),
             "competition_features": COMP_FEATURES, "dtype": "float32", "batch_size_final": batch,
             "disk_gb": disk_size_gb(shards), "device": str(device), "seconds": time.time() - t0,
             "gpu": gpu_report()}
    log(f"features: {rows:,} rows x {len(FEATURE_COLUMNS)} float32, {stats['disk_gb']:.2f} GB on disk")
    write_manifest(paths.features, stats)


if __name__ == "__main__":
    main()
